"""Turn a chess move into an ordered motion program and execute it.

The executor ties the pieces together: decompose -> assign -> expand handoffs -> map each
location to a board-frame point -> build the pick/place primitive steps. Then, unless in dry-run:
IK every waypoint -> map to the arm's command convention -> check joint limits -> min-jerk
trajectory -> stream to the arm.

Three modes:
  * dry-run (default): return the program, command nothing.
  * simulate: run the program on :class:`SimArm` instances (safety rule 4's sim pass), in
    URDF-zero joint angles (``model_to_urdf``) against the URDF joint limits.
  * live: gated. Refuses unless ``config.measured`` is true AND an IK -> servo joint mapping
    (``to_command``, docs/joint_conventions.md) is supplied.

The whole move is planned (IK + limits + trajectories) before anything moves, so an unreachable
or out-of-limit waypoint aborts with no motion. One arm moves at a time, and each arm returns to
its rest pose (where it started the move) before the next one goes, which also clears the pad
during a handoff. On any fault during motion every arm holds position (safety rule 3).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import chess
import numpy as np

from magnus_bot.config import ArmCorrection, Config, GraspChoice
from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.arm import Arm
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import model_to_urdf
from magnus_bot.kinematics.model import JointLimit
from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable, tilted_tip_target
from magnus_bot.motion.primitives import CLOSE, OPEN, MotionStep, Phase, pick_place_steps
from magnus_bot.motion.trajectory import plan_joint_trajectory_indexed
from magnus_bot.planning.assigner import HANDOFF, SINGLE, assign_transfers
from magnus_bot.planning.handoff import expand_handoff
from magnus_bot.planning.move_decomposer import BIN, PAD, SQUARE, STASH, Location, Transfer, decompose_move

log = logging.getLogger(__name__)

# Maps IK (model-convention) joint angles to the angles the arm is commanded in. Sim defaults to
# model_to_urdf; live must be given model -> servo once the URDF <-> servo step is verified
# on hardware (docs/joint_conventions.md).
JointMap = Callable[[dict[str, float]], dict[str, float]]

# Max |commanded - read| (rad) per arm joint after a trajectory before it counts as a fault.
# The gripper is excluded: it stalls on the piece before reaching its closed target.
VERIFY_TOLERANCE = 0.05
# The real fingertip must stay at least this high above the board at every waypoint (m): the jaw
# tips must never be driven into the table. Checked on the commanded tip minus the arm's measured
# sag (arms.z_correction), i.e. where the tip really goes.
TABLE_CLEARANCE = 0.003
# Before the gripper closes on a piece or lets go of one, the arm must be at the pose it was sent to
# (within VERIFY_TOLERANCE), waiting up to this long (s) for it to settle. 2026-10-25 live: the elbow
# stalled short of the h4 place pose, the gripper opened on schedule, and the rook fell onto the
# g-file. Now the arm stops and holds instead.
ARRIVE_TIMEOUT = 2.0
GRIPPER_PHASES = (Phase.GRIP, Phase.RELEASE)


class JointLimitError(ValueError):
    """A planned joint target lies outside its limit (raised before any motion)."""


class ExecutionFault(RuntimeError):
    """Motion failed partway; every arm was told to hold position."""


@dataclass(frozen=True)
class ArmSteps:
    """One arm's pick/place step sequence for a single transfer, in execution order."""

    arm_id: str
    transfer: Transfer
    steps: list[MotionStep]


@dataclass(frozen=True)
class ArmTrajectory:
    """Dense joint commands for one :class:`ArmSteps`, from rest pose back to rest pose."""

    arm_id: str
    transfer: Transfer
    samples: list[dict[str, float]]
    # (sample index, phase): the arm must have ARRIVED at samples[index] before streaming goes on,
    # because the next segment opens or closes the gripper there.
    checkpoints: tuple[tuple[int, str], ...] = ()
    grasp: str = ""  # which grasp was used (e.g. "rook @ h4", "rook default", "as given")


def location_to_xyz(loc: Location, config: Config, arm_id: str) -> np.ndarray:
    """Board-frame [x, y, z] of a location. BIN resolves to ``arm_id``'s own bin."""
    if loc.kind == SQUARE:
        assert loc.square is not None
        xy = board_geom.square_center(loc.square[0], loc.square[1], config.board.square_size)
        return np.array([xy[0], xy[1], config.arms.z_grasp])
    if loc.kind == PAD:
        p = config.handoff.pad
        return np.array([p.x, p.y, p.z])
    if loc.kind == STASH:
        p = config.handoff.promotion_stash
        return np.array([p.x, p.y, p.z])
    if loc.kind == BIN:
        if arm_id not in config.handoff.bins:
            raise KeyError(f"no bin configured for arm {arm_id!r}")
        p = config.handoff.bins[arm_id]
        return np.array([p.x, p.y, p.z])
    raise ValueError(f"unknown location kind: {loc.kind!r}")


def build_move_program(
    board: chess.Board,
    move: chess.Move,
    rmap: ReachabilityMap,
    config: Config,
) -> list[ArmSteps]:
    """Full ordered motion program for ``move``: a flat list of per-arm step sequences.

    Handoffs expand into two consecutive entries (pick_arm to pad, then place_arm from pad).
    """
    arms = set(config.arms.arms)
    program: list[ArmSteps] = []
    for assignment in assign_transfers(decompose_move(board, move), rmap, arms):
        if assignment.mode == SINGLE:
            program.append(_arm_steps(assignment.arm, assignment.transfer, config))  # type: ignore[arg-type]
        elif assignment.mode == HANDOFF:
            for leg in expand_handoff(assignment):
                program.append(_arm_steps(leg.arm, leg.transfer, config))
        else:  # pragma: no cover - guarded by assigner
            raise ValueError(f"unknown assignment mode: {assignment.mode!r}")
    return program


def _arm_steps(arm_id: str, transfer: Transfer, config: Config) -> ArmSteps:
    src = location_to_xyz(transfer.src, config, arm_id)
    dst = location_to_xyz(transfer.dst, config, arm_id)
    steps = pick_place_steps(src, dst, config.arms.z_clear)
    return ArmSteps(arm_id=arm_id, transfer=transfer, steps=steps)


def check_limits(joints: Mapping[str, float], limits: Mapping[str, JointLimit]) -> None:
    """Raise :class:`JointLimitError` naming every joint outside its limit (never clamp a plan)."""
    bad = [
        f"{n}={v:.3f} not in [{limits[n].lo:.3f}, {limits[n].hi:.3f}]"
        for n, v in joints.items()
        if not limits[n].lo <= v <= limits[n].hi
    ]
    if bad:
        raise JointLimitError(", ".join(bad))


GRASP_HEIGHT_PHASES = {  # steps that sit at the piece's grasp height, per end of the transfer
    "src": (Phase.DESCEND_PICK, Phase.GRIP),
    "dst": (Phase.DESCEND_PLACE, Phase.RELEASE),
}


def piece_name(symbol: str) -> str:
    """'R' / 'r' -> 'rook' (configs key pieces by name)."""
    return chess.piece_name(chess.Piece.from_symbol(symbol).piece_type)


def transfer_squares(transfer: Transfer) -> list[str]:
    """The board squares a transfer touches (src first), e.g. ['h4', 'h5']."""
    return [chess.square_name(chess.square(*loc.square)) for loc in (transfer.src, transfer.dst)
            if loc.kind == SQUARE and loc.square is not None]


def plan_joint_program(
    program: list[ArmSteps],
    config: Config,
    rest: Mapping[str, dict[str, float]],
    kinematics: SO101Kinematics,
    limits: Mapping[str, JointLimit],
    to_command: JointMap | None = None,
    hz: float = 30.0,
    pitch: float = 0.0,
    grip_depth: float | None = None,
    tuned_grasps: bool = False,
    side_offset: float = 0.0,
) -> list[ArmTrajectory]:
    """Joint trajectories for a whole program, each starting and ending at that arm's ``rest`` pose.

    ``pitch`` tilts the tool off vertical for every grasp waypoint (0 = top-down), holding the stem
    ``grip_depth`` back from the jaw tips (default: the pads; :func:`tilted_tip_target`), aimed
    ``side_offset`` to the arm's left of each square (the jaws open to one side of the aim). With
    ``tuned_grasps``, each transfer instead tries the piece's tuned grasps (configs/grasp.yaml:
    per-square overrides for its squares first, then the piece's default) and takes the first that
    plans at BOTH ends. Raises :class:`Unreachable` or :class:`JointLimitError` (with arm, transfer
    and phase) if a transfer cannot be commanded. Pure: nothing is sent to any arm.
    """
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    trajectories = []
    for arm_steps in program:
        home = dict(rest[arm_steps.arm_id])
        piece = piece_name(arm_steps.transfer.piece_symbol)
        choices = config.grasp_choices(piece, transfer_squares(arm_steps.transfer)) if tuned_grasps else []
        if not choices:
            choices = [GraspChoice(pitch, None, grip_depth, "as given", side_offset)]  # type: ignore[arg-type]
        first_error: Exception | None = None
        for choice in choices:
            try:
                waypoints, checkpoints = _arm_waypoints(arm_steps, config, home, placements[arm_steps.arm_id],
                                                        kinematics, limits, to_command, choice, piece)
                break
            except (Unreachable, JointLimitError) as exc:
                first_error = first_error or exc
        else:
            raise first_error  # type: ignore[misc]
        samples, at = plan_joint_trajectory_indexed(waypoints, config.arms.speed_scale, hz)
        trajectories.append(ArmTrajectory(arm_steps.arm_id, arm_steps.transfer, samples,
                                          tuple((at[i], phase) for i, phase in checkpoints), choice.label))
    return trajectories


def _arm_waypoints(arm_steps, config, home, placement, kinematics, limits, to_command, choice, piece):
    """Joint waypoints (rest first and last) for one transfer with one grasp ``choice``."""
    correction = config.arms.correction(arm_steps.arm_id)  # where the real arm lands vs the model
    at_grasp = {ph for end, loc in (("src", arm_steps.transfer.src), ("dst", arm_steps.transfer.dst))
                if loc.kind == SQUARE for ph in GRASP_HEIGHT_PHASES[end]}
    gripper = home["gripper"]
    waypoints = [home]
    checkpoints = []  # (waypoint index, phase) the arm must reach before the gripper acts
    for step in arm_steps.steps:
        if step.phase in GRIPPER_PHASES:
            checkpoints.append((len(waypoints) - 1, step.phase.value))
        if step.gripper == OPEN:
            gripper = config.arms.gripper_open
        elif step.gripper == CLOSE:
            gripper = config.gripper_closed_for(piece)
        where = f"{arm_steps.arm_id} {arm_steps.transfer.reason} {step.phase.value}"
        x, y = placement.board_to_arm(step.xyz[:2])
        z = choice.z if choice.z is not None and step.phase in at_grasp else step.xyz[2]
        aim = ArmCorrection(side=choice.side).apply(np.array([x, y, z]))  # the jaws open sideways
        target = correction.apply(tilted_tip_target(aim, choice.pitch, kinematics.model.tip_to_pad,
                                                    choice.grip_depth))
        tip_z = target[2] - correction.z  # where the real (sagging) tip goes
        if tip_z < TABLE_CLEARANCE:
            raise Unreachable(f"{where}: the fingertip would be {tip_z * 1000:.1f} mm above the board "
                              f"(< {TABLE_CLEARANCE * 1000:.0f} mm): the jaw tips would hit the table "
                              f"[{choice.label}]")
        try:
            model = kinematics.inverse(target, pitch=choice.pitch)
        except Unreachable as exc:
            raise Unreachable(f"{where}: {exc} [{choice.label}]") from exc
        command = to_command(model) if to_command else dict(model)
        command["gripper"] = gripper
        try:
            check_limits(command, limits)
        except JointLimitError as exc:
            raise JointLimitError(f"{where}: {exc} [{choice.label}]") from exc
        waypoints.append(command)
    waypoints.append(home)  # back to rest: clears the board (and the pad) for the other arm
    return waypoints, checkpoints


class Executor:
    """Executes motion programs: dry-run by default, sim on request, live behind gates."""

    def __init__(
        self,
        config: Config,
        arms: dict[str, Arm],
        rmap: ReachabilityMap,
        live: bool = False,
        simulate: bool = False,
        to_command: JointMap | None = None,
        limits: Mapping[str, JointLimit] | None = None,
        kinematics: SO101Kinematics | None = None,
        hz: float = 30.0,
        pitch: float = 0.0,
        grip_depth: float | None = None,
        tuned_grasps: bool = False,
        side_offset: float = 0.0,
    ) -> None:
        if live and simulate:
            raise ValueError("choose live or simulate, not both")
        self.pitch = pitch
        self.grip_depth = grip_depth
        self.tuned_grasps = tuned_grasps  # per-transfer grasps from configs/grasp.yaml
        self.side_offset = side_offset  # explicit mode: aim this far to the arm's left (m)
        self.config = config
        self.arms = arms
        self.rmap = rmap
        self.live = live
        self.simulate = simulate
        self.to_command = to_command or model_to_urdf
        self.kinematics = kinematics or SO101Kinematics()
        self.limits = limits or self.kinematics.model.limits
        self.hz = hz
        if live:
            # Safety rule 1: live needs measured geometry AND an explicit opt-in.
            config.require_measured()
            if to_command is None:
                raise RuntimeError(
                    "live needs the IK -> servo joint mapping (docs/joint_conventions.md); "
                    "IK angles are not servo angles"
                )
        else:
            config.warn_if_unmeasured()
        if simulate:
            real = sorted(a for a, arm in arms.items() if not isinstance(arm, SimArm))
            if real:
                raise TypeError(f"simulate=True with non-sim arms {real}")

    def execute_move(self, board: chess.Board, move: chess.Move) -> list[ArmSteps]:
        """Build the program and execute it. Dry-run returns the program without moving joints."""
        program = build_move_program(board, move, self.rmap, self.config)
        if not (self.live or self.simulate):
            return program  # dry-run: the program is the deliverable
        self.run(self.plan(program))
        return program

    def plan(self, program: list[ArmSteps]) -> list[ArmTrajectory]:
        """Plan every arm's joint trajectory for ``program`` from where the arms are now."""
        used = {s.arm_id for s in program}
        for arm_id in sorted(used):
            if not self.arms[arm_id].is_connected:
                raise RuntimeError(f"{arm_id}: not connected")
        rest = {arm_id: self.arms[arm_id].read_joints() for arm_id in used}
        return plan_joint_program(
            program, self.config, rest, self.kinematics, self.limits, self.to_command,
            self.hz, self.pitch, self.grip_depth, self.tuned_grasps, self.side_offset,
        )

    def run(self, trajectories: list[ArmTrajectory]) -> None:
        """Stream trajectories one arm at a time; on any fault, hold every arm and raise."""
        period = 1.0 / self.hz if self.live else 0.0  # sim runs as fast as it can
        try:
            for traj in trajectories:
                arm = self.arms[traj.arm_id]
                gates = dict(traj.checkpoints)
                deadline = time.monotonic()
                for i, command in enumerate(traj.samples):
                    arm.send_joints(command)
                    if period:
                        deadline += period
                        time.sleep(max(0.0, deadline - time.monotonic()))
                    if i in gates:  # the gripper acts next: the arm must really be there
                        self._arrive(arm, command, gates[i])
                        deadline = time.monotonic()
                self._verify(arm, traj.samples[-1])
        except Exception as exc:
            self._hold_all()
            raise ExecutionFault(f"motion aborted, arms holding: {exc}") from exc

    def _arrive(self, arm: Arm, target: dict[str, float], phase: str) -> None:
        """Wait (live: up to ARRIVE_TIMEOUT) until ``arm`` reads back ``target``; else fault."""
        give_up = time.monotonic() + (ARRIVE_TIMEOUT if self.live else 0.0)
        while True:
            actual = arm.read_joints()
            off = {n: round(math.degrees(actual[n] - v), 1) for n, v in target.items()
                   if n != "gripper" and not math.isclose(actual[n], v, abs_tol=VERIFY_TOLERANCE)}
            if not off:
                return
            if time.monotonic() >= give_up:
                raise ExecutionFault(f"{arm.arm_id} did not reach the pose before {phase} "
                                     f"(off by deg: {off}); the gripper was NOT moved")
            time.sleep(0.05)

    def _verify(self, arm: Arm, target: dict[str, float]) -> None:
        actual = arm.read_joints()
        off = {
            n: actual[n] - v
            for n, v in target.items()
            if n != "gripper" and not math.isclose(actual[n], v, abs_tol=VERIFY_TOLERANCE)
        }
        if off:
            raise ExecutionFault(f"{arm.arm_id} did not reach its target: {off}")

    def _hold_all(self) -> None:
        for arm in self.arms.values():
            try:
                if arm.is_connected:
                    arm.hold()
            except Exception:
                log.exception("%s: hold failed", arm.arm_id)

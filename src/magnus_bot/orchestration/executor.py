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

import dataclasses
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
# Choosing among grasps: a pose counts as reachable if no joint goes more than this past its servo's
# physical stop (rad). Under load the elbow stops ~1 deg BEFORE its nominal stop (2026-10-25: the h3
# drop-off planned ~2.4 deg past it came up 3.4 deg short, over the 2.9 deg arrival tolerance), so
# slack + 1 deg must stay under that tolerance. 1.6 keeps the rook's 60 deg on g (1.54 past).
STOP_SLACK = math.radians(1.6)
# When a tuned grasp cannot reach a square (e.g. h3-h6, beside follower_1's base), try steeper
# approaches with the same grip height on the piece, then the stem nearer the tips (10 mm).
EASE_STEP = math.radians(5.0)
EASE_MIN_PITCH = math.radians(15.0)
EASE_GRIP_DEPTH = 0.010
# Leaving rest and coming back to it, the arm passes through a point this far out from its base
# (m, along the line to the square) at the travel height: lift up, THEN back. A straight joint-space
# swing from the square to rest dipped across the board and knocked pieces over (2026-10-25).
VIA_REACH = 0.12
VIA_PITCHES = (math.radians(45.0), math.radians(30.0), math.radians(15.0), 0.0)
# After a move the arm PARKS high above the gap between its base and the board, instead of going
# back down to the (often low) pose it started from: that last swing down swept the edge pieces
# (2026-10-25). The next move starts from there. Highest height that fits the arm's limits wins.
PARK_REACH = 0.10  # m out from the base (~2 cm over the board edge on this rig)
PARK_HEIGHTS = (0.20, 0.18, 0.16, 0.14)
PARK_PITCHES = tuple(math.radians(d) for d in range(0, 91, 5))
# Before the gripper closes on a piece or lets go of one, the arm must be at the pose it was sent to
# (within VERIFY_TOLERANCE), waiting up to this long (s) for it to settle. 2026-10-25 live: the elbow
# stalled short of the h4 place pose, the gripper opened on schedule, and the rook fell onto the
# g-file. Now the arm stops and holds instead.
ARRIVE_TIMEOUT = 2.0
UP_TIMEOUT = 0.5  # s: at the top of a lift / retract, wait at most this long for the arm to read back
GRIPPER_PHASES = (Phase.GRIP, Phase.RELEASE)
# 2026-10-25: pieces were dragged along the board right after a pick and after a drop-off: the real
# shoulder lags the plan, so the sideways move began while the arm was still low. Now the arm must
# be UP (lift / retract read back) before moving sideways (moving on after ARRIVE_TIMEOUT rather
# than faulting), and it pauses after the gripper closes / opens so the piece is held / let go.
UP_PHASES = (Phase.LIFT, Phase.RETRACT)
GRIPPER_DWELL = 0.1  # s (0.3 looked stalled, operator 2026-10-25)
DWELL = "dwell"


class JointLimitError(ValueError):
    """A planned joint target lies outside its limit (raised before any motion)."""


class BeyondServoStop(JointLimitError):
    """A grasp would drive a joint past its servo's physical stop: try an easier grasp."""


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
# Segments arriving at these steps are long travel moves at travel height (the faster ones).
TRAVEL_PHASES = (Phase.HOVER, Phase.TRAVERSE)
SQUARE_PHASES = {  # every step over each end of the transfer (a square nudge moves all of them)
    "src": (Phase.HOVER, Phase.DESCEND_PICK, Phase.GRIP, Phase.LIFT),
    "dst": (Phase.TRAVERSE, Phase.DESCEND_PLACE, Phase.RELEASE, Phase.RETRACT),
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
    stops: Mapping[str, Mapping[str, JointLimit]] | None = None,
) -> list[ArmTrajectory]:
    """Joint trajectories for a whole program, each starting and ending at that arm's ``rest`` pose.

    ``pitch`` tilts the tool off vertical for every grasp waypoint (0 = top-down), holding the stem
    ``grip_depth`` back from the jaw tips (default: the pads; :func:`tilted_tip_target`), aimed
    ``side_offset`` to the arm's left of each square (the jaws open to one side of the aim). With
    ``tuned_grasps``, each transfer instead tries the piece's tuned grasps (configs/grasp.yaml:
    per-square overrides for its squares first, then the piece's default) and takes the first that
    plans at BOTH ends; a tuned grasp that cannot reach (URDF limits, the table, or ``stops``: per
    arm, the servos' physical stops in command angles) is eased to steeper tilts with the same grip
    height. Raises :class:`Unreachable` or :class:`JointLimitError` (with arm, transfer and phase) if
    a transfer cannot be commanded. Pure: nothing is sent to any arm.
    """
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    trajectories = []
    for arm_steps in program:
        home = dict(rest[arm_steps.arm_id])
        piece = piece_name(arm_steps.transfer.piece_symbol)
        choices = config.grasp_choices(piece, transfer_squares(arm_steps.transfer)) if tuned_grasps else []
        if choices:
            choices = [eased for c in choices for eased in _eased(c, kinematics.model.tip_to_pad)]
        else:
            choices = [GraspChoice(pitch, None, grip_depth, "as given", side_offset)]  # type: ignore[arg-type]
        arm_stops = (stops or {}).get(arm_steps.arm_id)
        first_error: Exception | None = None
        for choice in choices:
            try:
                waypoints, checkpoints, travel = _arm_waypoints(
                    arm_steps, config, home, placements[arm_steps.arm_id], kinematics, limits, to_command,
                    choice, piece, arm_stops)
                break
            except (Unreachable, JointLimitError) as exc:
                first_error = first_error or exc
        else:
            raise first_error  # type: ignore[misc]
        fast = config.arms.travel_speedup
        samples, at = plan_joint_trajectory_indexed(waypoints, config.arms.speed_scale, hz,
                                                    segment_scales=[fast if t else 1.0 for t in travel])
        trajectories.append(ArmTrajectory(arm_steps.arm_id, arm_steps.transfer, samples,
                                          tuple((at[i], phase) for i, phase in checkpoints), choice.label))
    return trajectories


def _eased(choice: GraspChoice, tip_to_pad: float) -> list[GraspChoice]:
    """``choice``, then steeper tilts (same grip height on the piece), then the same with the stem
    nearer the jaw tips (keeps the tips off the table when the tool is near vertical)."""
    pitches = []
    p = choice.pitch - EASE_STEP
    while p >= EASE_MIN_PITCH - 1e-9:
        pitches.append(p)
        p -= EASE_STEP
    out = [choice]
    for p in pitches:
        out.append(dataclasses.replace(choice, pitch=p,
                                       label=f"{choice.label}, eased to {math.degrees(p):.0f} deg"))
    if (choice.grip_depth if choice.grip_depth is not None else tip_to_pad) > EASE_GRIP_DEPTH:
        for p in [choice.pitch, *pitches]:
            out.append(dataclasses.replace(
                choice, pitch=p, grip_depth=EASE_GRIP_DEPTH,
                label=f"{choice.label}, eased to {math.degrees(p):.0f} deg, stem 10 mm from the tips"))
    return out


def _arm_waypoints(arm_steps, config, home, placement, kinematics, limits, to_command, choice, piece,
                   stops=None):
    """Joint waypoints (rest first and last) for one transfer with one grasp ``choice``."""
    correction = config.arms.correction(arm_steps.arm_id)  # where the real arm lands vs the model
    # Squares, the handoff pad and the capture bins all sit on the table: the piece is picked and
    # placed there at its own grasp height (the pad/bin z in handoff.yaml is only a fallback).
    at_grasp = {ph for end, loc in (("src", arm_steps.transfer.src), ("dst", arm_steps.transfer.dst))
                if loc.kind in (SQUARE, PAD, BIN) for ph in GRASP_HEIGHT_PHASES[end]}
    nudges = config.arms.square_nudge.get(arm_steps.arm_id, {})
    phase_nudge = {}  # phase -> (reach, side) for the square that step sits over
    for end, loc in (("src", arm_steps.transfer.src), ("dst", arm_steps.transfer.dst)):
        if loc.kind == SQUARE and loc.square is not None:
            nudge = nudges.get(chess.square_name(chess.square(*loc.square)))
            if nudge:
                phase_nudge.update({ph: nudge for ph in SQUARE_PHASES[end]})
    gripper = home["gripper"]
    waypoints = [home]
    checkpoints = []  # (waypoint index, phase) the arm must reach before the gripper acts

    def reachable(point: np.ndarray, pitches, grip: float) -> dict[str, float] | None:
        """Command for the tip at ``point`` (arm frame, corrected) at the first pitch that fits."""
        for p in pitches:
            try:
                cmd = (to_command or dict)(kinematics.inverse(point, pitch=p))
                cmd["gripper"] = grip
                check_limits(cmd, limits)
            except (Unreachable, JointLimitError):
                continue
            if not stops or all(stops[n].lo - STOP_SLACK <= v <= stops[n].hi + STOP_SLACK
                                for n, v in cmd.items() if n in stops):
                return cmd
        return None

    def park(grip: float) -> dict[str, float] | None:
        """High above the arm's own board edge, on its centre line; None if nothing fits."""
        for z in PARK_HEIGHTS:
            cmd = reachable(correction.apply(np.array([PARK_REACH, 0.0, z])), PARK_PITCHES, grip)
            if cmd is not None:
                return cmd
        return None

    def via(xy_board: np.ndarray, grip: float) -> dict[str, float] | None:
        """Up at travel height near the arm's own board edge, on the line to ``xy_board``; None if no
        tilt reaches it (the move then goes without it)."""
        xy = placement.board_to_arm(xy_board)
        point = correction.apply(np.append(xy / max(np.hypot(*xy), 1e-9) * VIA_REACH, config.arms.z_clear))
        for p in (choice.pitch, *VIA_PITCHES):
            try:
                cmd = (to_command or dict)(kinematics.inverse(point, pitch=p))
                cmd["gripper"] = grip
                check_limits(cmd, limits)
            except (Unreachable, JointLimitError):
                continue
            if not stops or all(stops[n].lo - STOP_SLACK <= v <= stops[n].hi + STOP_SLACK
                                for n, v in cmd.items() if n in stops):
                return cmd
        return None

    travel = []  # per segment (the one arriving at each waypoint after rest): a long travel move?
    if (start := via(arm_steps.steps[0].xyz[:2], gripper)) is not None:
        waypoints.append(start)  # out of rest: up and over before reaching across the board
        travel.append(True)
    for step in arm_steps.steps:
        if step.phase in GRIPPER_PHASES:  # the arm must be there before the gripper acts
            checkpoints.append((len(waypoints) - 1, step.phase.value))
        if step.phase in UP_PHASES:  # the step before (grip / release) ends with a pause
            checkpoints.append((len(waypoints) - 1, DWELL))
        if step.gripper == OPEN:
            gripper = config.arms.gripper_open
        elif step.gripper == CLOSE:
            gripper = config.gripper_closed_for(piece)
        where = f"{arm_steps.arm_id} {arm_steps.transfer.reason} {step.phase.value}"
        x, y = placement.board_to_arm(step.xyz[:2])
        z = choice.z if choice.z is not None and step.phase in at_grasp else step.xyz[2]
        side = choice.side + config.arms.grasp_side_offset.get(arm_steps.arm_id, 0.0)  # jaws open sideways
        reach_nudge, side_nudge = phase_nudge.get(step.phase, (0.0, 0.0))
        aim = ArmCorrection(reach=reach_nudge, side=side + side_nudge).apply(np.array([x, y, z]))
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
        past = {n: math.degrees(max(stops[n].lo - v, v - stops[n].hi)) for n, v in command.items()
                if stops and n in stops and not stops[n].lo - STOP_SLACK <= v <= stops[n].hi + STOP_SLACK}
        if past:
            detail = ", ".join(f"{n} {d:.1f} deg" for n, d in past.items())
            raise BeyondServoStop(f"{where}: past the servo stop by {detail} [{choice.label}]")
        waypoints.append(command)
        travel.append(step.phase in TRAVEL_PHASES)
        if step.phase in UP_PHASES:  # up first: no sideways move before the arm is really up
            checkpoints.append((len(waypoints) - 1, step.phase.value))
    if (high := park(gripper)) is not None:
        waypoints.append(high)  # already up: back and further up to park, clear of the board
        travel.append(True)
        return waypoints, checkpoints, travel
    if (end := via(arm_steps.steps[-1].xyz[:2], gripper)) is not None:
        waypoints.append(end)  # after the place: already up; now back toward the arm, still high
        travel.append(True)
    waypoints.append(home)  # back to rest: clears the board (and the pad) for the other arm
    travel.append(True)
    return waypoints, checkpoints, travel


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
        stops: Mapping[str, Mapping[str, JointLimit]] | None = None,
    ) -> None:
        if live and simulate:
            raise ValueError("choose live or simulate, not both")
        self.pitch = pitch
        self.grip_depth = grip_depth
        self.tuned_grasps = tuned_grasps  # per-transfer grasps from configs/grasp.yaml
        self.side_offset = side_offset  # explicit mode: aim this far to the arm's left (m)
        self.stops = stops  # per arm: servo stops in command angles, to choose reachable grasps
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
            self.hz, self.pitch, self.grip_depth, self.tuned_grasps, self.side_offset, self.stops,
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
                    if i in gates:
                        if gates[i] == DWELL:
                            if self.live:
                                time.sleep(GRIPPER_DWELL)  # let the jaws finish closing / opening
                        else:  # the gripper acts next, or the arm moves sideways next
                            self._arrive(arm, command, gates[i], strict=gates[i] not in
                                         {ph.value for ph in UP_PHASES})
                        deadline = time.monotonic()
                self._verify(arm, traj.samples[-1])
        except Exception as exc:
            self._hold_all()
            raise ExecutionFault(f"motion aborted, arms holding: {exc}") from exc

    def _arrive(self, arm: Arm, target: dict[str, float], phase: str, strict: bool = True) -> None:
        """Wait (live: up to ARRIVE_TIMEOUT) until ``arm`` reads back ``target``; else fault, or (not
        ``strict``: the lift / retract) carry on, since it is as high as it will get."""
        give_up = time.monotonic() + ((ARRIVE_TIMEOUT if strict else UP_TIMEOUT) if self.live else 0.0)
        while True:
            actual = arm.read_joints()
            off = {n: round(math.degrees(actual[n] - v), 1) for n, v in target.items()
                   if n != "gripper" and not math.isclose(actual[n], v, abs_tol=VERIFY_TOLERANCE)}
            if not off:
                return
            if time.monotonic() >= give_up:
                if not strict:
                    log.warning("%s: %s settled %s deg off; moving on", arm.arm_id, phase, off)
                    return
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

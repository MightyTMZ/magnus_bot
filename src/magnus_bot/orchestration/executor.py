"""Turn a chess move into an ordered motion program and (dry-run) execute it.

The executor ties the pieces together: decompose -> assign -> expand handoffs -> map each
location to a board-frame point -> build the pick/place primitive steps. In dry-run (the
default) it records the program without commanding joints. Live execution (joint IK +
trajectories + real arms) is M1 and is gated: it refuses unless ``config.measured`` is true and
``live=True`` was explicitly requested.

One arm moves at a time; a handoff interlocks on the pad (see planning.handoff).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from magnus_bot.config import Config
from magnus_bot.geometry import board as board_geom
from magnus_bot.hardware.arm import Arm
from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.motion.primitives import MotionStep, pick_place_steps
from magnus_bot.planning.assigner import HANDOFF, SINGLE, assign_transfers
from magnus_bot.planning.handoff import expand_handoff
from magnus_bot.planning.move_decomposer import BIN, PAD, SQUARE, STASH, Location, Transfer, decompose_move


@dataclass(frozen=True)
class ArmSteps:
    """One arm's pick/place step sequence for a single transfer, in execution order."""

    arm_id: str
    transfer: Transfer
    steps: list[MotionStep]


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


class Executor:
    """Executes motion programs. Dry-run by default; live is gated and unimplemented (M1)."""

    def __init__(
        self,
        config: Config,
        arms: dict[str, Arm],
        rmap: ReachabilityMap,
        live: bool = False,
    ) -> None:
        self.config = config
        self.arms = arms
        self.rmap = rmap
        self.live = live
        if live:
            # Safety rule 1: live needs measured geometry AND an explicit opt-in.
            config.require_measured()
        else:
            config.warn_if_unmeasured()

    def execute_move(self, board: chess.Board, move: chess.Move) -> list[ArmSteps]:
        """Build the program and execute it. Dry-run returns the program without moving joints."""
        program = build_move_program(board, move, self.rmap, self.config)
        if not self.live:
            return program  # dry-run: the program is the deliverable
        # TODO(M1): for each ArmSteps, IK each waypoint -> min-jerk joint trajectory ->
        # arm.send_joints(...) with one arm moving at a time and the pad interlock; verify after.
        raise NotImplementedError("live joint execution lands in milestone M1")

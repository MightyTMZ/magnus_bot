"""The pick-and-place primitive: the fixed phase sequence for moving one piece.

Every single-arm transfer is the same eight phases (CLAUDE.md "Design decisions"):

    HOVER over src (z_clear) -> DESCEND to src -> GRIP -> LIFT to z_clear ->
    TRAVERSE to above dst (z_clear) -> DESCEND to dst -> RELEASE -> RETRACT to z_clear

This module emits the ordered :class:`MotionStep` waypoints in Cartesian board-frame
coordinates and gripper commands. Converting waypoints to joint trajectories is
:mod:`magnus_bot.motion.trajectory` (+ IK); mapping a :class:`~magnus_bot.planning.Location`
to a concrete xyz is the executor's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Phase(Enum):
    HOVER = "hover"
    DESCEND_PICK = "descend_pick"
    GRIP = "grip"
    LIFT = "lift"
    TRAVERSE = "traverse"
    DESCEND_PLACE = "descend_place"
    RELEASE = "release"
    RETRACT = "retract"


# Gripper intent for a step: hold current, open, or close.
HOLD = "hold"
OPEN = "open"
CLOSE = "close"


@dataclass(frozen=True)
class MotionStep:
    """One waypoint: move the fingertip to ``xyz`` (board frame, meters) and act on the gripper."""

    phase: Phase
    xyz: np.ndarray
    gripper: str  # HOLD | OPEN | CLOSE


def pick_place_steps(src_xyz: np.ndarray, dst_xyz: np.ndarray, z_clear: float) -> list[MotionStep]:
    """The eight-phase pick/place sequence moving a piece from ``src_xyz`` to ``dst_xyz``.

    ``src_xyz``/``dst_xyz`` are the grasp/release points (piece top or grasp height) in the board
    frame; ``z_clear`` is the safe traverse height (> tallest piece + margin).
    """
    src = np.asarray(src_xyz, dtype=float)
    dst = np.asarray(dst_xyz, dtype=float)
    if z_clear <= max(src[2], dst[2]):
        raise ValueError(f"z_clear ({z_clear}) must be above both grasp heights {src[2]}, {dst[2]}")

    above_src = np.array([src[0], src[1], z_clear])
    above_dst = np.array([dst[0], dst[1], z_clear])
    return [
        MotionStep(Phase.HOVER, above_src, OPEN),
        MotionStep(Phase.DESCEND_PICK, src, HOLD),
        MotionStep(Phase.GRIP, src, CLOSE),
        MotionStep(Phase.LIFT, above_src, HOLD),
        MotionStep(Phase.TRAVERSE, above_dst, HOLD),
        MotionStep(Phase.DESCEND_PLACE, dst, HOLD),
        MotionStep(Phase.RELEASE, dst, OPEN),
        MotionStep(Phase.RETRACT, above_dst, HOLD),
    ]

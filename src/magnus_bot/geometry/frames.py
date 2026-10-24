"""Arm placements in the board frame and board<->arm coordinate transforms.

Geometry (see docs/frames.md), both arms sit OFF the board beyond opposite short edges,
each facing along a rank:

  * follower_1: beyond the h-file edge, closest to h4/h5. Base at (8s + d, 4s), facing -x
    (toward the a-file). From its view LEFT = h1, RIGHT = h8.
  * follower_2: beyond the a-file edge, closest to a4/a5. Base at (-d, 4s), facing +x
    (toward the h-file). From its view LEFT = a8, RIGHT = a1.

where s = square size and d = base-to-board-edge gap (both measured; placeholders for now).

Arm frame A_i: origin on the base yaw axis, +x forward (``facing``), +y left
(``perp_left(facing)``), +z up. z is shared with the board frame (both +z up), so only the
(x, y) plane is transformed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from magnus_bot.geometry import board
from magnus_bot.geometry.transforms import perp_left, unit

BOARD_SIZE = board.BOARD_SIZE


@dataclass(frozen=True)
class ArmPlacement:
    """Where an arm's base sits in frame B and which way it faces.

    ``facing`` and ``left`` are unit vectors in the board-frame (x, y) plane.
    """

    arm_id: str
    base_xy: np.ndarray
    facing: np.ndarray

    @property
    def left(self) -> np.ndarray:
        """Unit vector pointing to the arm's left (``facing`` rotated +90 degrees)."""
        return perp_left(self.facing)

    def board_to_arm(self, xy: np.ndarray) -> np.ndarray:
        """Board-frame [x, y] -> arm-frame [x_forward, y_left]."""
        rel = np.asarray(xy, dtype=float) - self.base_xy
        return np.array([rel @ self.facing, rel @ self.left], dtype=float)

    def arm_to_board(self, xy_arm: np.ndarray) -> np.ndarray:
        """Arm-frame [x_forward, y_left] -> board-frame [x, y]."""
        xy_arm = np.asarray(xy_arm, dtype=float)
        return self.base_xy + xy_arm[0] * self.facing + xy_arm[1] * self.left

    def distance_to(self, xy: np.ndarray) -> float:
        """Planar distance from the base to a board-frame point (used by reach heuristics)."""
        return float(np.linalg.norm(np.asarray(xy, dtype=float) - self.base_xy))


def arm_placements(square_size: float, gap: float | Mapping[str, float]) -> dict[str, ArmPlacement]:
    """The two follower placements for a given square size ``s`` and edge gap(s) ``d``.

    ``gap`` is either a single distance applied to both arms, or a per-arm mapping
    ``{arm_id: gap}`` (the arms are not equidistant from the board on the measured rig). See the
    module docstring for the geometry these encode (and the unit tests that pin it).
    """
    s = float(square_size)
    if isinstance(gap, Mapping):
        d1, d2 = float(gap["follower_1"]), float(gap["follower_2"])
    else:
        d1 = d2 = float(gap)
    mid = (BOARD_SIZE / 2) * s  # y of the board's mid-line (rank 4/5 boundary): 4s
    return {
        "follower_1": ArmPlacement(
            arm_id="follower_1",
            base_xy=np.array([BOARD_SIZE * s + d1, mid], dtype=float),
            facing=unit(np.array([-1.0, 0.0])),
        ),
        "follower_2": ArmPlacement(
            arm_id="follower_2",
            base_xy=np.array([-d2, mid], dtype=float),
            facing=unit(np.array([1.0, 0.0])),
        ),
    }

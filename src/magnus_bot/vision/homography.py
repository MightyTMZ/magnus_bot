"""STUB (M4): map between camera pixels and board-frame squares.

A homography computed from the four board corners (or an ArUco/checker calibration target)
warps the overhead image to a top-down board, so each pixel maps to a (file, rank) square and
back. Used for move verification and physical move input.
"""

from __future__ import annotations

import numpy as np


class BoardHomography:
    """STUB (M4): planar homography between image pixels and board-frame coordinates."""

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self.matrix = matrix  # 3x3 image<-board homography once calibrated

    @classmethod
    def from_corners(cls, image_corners: np.ndarray, square_size: float) -> BoardHomography:
        # TODO(M4): cv2.getPerspectiveTransform from the four board corners in the image to
        # their board-frame coordinates; store the 3x3 matrix.
        raise NotImplementedError("BoardHomography lands in milestone M4")

    def pixel_to_square(self, pixel: np.ndarray) -> tuple[int, int]:
        raise NotImplementedError("BoardHomography lands in milestone M4")

    def square_to_pixel(self, file: int, rank: int) -> np.ndarray:
        raise NotImplementedError("BoardHomography lands in milestone M4")

"""Small, dependency-light transform helpers (SI: meters, radians).

Kept deliberately tiny and pure so the frame math in :mod:`frames` and the
analytic IK in :mod:`magnus_bot.kinematics` share one tested implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Vec2 = np.ndarray  # shape (2,)


def deg2rad(deg: float) -> float:
    """Degrees -> radians. Degrees appear only at the hardware edge; keep conversions here."""
    return float(np.deg2rad(deg))


def rad2deg(rad: float) -> float:
    """Radians -> degrees."""
    return float(np.rad2deg(rad))


def unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit length. Raises on a zero vector."""
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return np.asarray(v, dtype=float) / n


def rot2d(theta: float) -> np.ndarray:
    """2x2 rotation matrix for a counter-clockwise rotation by ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate(v: np.ndarray, theta: float) -> np.ndarray:
    """Rotate 2-vector ``v`` counter-clockwise by ``theta`` radians."""
    return rot2d(theta) @ np.asarray(v, dtype=float)


def perp_left(v: np.ndarray) -> np.ndarray:
    """"Left" of a heading: ``v`` rotated +90 degrees, i.e. (x, y) -> (-y, x).

    This is the single source of truth for the project's definition of left/right
    (see docs/frames.md). An arm facing direction ``f`` has its +y (left) axis along
    ``perp_left(f)``.
    """
    v = np.asarray(v, dtype=float)
    return np.array([-v[1], v[0]], dtype=float)


@dataclass(frozen=True)
class Pose2D:
    """A rigid placement in the plane: translation (x, y) and heading ``theta``.

    Transforms a point from the *local* frame (where +x is along ``theta`` and +y is
    90 degrees to the left) into the parent frame, and back.
    """

    x: float
    y: float
    theta: float

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def local_to_parent(self, p: np.ndarray) -> np.ndarray:
        """Map a point expressed in this frame's local axes into the parent frame."""
        return rot2d(self.theta) @ np.asarray(p, dtype=float) + self.translation

    def parent_to_local(self, p: np.ndarray) -> np.ndarray:
        """Map a parent-frame point into this frame's local axes."""
        return rot2d(-self.theta) @ (np.asarray(p, dtype=float) - self.translation)

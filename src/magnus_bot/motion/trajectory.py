"""Min-jerk interpolation between waypoints.

The scalar min-jerk profile is pure and tested now; the full joint-space planner that respects
per-joint velocity limits and the configured speed scale is M1 (it needs IK to turn Cartesian
waypoints into joint waypoints first).
"""

from __future__ import annotations

import numpy as np


def min_jerk_scalar(t_norm: float) -> float:
    """Min-jerk easing s(t) for normalized time ``t_norm`` in [0, 1].

    s(t) = 10 t^3 - 15 t^4 + 6 t^5, with s(0)=0, s(1)=1 and zero velocity/acceleration at both
    ends. Inputs are clamped to [0, 1].
    """
    t = min(1.0, max(0.0, t_norm))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def min_jerk_interp(start: np.ndarray, end: np.ndarray, num_steps: int) -> list[np.ndarray]:
    """Min-jerk path from ``start`` to ``end`` as ``num_steps`` points (inclusive of both ends)."""
    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2, got {num_steps}")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return [start + (end - start) * min_jerk_scalar(i / (num_steps - 1)) for i in range(num_steps)]


def plan_joint_trajectory(
    joint_waypoints: list[dict[str, float]],
    speed_scale: float,
    hz: float = 30.0,
) -> list[dict[str, float]]:
    """STUB (M1): dense min-jerk joint trajectory honoring velocity limits and ``speed_scale``.

    TODO(M1): between consecutive joint waypoints, time-scale each segment by the slowest joint's
    velocity limit / speed_scale, then sample at ``hz`` using :func:`min_jerk_interp` per joint.
    """
    raise NotImplementedError("joint-space trajectory planning lands in milestone M1")

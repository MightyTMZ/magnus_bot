"""Min-jerk interpolation between waypoints.

:func:`plan_joint_trajectory` turns sparse joint waypoints (from IK) into a dense, fixed-rate
command stream: each segment is a min-jerk move whose duration is set by the joint that needs
the longest, so all joints start and stop together and none exceeds its velocity limit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from magnus_bot.kinematics.model import MAX_JOINT_VELOCITY
from magnus_bot.motion.safety import clamp_speed_scale

# Peak of ds/dt for the min-jerk profile (at t = 0.5): 30t^2 - 60t^3 + 30t^4 -> 15/8.
MIN_JERK_PEAK = 1.875


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


def segment_duration(
    start: Mapping[str, float],
    end: Mapping[str, float],
    max_velocity: Mapping[str, float],
    speed_scale: float,
) -> float:
    """Seconds a min-jerk move from ``start`` to ``end`` must take so no joint is too fast.

    Each joint ``j`` may move at most ``max_velocity[j] * speed_scale``. ``start`` and ``end``
    have the same joint keys, and ``max_velocity`` covers all of them. Min-jerk peaks at
    ``MIN_JERK_PEAK * distance / duration`` (at mid-segment), so each joint needs at least
    ``MIN_JERK_PEAK * |delta| / v``; the slowest joint sets the segment. A segment with no
    motion takes 0 s.
    """
    return max(
        (MIN_JERK_PEAK * abs(end[n] - start[n]) / (max_velocity[n] * speed_scale) for n in start),
        default=0.0,
    )


def plan_joint_trajectory(
    joint_waypoints: list[dict[str, float]],
    speed_scale: float,
    hz: float = 30.0,
    max_velocity: Mapping[str, float] | None = None,
) -> list[dict[str, float]]:
    """Dense min-jerk joint trajectory through ``joint_waypoints``, sampled at ``hz``.

    Consecutive samples are ``1 / hz`` seconds apart. Every waypoint appears exactly once in the
    output, and the first and last samples equal the first and last waypoints.
    """
    if not joint_waypoints:
        raise ValueError("need at least one waypoint")
    if hz <= 0:
        raise ValueError(f"hz must be > 0, got {hz}")
    scale = clamp_speed_scale(speed_scale)
    limits = MAX_JOINT_VELOCITY if max_velocity is None else max_velocity

    names = list(joint_waypoints[0])
    missing = [n for n in names if n not in limits]
    if missing:
        raise ValueError(f"no velocity limit for joints {missing}")

    samples = [dict(joint_waypoints[0])]
    for start, end in zip(joint_waypoints, joint_waypoints[1:], strict=False):
        if set(end) != set(names):
            raise ValueError(f"waypoint joints {sorted(end)} differ from {sorted(names)}")
        duration = segment_duration(start, end, limits, scale)
        num_steps = max(2, math.ceil(duration * hz) + 1)
        path = min_jerk_interp(
            np.array([start[n] for n in names]), np.array([end[n] for n in names]), num_steps
        )
        samples.extend(dict(zip(names, (float(v) for v in p), strict=True)) for p in path[1:])
    return samples

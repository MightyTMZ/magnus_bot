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
    return plan_joint_trajectory_indexed(joint_waypoints, speed_scale, hz, max_velocity)[0]


def plan_joint_trajectory_indexed(
    joint_waypoints: list[dict[str, float]],
    speed_scale: float,
    hz: float = 30.0,
    max_velocity: Mapping[str, float] | None = None,
    segment_scales: list[float] | None = None,
    pass_through: set[int] | frozenset[int] = frozenset(),
) -> tuple[list[dict[str, float]], list[int]]:
    """:func:`plan_joint_trajectory`, plus the sample index at which each waypoint is reached.

    ``segment_scales`` (one per segment) multiplies ``speed_scale`` for that segment, e.g. faster
    long travel moves than the careful moves near the pieces. The product is still clamped to 1.
    Waypoints in ``pass_through`` (by index) are flowed through without stopping: a run of them is
    one min-jerk motion along the joint-space polyline, timed as the sum of its segments."""
    if not joint_waypoints:
        raise ValueError("need at least one waypoint")
    if hz <= 0:
        raise ValueError(f"hz must be > 0, got {hz}")
    if segment_scales is not None and len(segment_scales) != len(joint_waypoints) - 1:
        raise ValueError(f"need {len(joint_waypoints) - 1} segment scales, got {len(segment_scales)}")
    factors = segment_scales or [1.0] * (len(joint_waypoints) - 1)
    limits = MAX_JOINT_VELOCITY if max_velocity is None else max_velocity

    names = list(joint_waypoints[0])
    missing = [n for n in names if n not in limits]
    if missing:
        raise ValueError(f"no velocity limit for joints {missing}")

    for w in joint_waypoints[1:]:
        if set(w) != set(names):
            raise ValueError(f"waypoint joints {sorted(w)} differ from {sorted(names)}")
    samples = [dict(joint_waypoints[0])]
    indices = [0]
    last = len(joint_waypoints) - 1
    i = 0
    while i < last:
        j = i + 1
        while j < last and j in pass_through:
            j += 1
        points = [np.array([w[n] for n in names]) for w in joint_waypoints[i:j + 1]]
        times = [segment_duration(joint_waypoints[k], joint_waypoints[k + 1], limits,
                                  clamp_speed_scale(speed_scale * factors[k])) for k in range(i, j)]
        total = sum(times)
        bounds = np.cumsum([0.0, *times]) / total if total > 0 else np.linspace(0.0, 1.0, len(points))
        num_steps = max(2, math.ceil(total * hz) + 1)
        reached = [False] * len(points)
        for step in range(1, num_steps):
            s = min_jerk_scalar(step / (num_steps - 1))
            k = min(int(np.searchsorted(bounds, s, side="right")) - 1, len(points) - 2)
            span = bounds[k + 1] - bounds[k]
            local = 0.0 if span <= 0 else min(max((s - bounds[k]) / span, 0.0), 1.0)
            point = points[k] + (points[k + 1] - points[k]) * local
            samples.append(dict(zip(names, (float(v) for v in point), strict=True)))
            for m in range(1, len(points)):  # the sample index at which each waypoint is passed
                if not reached[m] and s >= bounds[m] - 1e-12:
                    reached[m] = True
                    indices.append(len(samples) - 1)
        while len(indices) < j + 1:  # rounding: any waypoint not yet marked is reached at the end
            indices.append(len(samples) - 1)
        i = j
    return samples, indices

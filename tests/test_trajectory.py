"""Joint-space min-jerk trajectory planning (motion/trajectory.plan_joint_trajectory)."""

import numpy as np
import pytest

from magnus_bot.motion.trajectory import (
    plan_joint_trajectory,
    plan_joint_trajectory_indexed,
    segment_duration,
)

HZ = 30.0
VMAX = {"a": 1.0, "b": 2.0}


def _peak_speeds(samples, hz=HZ):
    """Largest finite-difference speed of each joint across the sampled trajectory."""
    names = list(samples[0])
    q = np.array([[s[n] for n in names] for s in samples])
    return dict(zip(names, np.abs(np.diff(q, axis=0)).max(axis=0) * hz, strict=True))


def test_endpoints_and_waypoints_are_hit_exactly():
    wps = [{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": -0.5}, {"a": 0.2, "b": 0.4}]
    out = plan_joint_trajectory(wps, speed_scale=1.0, hz=HZ, max_velocity=VMAX)
    assert out[0] == pytest.approx(wps[0])
    assert out[-1] == pytest.approx(wps[-1])
    assert any(s == pytest.approx(wps[1]) for s in out)


@pytest.mark.parametrize("scale", [1.0, 0.3])
def test_no_joint_exceeds_its_scaled_limit(scale):
    wps = [{"a": 0.0, "b": 0.0}, {"a": 1.2, "b": 3.0}, {"a": -0.4, "b": 2.9}]
    out = plan_joint_trajectory(wps, speed_scale=scale, hz=HZ, max_velocity=VMAX)
    for name, peak in _peak_speeds(out).items():
        assert peak <= VMAX[name] * scale + 1e-9, f"{name} too fast: {peak:.3f} rad/s"


def test_limiting_joint_is_not_needlessly_slow():
    # A long single-joint move should get close to the limit, not crawl far below it.
    wps = [{"a": 0.0, "b": 0.0}, {"a": 2.0, "b": 0.0}]
    out = plan_joint_trajectory(wps, speed_scale=1.0, hz=HZ, max_velocity=VMAX)
    assert _peak_speeds(out)["a"] >= 0.9 * VMAX["a"]


def test_lower_speed_scale_takes_longer():
    start, end = {"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}
    fast = segment_duration(start, end, VMAX, 1.0)
    slow = segment_duration(start, end, VMAX, 0.3)
    assert slow == pytest.approx(fast / 0.3)


def test_slowest_joint_sets_the_duration():
    # b moves twice as far as a but is allowed twice the speed: both need the same time.
    start = {"a": 0.0, "b": 0.0}
    only_a = segment_duration(start, {"a": 1.0, "b": 0.0}, VMAX, 1.0)
    both = segment_duration(start, {"a": 1.0, "b": 2.0}, VMAX, 1.0)
    assert both == pytest.approx(only_a)


def test_stationary_segment_is_safe():
    wps = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
    out = plan_joint_trajectory(wps, speed_scale=1.0, hz=HZ, max_velocity=VMAX)
    assert all(s == pytest.approx(wps[0]) for s in out)


def test_rejects_mismatched_joints_and_missing_limits():
    with pytest.raises(ValueError):
        plan_joint_trajectory([{"a": 0.0}, {"b": 1.0}], speed_scale=1.0, max_velocity=VMAX)
    with pytest.raises(ValueError):
        plan_joint_trajectory([{"c": 0.0}, {"c": 1.0}], speed_scale=1.0, max_velocity=VMAX)


def test_a_faster_segment_takes_proportionally_less_time():
    a = {"shoulder_pan": 0.0, "gripper": 0.0}
    b = {"shoulder_pan": 1.0, "gripper": 0.0}
    _, plain = plan_joint_trajectory_indexed([a, b, a], 0.3, 30.0)
    _, quick = plan_joint_trajectory_indexed([a, b, a], 0.3, 30.0, segment_scales=[1.25, 1.0])
    assert (quick[1] - quick[0]) == pytest.approx((plain[1] - plain[0]) / 1.25, abs=1)  # first: 25% faster
    assert (quick[2] - quick[1]) == plain[2] - plain[1]  # second: unchanged
    with pytest.raises(ValueError, match="segment scales"):
        plan_joint_trajectory_indexed([a, b, a], 0.3, 30.0, segment_scales=[1.0])


def test_pass_through_waypoints_are_flowed_through_without_stopping():
    a = {"shoulder_pan": 0.0, "gripper": 0.0}
    b = {"shoulder_pan": 0.5, "gripper": 0.0}
    c = {"shoulder_pan": 1.0, "gripper": 0.0}
    stop, at_stop = plan_joint_trajectory_indexed([a, b, c], 0.3, 30.0)
    flow, at_flow = plan_joint_trajectory_indexed([a, b, c], 0.3, 30.0, pass_through={1})
    assert flow[-1]["shoulder_pan"] == pytest.approx(1.0) and len(at_flow) == 3
    # stopping at b means zero speed there; flowing through, it is still moving fast at b
    moving = abs(flow[at_flow[1] + 1]["shoulder_pan"] - flow[at_flow[1]]["shoulder_pan"])
    still = abs(stop[at_stop[1] + 1]["shoulder_pan"] - stop[at_stop[1]]["shoulder_pan"])
    assert moving > 10 * still
    assert abs(len(flow) - len(stop)) <= 2  # same total time, it just does not stop at b

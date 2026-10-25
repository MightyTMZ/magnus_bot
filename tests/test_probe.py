"""Reach probe: planning the held pose, and splitting a miss into motor and geometry error."""

import math

import numpy as np
import pytest

from magnus_bot.geometry.frames import arm_placements
from magnus_bot.kinematics.conventions import model_to_urdf, urdf_to_model
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable
from magnus_bot.orchestration.executor import JointLimitError
from magnus_bot.orchestration.probe import (
    ProbeReport,
    analyze,
    plan_probe,
    remove_parallax,
    split_mm,
    verdict,
)

SIZE = 0.056
KIN = SO101Kinematics()
F1 = arm_placements(SIZE, {"follower_1": 0.085, "follower_2": 0.067})["follower_1"]


def plan(square="g3", z=0.010, z2=None):
    return plan_probe(square, F1, KIN, model_to_urdf, KIN.model.limits, square_size=SIZE,
                      z=z, hover=0.06, pitch=0.0, gripper=10.0, z2=z2)


def grade(p, reached, camera_xy=None, camera_xy2=None):
    return analyze(p, reached, placement=F1, kinematics=KIN, to_model=urdf_to_model,
                   camera_xy=camera_xy, camera_xy2=camera_xy2)


def test_second_hold_is_straight_above_the_first():
    p = plan("g3", z2=0.040)
    low, high = KIN.forward(urdf_to_model(p.at)), KIN.forward(urdf_to_model(p.at2))
    assert np.allclose(low[:2], high[:2], atol=1e-6)
    assert high[2] - low[2] == pytest.approx(0.030, abs=1e-6)
    lift = p.raise_to_z2(0.2, 30.0)
    assert lift[0] == p.at and lift[-1] == p.at2
    assert p.up({**p.at, "gripper": 50.0}, 0.2, 30.0)[0] == p.at2  # goes home from the higher hold


def test_parallax_extrapolates_to_the_board_surface():
    true_xy = np.array([0.30, 0.15])
    shift_per_m = np.array([0.2, 1.1])  # apparent drift with height (a camera ray)
    low, high = true_xy + 0.010 * shift_per_m, true_xy + 0.040 * shift_per_m
    assert remove_parallax(low, high, 0.010, 0.040) == pytest.approx(true_xy)


def test_two_heights_recover_a_geometry_error_hidden_by_parallax():
    p = plan("g3", z2=0.040)
    reach_dir = (p.target_xy - F1.base_xy) / np.linalg.norm(p.target_xy - F1.base_xy)
    true_tip = p.target_xy - 0.025 * reach_dir  # really 25 mm short
    drift = np.array([0.0, 1.1])  # ~11 mm per 10 mm of height toward rank 8
    r = grade(p, dict(p.at), camera_xy=true_tip + 0.010 * drift, camera_xy2=true_tip + 0.040 * drift)
    assert r.camera_mm == pytest.approx((-25.0, 0.0), abs=1e-6)
    assert r.parallax_mm is not None and np.hypot(*r.parallax_mm) == pytest.approx(11.0)
    assert any("parallax removed" in line for line in r.lines())


def test_the_held_pose_puts_the_model_fingertip_on_the_square():
    p = plan("g3")
    tip = KIN.forward(urdf_to_model(p.at))
    assert np.allclose(F1.arm_to_board(tip[:2]), p.target_xy, atol=1e-6)
    assert tip[2] == pytest.approx(0.010, abs=1e-6)
    assert p.at["gripper"] == 10.0 and p.above["gripper"] == 10.0


def test_trajectories_start_and_end_at_rest():
    p = plan()
    rest = {**{n: 0.0 for n in p.at}, "gripper": 50.0}
    down, up = p.down(rest, 0.2, 30.0), p.up(rest, 0.2, 30.0)
    assert down[0] == rest and down[-1] == p.at
    assert up[0] == p.at and up[-1] == rest


def test_a_square_the_arm_cannot_reach_is_refused_before_moving():
    with pytest.raises((Unreachable, JointLimitError)):
        plan("a8")  # the far corner from follower_1


def test_offsets_split_into_reach_and_side():
    reach_dir = np.array([-1.0, 0.0])  # follower_1 reaches toward -x
    assert split_mm(np.array([-0.010, 0.0]), reach_dir) == pytest.approx((10.0, 0.0))  # 10 mm too far
    assert split_mm(np.array([0.020, 0.0]), reach_dir) == pytest.approx((-20.0, 0.0))  # 20 mm short
    # The arm faces -x, so its left is -y.
    assert split_mm(np.array([0.0, -0.005]), reach_dir) == pytest.approx((0.0, 5.0))


def test_a_perfect_reach_reports_no_error():
    p = plan()
    r = grade(p, dict(p.at), camera_xy=p.target_xy.copy())
    assert max(abs(e) for e in r.joint_error_deg.values()) == pytest.approx(0.0)
    assert r.model_mm == pytest.approx((0.0, 0.0), abs=1e-6)
    assert r.camera_mm == pytest.approx((0.0, 0.0), abs=1e-6)
    assert r.geometry_mm == pytest.approx((0.0, 0.0), abs=1e-6)


def test_motor_sag_shows_as_a_joint_gap_the_model_explains():
    p = plan()
    sagged = {**p.at, "shoulder_lift": p.at["shoulder_lift"] + math.radians(3)}
    tip = KIN.forward(urdf_to_model(sagged))
    camera = F1.arm_to_board(tip[:2])  # the camera sees exactly where the sagged arm is
    r = grade(p, sagged, camera_xy=camera)
    assert r.worst_joint == ("shoulder_lift", pytest.approx(3.0))
    assert r.model_mm[0] < -3  # the sag alone leaves the tip short of the square
    assert r.geometry_mm == pytest.approx((0.0, 0.0), abs=1e-6)  # nothing left for geometry


def test_a_tip_seen_off_the_model_is_a_geometry_error():
    p = plan()
    reach_dir = (p.target_xy - F1.base_xy) / np.linalg.norm(p.target_xy - F1.base_xy)
    r = grade(p, dict(p.at), camera_xy=p.target_xy - 0.020 * reach_dir)  # 20 mm short
    assert r.camera_mm == pytest.approx((-20.0, 0.0), abs=1e-6)
    assert r.geometry_mm == pytest.approx((-20.0, 0.0), abs=1e-6)
    assert any("GEOMETRY" in line and "-20 mm" in line for line in verdict([r]))


def test_an_unseen_fingertip_is_reported_not_guessed():
    r = grade(plan(), dict(plan().at), camera_xy=None)
    assert r.camera_mm is None and r.geometry_mm is None
    assert any("NOT seen" in line for line in r.lines())
    assert r.to_record()["camera_mm"] is None


def report(joint_deg=0.0, geometry=(0.0, 0.0), square="g3"):
    return ProbeReport(square, {"shoulder_lift": joint_deg}, (0.0, 0.0), 0.0, geometry, geometry)


def test_verdict_names_motors_and_geometry_separately():
    assert verdict([report()]) == [
        "motors OK: every joint within 1.5 deg of its command.",
        "geometry OK: on average within 5 mm, and every square within 10 mm.",
    ]
    lines = verdict([report(joint_deg=-4.0, geometry=(-15.0, 2.0))])
    assert lines[0].startswith("MOTORS: shoulder_lift ended -4.0 deg")
    assert lines[1].startswith("GEOMETRY: on average the tip lands -15 mm along the reach")
    assert verdict([]) == ["nothing was probed"]


def test_errors_that_cancel_on_average_are_still_reported():
    # The 2026-10-24 live run: sideways errors of -20 and +10 mm averaged to about zero.
    reports = [report(geometry=(-5.0, -20.0), square="g3"), report(geometry=(3.0, -2.0), square="h4"),
               report(geometry=(3.0, 4.0), square="g5"), report(geometry=(4.0, 10.0), square="g6")]
    lines = verdict(reports)
    assert not any(line.startswith("geometry OK") for line in lines)
    assert any(line.startswith("GEOMETRY VARIES: g3 is off by -5 mm reach, -20 mm side") for line in lines)


def test_probe_commands_corrected_heights_but_reports_against_the_command():
    p = plan_probe("g3", F1, KIN, model_to_urdf, KIN.model.limits, square_size=SIZE, z=0.010,
                   hover=0.06, pitch=0.0, gripper=10.0, z_correction=0.005)
    assert KIN.forward(urdf_to_model(p.at))[2] == pytest.approx(0.015)  # commanded 5 mm higher
    assert p.target_z == pytest.approx(0.010)  # the height we actually want
    assert grade(p, dict(p.at)).model_z_mm == pytest.approx(0.0, abs=1e-6)

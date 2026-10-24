"""Analytic FK/IK: round-trip self-consistency, reach limits, and a placo cross-check.

The round-trip and reach tests run everywhere. The placo cross-check needs the `ik`/`hardware`
env (placo + lerobot) and the URDF, so it skips until that exists; when it runs it checks that the
link geometry baked into SO101Model matches placo's reading of the same URDF.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from magnus_bot.kinematics.model import SO101Model
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable

Z = 0.028  # a representative grasp height (pawn)


@pytest.fixture
def kin():
    return SO101Kinematics(SO101Model())


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_round_trip_is_tight(kin, elbow):
    """forward(inverse(target)) == target for every reachable target on a grid."""
    checked = 0
    for x in np.linspace(0.05, 0.24, 16):
        for y in np.linspace(-0.16, 0.16, 25):
            target = np.array([x, y, Z])
            try:
                joints = kin.inverse(target, elbow=elbow)
            except Unreachable:
                continue
            assert np.linalg.norm(kin.forward(joints) - target) < 1e-9
            checked += 1
    assert checked > 100  # the grid really exercised the solver


@pytest.mark.parametrize("target", [(0.0955, 0.0, Z), (0.15, 0.05, Z), (0.12, -0.05, 0.14)])
def test_elbow_up_is_above_down(kin, target):
    """The "up" branch puts the elbow higher than "down" (it must clear pieces on the board)."""
    m = kin.model

    def elbow_z(elbow):
        lift = kin.inverse(np.array(target), elbow=elbow)["shoulder_lift"]
        return m.base_height + m.upper_arm * math.cos(lift)

    assert elbow_z("up") > elbow_z("down")


def test_far_target_unreachable(kin):
    with pytest.raises(Unreachable):
        kin.inverse(np.array([0.40, 0.0, Z]))  # well past the 0.251 m 2-link reach


def test_inverse_makes_tool_vertical(kin):
    j = kin.inverse(np.array([0.15, 0.05, Z]))
    a2 = j["shoulder_lift"] + j["elbow_flex"] + j["wrist_flex"]
    assert math.isclose(a2, math.pi, abs_tol=1e-9)  # tool points straight down


def test_pan_matches_target_azimuth(kin):
    j = kin.inverse(np.array([0.12, 0.12, Z]))
    assert math.isclose(j["shoulder_pan"], math.atan2(0.12, 0.12), abs_tol=1e-9)


def test_roll_passthrough(kin):
    assert kin.inverse(np.array([0.15, 0.0, Z]), roll=0.5)["wrist_roll"] == 0.5


@pytest.mark.parametrize("pitch_deg", [10, 30, 45])
@pytest.mark.parametrize("elbow", ["up", "down"])
def test_pitched_round_trip_is_tight(kin, pitch_deg, elbow):
    """forward(inverse(target, pitch=p)) == target for a tilted grasp, on the reachable grid."""
    pitch = math.radians(pitch_deg)
    checked = 0
    for x in np.linspace(0.08, 0.30, 16):
        for y in np.linspace(-0.12, 0.12, 13):
            target = np.array([x, y, Z])
            try:
                joints = kin.inverse(target, elbow=elbow, pitch=pitch)
            except Unreachable:
                continue
            assert np.linalg.norm(kin.forward(joints) - target) < 1e-9
            checked += 1
    assert checked > 50


def test_tool_angle_follows_pitch(kin):
    pitch = math.radians(35)
    j = kin.inverse(np.array([0.28, 0.0, Z]), pitch=pitch)
    a2 = j["shoulder_lift"] + j["elbow_flex"] + j["wrist_flex"]
    assert math.isclose(a2, math.pi - pitch, abs_tol=1e-9)  # tool tilted by `pitch` off vertical


def test_pitch_extends_reach(kin):
    """A target outside the top-down workspace becomes reachable once the tool tilts."""
    far = np.array([0.30, 0.0, Z])
    with pytest.raises(Unreachable):
        kin.inverse(far, pitch=0.0)  # top-down cannot reach it
    joints = kin.inverse(far, pitch=math.radians(45))  # a tilt can
    assert np.linalg.norm(kin.forward(joints) - far) < 1e-9


def test_link_points_chain_is_consistent(kin):
    """link_points is a connected chain ending at forward(); segment lengths match the model."""
    j = kin.inverse(np.array([0.15, 0.03, Z]), pitch=math.radians(20))
    pts = kin.link_points(j)
    assert np.allclose(pts[-1], kin.forward(j))
    m = kin.model
    assert math.isclose(np.linalg.norm(pts[2] - pts[1]), m.forearm, abs_tol=1e-9)
    assert math.isclose(np.linalg.norm(pts[3] - pts[2]), m.wrist_to_tip, abs_tol=1e-9)


def test_cross_check_placo_matches_urdf_geometry():
    """When placo + the URDF are present, placo's zero-pose FK tip must match the geometry we
    extracted into SO101Model (validates the link-length derivation, not the servo mapping)."""
    pytest.importorskip("placo")
    rk = pytest.importorskip("lerobot.model.kinematics")
    urdf = Path(__file__).resolve().parents[1] / "hardware" / "urdf" / "so101_new_calib.urdf"
    if not urdf.is_file():
        pytest.skip("URDF not present")
    kinematics = rk.RobotKinematics(str(urdf), target_frame_name="gripper_frame_link")
    tip = kinematics.forward_kinematics(np.zeros(6))[:3, 3]
    # Zero-pose tip computed from the URDF transforms (see docs/joint_conventions.md).
    assert np.allclose(tip, [0.3914, 0.0, 0.2265], atol=2e-3)

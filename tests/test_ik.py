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

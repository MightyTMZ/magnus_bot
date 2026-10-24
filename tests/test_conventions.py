"""IK model vs. the URDF chain: geometry, the model <-> URDF angle mapping, and the limits.

A small numpy FK walks hardware/urdf/so101_new_calib.urdf (no placo needed), so this pins the
mapping in kinematics/conventions.py and the lengths in SO101Model everywhere tests run.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from magnus_bot.kinematics.conventions import model_to_urdf, urdf_to_model
from magnus_bot.kinematics.model import ARM_JOINTS, SO101Model
from magnus_bot.kinematics.so101_ik import SO101Kinematics

URDF = Path(__file__).resolve().parents[1] / "hardware" / "urdf" / "so101_new_calib.urdf"
CHAIN = (*ARM_JOINTS, "gripper_frame_joint")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis, q):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k


@pytest.fixture(scope="module")
def urdf_fk():
    joints = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        origin, axis = j.find("origin"), j.find("axis")
        joints[j.get("name")] = (
            np.array(origin.get("xyz").split(), dtype=float),
            np.array(origin.get("rpy").split(), dtype=float),
            np.array(axis.get("xyz").split(), dtype=float) if axis is not None else np.zeros(3),
        )

    def fk(q):
        """Fingertip (gripper_frame_link) position in base_link for URDF joint angles ``q``."""
        t = np.eye(4)
        for name in CHAIN:
            xyz, rpy, axis = joints[name]
            step = np.eye(4)
            step[:3, :3], step[:3, 3] = _rpy(*rpy), xyz
            t = t @ step
            if axis.any():
                rot = np.eye(4)
                rot[:3, :3] = _axis_angle(axis, q.get(name, 0.0))
                t = t @ rot
        return t[:3, 3]

    pan_xyz = joints["shoulder_pan"][0]
    return fk, np.array([pan_xyz[0], pan_xyz[1], 0.0])  # arm frame origin: yaw axis, table level


def test_zero_pose_tip_matches_urdf(urdf_fk):
    fk, _ = urdf_fk
    assert np.allclose(fk({}), [0.3914, 0.0, 0.2265], atol=2e-4)  # same value as the placo check


def test_model_fk_matches_urdf_at_zero_roll(urdf_fk):
    fk, origin = urdf_fk
    kin, rng = SO101Kinematics(), np.random.default_rng(0)
    for _ in range(500):
        q = {n: rng.uniform(-1.6, 1.6) for n in ARM_JOINTS}
        q["wrist_roll"] = 0.0
        assert np.linalg.norm(kin.forward(urdf_to_model(q)) - (fk(q) - origin)) < 1e-4


def test_roll_moves_tip_at_most_the_axis_offset(urdf_fk):
    # The tip sits 7.9 mm off the wrist_roll axis, so roll != 0 is approximate in the model.
    fk, origin = urdf_fk
    kin, rng = SO101Kinematics(), np.random.default_rng(1)
    for _ in range(200):
        q = {n: rng.uniform(-1.6, 1.6) for n in ARM_JOINTS}
        assert np.linalg.norm(kin.forward(urdf_to_model(q)) - (fk(q) - origin)) < 2 * 0.0079 + 1e-4


def test_ik_target_is_hit_in_the_urdf(urdf_fk):
    fk, origin = urdf_fk
    target = np.array([0.15, 0.04, 0.028])
    q = model_to_urdf(SO101Kinematics().inverse(target))
    assert np.linalg.norm(fk(q) - origin - target) < 1e-4


def test_mapping_round_trips():
    q = {"shoulder_pan": 0.3, "shoulder_lift": -0.7, "elbow_flex": 1.1, "wrist_flex": 0.2,
         "wrist_roll": -0.4, "gripper": 40.0}
    assert urdf_to_model(model_to_urdf(q)) == pytest.approx(q)
    assert model_to_urdf(q)["gripper"] == 40.0  # percent passes through


def test_limits_are_the_urdf_limits():
    root = ET.parse(URDF).getroot()
    limits = SO101Model().limits
    for name in ARM_JOINTS:
        lim = root.find(f"joint[@name='{name}']/limit")
        assert limits[name].lo == pytest.approx(float(lim.get("lower")), abs=1e-5)
        assert limits[name].hi == pytest.approx(float(lim.get("upper")), abs=1e-5)

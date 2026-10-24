"""URDF-zero <-> servo map: fitting from nudges, conversions, limits, and the guided script."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from magnus_bot.kinematics.conventions import (
    ARM_JOINTS,
    NudgeError,
    ServoMap,
    fit_servo_map,
    load_servo_map,
    model_to_urdf,
)
from magnus_bot.kinematics.model import SO101Model
from magnus_bot.kinematics.so101_ik import SO101Kinematics

# A made-up "real" arm: two joints count backwards, all have some offset.
TRUE = ServoMap("follower_1", {
    "shoulder_pan": (1, 0.03),
    "shoulder_lift": (-1, -0.10),
    "elbow_flex": (1, 0.20),
    "wrist_flex": (-1, 0.05),
    "wrist_roll": (1, -0.40),
})
ZERO = dict.fromkeys(ARM_JOINTS, 0.0)


def _reading(urdf):
    return TRUE.urdf_to_servo(urdf)


def _nudges(size=0.2):
    return {j: _reading({**ZERO, j: size}) for j in ARM_JOINTS}


def test_fit_recovers_sign_and_offset():
    fitted = fit_servo_map("follower_1", _reading(ZERO), _nudges())
    for j in ARM_JOINTS:
        assert fitted.joints[j][0] == TRUE.joints[j][0]
        assert fitted.joints[j][1] == pytest.approx(TRUE.joints[j][1])


def test_fit_with_a_non_zero_reference_pose():
    ref = {**ZERO, "shoulder_lift": -0.2438}  # e.g. upper arm exactly vertical
    nudged = {j: _reading({**ref, j: ref[j] + 0.2}) for j in ARM_JOINTS}
    fitted = fit_servo_map("follower_1", _reading(ref), nudged, reference=ref)
    assert fitted.joints["shoulder_lift"][1] == pytest.approx(TRUE.joints["shoulder_lift"][1])


def test_scrambled_ids_are_caught():
    # Wires for elbow and wrist swapped: nudging elbow shows up on wrist_flex.
    nudged = _nudges()
    e, w = nudged["elbow_flex"], dict(nudged["elbow_flex"])
    w["elbow_flex"], w["wrist_flex"] = e["wrist_flex"], e["elbow_flex"]
    nudged["elbow_flex"] = w
    with pytest.raises(NudgeError, match="nudging elbow_flex moved wrist_flex most"):
        fit_servo_map("follower_1", _reading(ZERO), nudged)


def test_too_small_a_nudge_is_rejected():
    with pytest.raises(NudgeError, match="nudge >= 3 deg"):
        fit_servo_map("follower_1", _reading(ZERO), _nudges(size=0.02))


def test_conversions_round_trip_and_gripper_passes_through():
    q = {"shoulder_pan": 0.3, "shoulder_lift": -0.7, "elbow_flex": 1.1, "wrist_flex": 0.2,
         "wrist_roll": -0.4, "gripper": 40.0}
    assert TRUE.servo_to_urdf(TRUE.urdf_to_servo(q)) == pytest.approx(q)
    assert TRUE.urdf_to_servo(q)["gripper"] == 40.0
    model = SO101Kinematics().inverse(np.array([0.15, 0.03, 0.03]))
    assert TRUE.model_to_servo(model) == pytest.approx(TRUE.urdf_to_servo(model_to_urdf(model)))


def test_servo_limits_follow_the_map():
    limits = TRUE.servo_limits(SO101Model().limits)
    lift = SO101Model().limits["shoulder_lift"]
    # sign -1, offset -0.10: the ends swap and shift.
    assert limits["shoulder_lift"].lo == pytest.approx(-lift.hi - 0.10)
    assert limits["shoulder_lift"].hi == pytest.approx(-lift.lo - 0.10)
    assert limits["gripper"] == SO101Model().limits["gripper"]


def test_json_round_trip_and_validation(tmp_path):
    path = tmp_path / "follower_1.json"
    path.write_text(json.dumps(TRUE.to_dict()))
    assert load_servo_map(path) == TRUE
    bad = TRUE.to_dict()
    bad["joints"]["wrist_roll"]["sign"] = 0
    with pytest.raises(ValueError):
        ServoMap.from_dict(bad)


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "joint_zero_check.py"
    spec = importlib.util.spec_from_file_location("joint_zero_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Operator:
    """Plays the person at the rig: each Enter moves the fake arm to the next pose."""

    def __init__(self, poses):
        self.poses, self.current = iter(poses), None

    def ask(self, _prompt):
        self.current = next(self.poses)
        return ""

    def read(self):
        return _reading(self.current)


def test_guided_check_end_to_end(script):
    on_table = model_to_urdf(SO101Kinematics().inverse(np.array([0.20, 0.02, 0.0])))
    poses = [ZERO, *({**ZERO, j: 0.25} for j in ARM_JOINTS), on_table]
    op, said = _Operator(poses), []
    smap, height, reach = script.run_check("follower_1", op.read, op.ask, said.append, pause=0)
    for j in ARM_JOINTS:
        assert smap.joints[j][0] == TRUE.joints[j][0]
        assert smap.joints[j][1] == pytest.approx(TRUE.joints[j][1])
    assert height == pytest.approx(0.0, abs=1e-6)
    assert reach == pytest.approx(math.hypot(0.20, 0.02), abs=1e-6)
    assert not any("WARNING" in line for line in said)
    assert "sign -1" in script.report(smap)


def test_guided_check_warns_when_the_tip_is_not_on_the_table(script):
    lifted = model_to_urdf(SO101Kinematics().inverse(np.array([0.20, 0.0, 0.05])))
    op, said = _Operator([ZERO, *({**ZERO, j: 0.25} for j in ARM_JOINTS), lifted]), []
    script.run_check("follower_1", op.read, op.ask, said.append, pause=0)
    assert any("WARNING" in line for line in said)


def test_every_joint_has_a_nudge_hint(script):
    assert set(script.NUDGE) == set(ARM_JOINTS)

"""SimArm, safety clamps, and motion primitives."""

import math

import numpy as np
import pytest

from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.model import JointLimit
from magnus_bot.motion import safety
from magnus_bot.motion.primitives import CLOSE, OPEN, Phase, pick_place_steps

# --- SimArm ---------------------------------------------------------------

def test_sim_arm_requires_connection():
    arm = SimArm("follower_1")
    with pytest.raises(RuntimeError):
        arm.read_joints()
    arm.connect()
    assert arm.is_connected
    assert arm.read_joints()["gripper"] == 100.0


def test_sim_arm_clamps_and_logs():
    arm = SimArm("follower_1")
    arm.connect()
    sent = arm.send_joints({"shoulder_pan": 10.0})  # far beyond the +/- pi/2 limit
    assert sent["shoulder_pan"] == pytest.approx(math.pi / 2)
    assert arm.read_joints()["shoulder_pan"] == pytest.approx(math.pi / 2)
    assert arm.command_log == [sent]


def test_sim_arm_hold_repeats_current():
    arm = SimArm("follower_1")
    arm.connect()
    arm.send_joints({"elbow_flex": 0.1})
    held = arm.hold()
    assert held["elbow_flex"] == pytest.approx(0.1)


# --- safety ---------------------------------------------------------------

def test_clamp_joint_and_speed():
    limits = {"j": JointLimit(-1.0, 1.0)}
    assert safety.clamp_joint("j", 5.0, limits) == 1.0
    assert safety.clamp_joint("j", -5.0, limits) == -1.0
    assert safety.clamp_speed_scale(0.3) == 0.3
    assert safety.clamp_speed_scale(2.0) == 1.0
    with pytest.raises(ValueError):
        safety.clamp_speed_scale(0.0)


# --- primitives -----------------------------------------------------------

def test_pick_place_phase_sequence():
    steps = pick_place_steps([0.1, 0.1, 0.02], [0.2, 0.2, 0.02], z_clear=0.12)
    assert [s.phase for s in steps] == [
        Phase.HOVER, Phase.DESCEND_PICK, Phase.GRIP, Phase.LIFT,
        Phase.TRAVERSE, Phase.DESCEND_PLACE, Phase.RELEASE, Phase.RETRACT,
    ]
    assert steps[0].gripper == OPEN  # hover open
    assert steps[2].gripper == CLOSE  # grip closes
    assert steps[6].gripper == OPEN  # release opens
    # Traverse happens at z_clear above both ends.
    assert steps[4].xyz[2] == pytest.approx(0.12)
    assert np.allclose(steps[4].xyz[:2], [0.2, 0.2])


def test_pick_place_rejects_low_clearance():
    with pytest.raises(ValueError):
        pick_place_steps([0, 0, 0.1], [0, 0, 0.1], z_clear=0.05)

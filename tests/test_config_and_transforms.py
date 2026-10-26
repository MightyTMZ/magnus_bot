"""Config loading (packaged configs) and transform helpers."""

import dataclasses

import numpy as np
import pytest

from magnus_bot.config import load_config
from magnus_bot.geometry.transforms import Pose2D, perp_left
from magnus_bot.motion.trajectory import min_jerk_interp, min_jerk_scalar


def test_packaged_config_loads():
    config = load_config()
    assert config.board.human_color == "black"
    assert config.board.robot_color == "white"
    assert set(config.arms.arms) == {"follower_1", "follower_2"}
    assert config.arms.arms["follower_1"].port == "COM3"


def test_require_measured_refuses_on_placeholders():
    # Build an explicitly-unmeasured config (the shipped one may be measured now the rig is real).
    config = dataclasses.replace(load_config(), measured=False)
    with pytest.raises(RuntimeError):
        config.require_measured()


def test_z_clear_above_tallest_piece():
    config = load_config()
    assert config.arms.z_clear > max(config.board.piece_heights.values())


def test_pose2d_roundtrip():
    pose = Pose2D(x=1.0, y=2.0, theta=0.7)
    p = np.array([0.3, -0.4])
    assert np.allclose(pose.parent_to_local(pose.local_to_parent(p)), p)


def test_perp_left_rotation():
    assert np.allclose(perp_left([2.0, 0.0]), [0.0, 2.0])


def test_min_jerk_profile():
    assert min_jerk_scalar(0.0) == 0.0
    assert min_jerk_scalar(1.0) == pytest.approx(1.0)
    assert min_jerk_scalar(0.5) == pytest.approx(0.5)  # symmetric midpoint
    path = min_jerk_interp([0.0], [10.0], num_steps=5)
    assert path[0][0] == pytest.approx(0.0)
    assert path[-1][0] == pytest.approx(10.0)

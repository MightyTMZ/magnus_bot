"""Executor: program building (dry-run) and joint execution (simulate, live gates)."""

import dataclasses
import math

import chess
import numpy as np
import pytest

from magnus_bot.cli import _full_reach_map
from magnus_bot.config import load_config
from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.model import JOINT_NAMES, JointLimit
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable
from magnus_bot.orchestration.executor import (
    ExecutionFault,
    Executor,
    JointLimitError,
    build_move_program,
    check_limits,
)


def test_build_program_normal_move_is_one_sequence():
    config = load_config()
    rmap = _full_reach_map(config)
    program = build_move_program(chess.Board(), chess.Move.from_uci("e2e4"), rmap, config)
    assert len(program) == 1
    assert program[0].transfer.reason == "move"
    assert len(program[0].steps) == 8  # full pick/place primitive


def test_build_program_capture_has_victim_then_move():
    config = load_config()
    rmap = _full_reach_map(config)
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    program = build_move_program(board, chess.Move.from_uci("e4d5"), rmap, config)
    assert [p.transfer.reason for p in program] == ["capture-victim", "move"]


def test_executor_dry_run_returns_program_without_moving():
    config = load_config()
    rmap = _full_reach_map(config)
    arms = {a: SimArm(a) for a in config.arms.arms}
    executor = Executor(config, arms, rmap, live=False)
    program = executor.execute_move(chess.Board(), chess.Move.from_uci("e2e4"))
    assert program and all(not arm.command_log for arm in arms.values())  # dry-run commands nothing


# --- joint execution (simulate / live gates) -------------------------------


# IK angles are model convention (not servo), so the placeholder +/- pi/2 servo limits reject
# them. Until the joint mapping lands, sim runs use limits wide enough for model angles.
WIDE = {n: JointLimit(-2 * math.pi, 2 * math.pi) for n in JOINT_NAMES[:5]}
WIDE["gripper"] = JointLimit(0.0, 100.0)
ROOK_H4 = chess.Board("4k3/8/8/8/7R/8/8/4K3 w - - 0 1")  # h4h5 is within follower_1's reach
H4H5 = chess.Move.from_uci("h4h5")


def _sim(config, **kwargs):
    arms = {a: SimArm(a, limits=WIDE) for a in config.arms.arms}
    for arm in arms.values():
        arm.connect()
    kwargs.setdefault("limits", WIDE)
    return arms, Executor(config, arms, _full_reach_map(config), simulate=True, **kwargs)


def test_simulate_moves_only_the_assigned_arm_and_returns_to_rest():
    config = load_config()
    arms, executor = _sim(config)
    start = arms["follower_1"].read_joints()
    executor.execute_move(ROOK_H4, H4H5)
    assert arms["follower_1"].command_log
    assert not arms["follower_2"].command_log  # one arm at a time; the other is untouched
    assert arms["follower_1"].read_joints() == pytest.approx(start)


def test_simulated_grip_happens_at_the_source_square():
    config = load_config()
    arms, executor = _sim(config)
    executor.execute_move(ROOK_H4, H4H5)
    grip = next(c for c in arms["follower_1"].command_log if c["gripper"] == config.arms.gripper_closed)
    # FK of the first fully-closed command, mapped back to the board, is h4 at grasp height.
    tip = SO101Kinematics().forward(grip)
    placement = arm_placements(config.board.square_size, config.arms.edge_gap)["follower_1"]
    xy = placement.arm_to_board(tip[:2])
    assert xy == pytest.approx(board_geom.square_center(7, 3, config.board.square_size), abs=1e-6)
    assert tip[2] == pytest.approx(config.arms.z_grasp, abs=1e-6)


def test_unreachable_move_aborts_before_any_motion():
    config = load_config()
    arms, executor = _sim(config)
    with pytest.raises(Unreachable, match="follower_1"):
        executor.execute_move(chess.Board(), chess.Move.from_uci("e2e4"))  # e2 is out of reach
    assert all(not arm.command_log for arm in arms.values())


def test_placeholder_limits_reject_model_angles_before_motion():
    config = load_config()
    arms, executor = _sim(config, limits=SimArm("x")._limits)  # the +/- pi/2 placeholders
    with pytest.raises(JointLimitError, match="h4h5|move"):
        executor.execute_move(ROOK_H4, H4H5)
    assert all(not arm.command_log for arm in arms.values())


def test_check_limits_names_every_offender():
    with pytest.raises(JointLimitError, match="a=2.000.*b=-3.000"):
        check_limits({"a": 2.0, "b": -3.0, "c": 0.0}, {n: JointLimit(-1, 1) for n in "abc"})


class _FailingArm(SimArm):
    def __init__(self, *args, fail_at: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_at, self.sent, self.held = fail_at, 0, False

    def send_joints(self, targets):
        self.sent += 1
        if self.sent == self.fail_at:
            raise OSError("bus timeout")
        return super().send_joints(targets)

    def hold(self):
        self.held = True
        return super().hold()


def test_fault_mid_motion_holds_every_arm():
    config = load_config()
    arms = {
        "follower_1": _FailingArm("follower_1", limits=WIDE, fail_at=10),
        "follower_2": _FailingArm("follower_2", limits=WIDE, fail_at=-1),
    }
    for arm in arms.values():
        arm.connect()
    executor = Executor(config, arms, _full_reach_map(config), simulate=True, limits=WIDE)
    with pytest.raises(ExecutionFault, match="bus timeout"):
        executor.execute_move(ROOK_H4, H4H5)
    assert arms["follower_1"].held and arms["follower_2"].held


def test_live_refuses_without_joint_mapping():
    config = dataclasses.replace(load_config(), measured=True)
    with pytest.raises(RuntimeError, match="joint mapping"):
        Executor(config, {}, _full_reach_map(config), live=True)


def test_live_refuses_on_unmeasured_config():
    config = load_config()
    with pytest.raises(Exception, match="(?i)measured"):
        Executor(config, {}, _full_reach_map(config), live=True, to_command=dict)


def test_simulate_refuses_real_arms():
    config = load_config()
    with pytest.raises(TypeError):
        Executor(config, {"follower_1": object()}, _full_reach_map(config), simulate=True)


def test_identity_command_map_matches_default():
    config = load_config()
    _, default = _sim(config)
    _, mapped = _sim(config, to_command=dict)
    program = build_move_program(ROOK_H4, H4H5, _full_reach_map(config), config)
    a, b = default.plan(program), mapped.plan(program)
    assert np.allclose([list(s.values()) for s in a[0].samples], [list(s.values()) for s in b[0].samples])

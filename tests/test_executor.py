"""Executor program building (dry-run), end to end over decompose + assign + primitives."""

import chess

from magnus_bot.cli import _full_reach_map
from magnus_bot.config import load_config
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.orchestration.executor import Executor, build_move_program


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

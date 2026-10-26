"""scripts/play_stockfish.py: Stockfish plays the robots' side (fallback moves, sim game)."""

import importlib.util
import sys
from pathlib import Path

import chess
import pytest

from magnus_bot.config import TunedGrasp, save_tuned_grasp

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def play():
    sys.path.insert(0, str(SCRIPTS))  # it imports play_moves, as when run from scripts/
    try:
        spec = importlib.util.spec_from_file_location("play_stockfish", SCRIPTS / "play_stockfish.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["play_stockfish"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SCRIPTS))


class _Engine:
    """Stand-in for StockfishSource: plays ``picks`` in turn; ``ranked`` are its fallbacks."""

    def __init__(self, picks=(), ranked=()):
        self.picks, self.ranked = list(picks), list(ranked)

    def next_move(self, board):
        return chess.Move.from_uci(self.picks.pop(0))

    def candidates(self, board, n=5):
        return [chess.Move.from_uci(u) for u in self.ranked[:n]]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def test_the_next_best_engine_move_is_played_when_the_arms_cannot_play_its_pick(play):
    board = chess.Board()
    engine = _Engine(picks=["e2e4"], ranked=["e2e4", "d2d4", "g1f3"])

    def plan(move):
        if move.uci() == "e2e4":
            raise play.Unplayable("out of reach")
        return f"program {move.uci()}"

    move, program = play.choose_move(board, engine, plan)
    assert move.uci() == "d2d4" and program == "program d2d4"


def test_no_move_when_the_arms_can_play_none_of_the_engine_moves(play):
    def plan(move):
        raise play.Unplayable("out of reach")

    assert play.choose_move(chess.Board(), _Engine(picks=["e2e4"], ranked=["d2d4"]), plan) is None


def test_a_sim_game_plans_each_engine_move_and_takes_typed_replies(play, config_dir, monkeypatch, capsys):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "pawn", TunedGrasp(pitch_deg=45.0, z_offset_mm=-2.0))
    monkeypatch.setattr(play, "StockfishSource", lambda *a, **k: _Engine(picks=["e2e4", "d2d4"]))
    replies = iter(["not-a-move", "e5", ""])  # rejected, SAN, then stop
    assert play.main(["--config-dir", str(cfg)], prompt=lambda _: next(replies)) == 0
    captured = capsys.readouterr()
    assert "SIM OK e2e4 (e4)" in captured.out and "SIM OK d2d4 (d4)" in captured.out
    assert "not legal here: not-a-move" in captured.err
    assert "position now: rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b" in captured.out


def test_the_engine_is_not_needed_installed_to_report_it_missing(play, config_dir, monkeypatch, capsys):
    def missing(*a, **k):
        raise FileNotFoundError("Stockfish not found")

    monkeypatch.setattr(play, "StockfishSource", missing)
    assert play.main(["--config-dir", str(config_dir(measured=False))]) == 1
    assert "Stockfish not found" in capsys.readouterr().err

"""Move sources: RandomLegalSource and CLISource."""

import chess
import pytest

from magnus_bot.sources.cli_source import CLISource
from magnus_bot.sources.move_source import IllegalMoveError, validate_move
from magnus_bot.sources.random_source import RandomLegalSource


def test_random_source_is_legal_and_reproducible():
    board = chess.Board()
    a = RandomLegalSource(seed=42).next_move(board)
    b = RandomLegalSource(seed=42).next_move(board)
    assert a == b
    assert a in board.legal_moves


def test_validate_move_rejects_illegal():
    board = chess.Board()
    with pytest.raises(IllegalMoveError):
        validate_move(board, chess.Move.from_uci("e2e5"))


def test_cli_source_accepts_uci_and_san():
    board = chess.Board()
    src = CLISource(reader=lambda _prompt: "e2e4", writer=lambda _msg: None)
    assert src.next_move(board) == chess.Move.from_uci("e2e4")

    src_san = CLISource(reader=lambda _prompt: "Nf3", writer=lambda _msg: None)
    assert src_san.next_move(board) == chess.Move.from_uci("g1f3")


def test_cli_source_reprompts_on_illegal_then_accepts():
    board = chess.Board()
    inputs = iter(["e2e5", "e2e4"])  # first illegal, then legal
    messages = []
    src = CLISource(reader=lambda _prompt: next(inputs), writer=messages.append)
    assert src.next_move(board) == chess.Move.from_uci("e2e4")
    assert messages  # it complained about the illegal move

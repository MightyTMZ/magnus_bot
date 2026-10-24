"""A move source that plays a uniformly random legal move. Used by tests and demos."""

from __future__ import annotations

import random

import chess

from magnus_bot.sources.move_source import validate_move


class RandomLegalSource:
    """Picks a uniformly random legal move. Seedable for reproducible tests."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def next_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(f"no legal moves in position {board.fen()}")
        return validate_move(board, self._rng.choice(legal))

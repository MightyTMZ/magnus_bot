"""The :class:`MoveSource` protocol and a shared legality check.

A move source answers one question: given the current position, what is the next move for
the side to play? The orchestrator does not care whether that comes from an engine, a human
typing UCI, a random generator (tests), or the physical board read by a camera.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chess


@runtime_checkable
class MoveSource(Protocol):
    """Produces the next move for ``board.turn``. Implementations must return a legal move."""

    def next_move(self, board: chess.Board) -> chess.Move:
        """Return the move to play in ``board``. Must be in ``board.legal_moves``."""
        ...


def validate_move(board: chess.Board, move: chess.Move) -> chess.Move:
    """Return ``move`` if legal in ``board``; otherwise raise :class:`IllegalMoveError`."""
    if move not in board.legal_moves:
        raise IllegalMoveError(move, board)
    return move


class IllegalMoveError(ValueError):
    """Raised when a source proposes a move that is not legal in the position."""

    def __init__(self, move: chess.Move, board: chess.Board) -> None:
        self.move = move
        self.fen = board.fen()
        super().__init__(f"illegal move {move.uci()} in position {self.fen}")

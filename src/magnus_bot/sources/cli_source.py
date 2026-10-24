"""A move source that reads a move typed by a human (UCI or SAN) and validates it."""

from __future__ import annotations

from collections.abc import Callable

import chess

from magnus_bot.sources.move_source import IllegalMoveError, validate_move


class CLISource:
    """Prompt for a move on each turn, accepting UCI (e2e4, e7e8q) or SAN (Nf3, exd5).

    ``reader`` returns one line of input (defaults to :func:`input`); ``writer`` prints
    prompts/errors (defaults to :func:`print`). Both are injectable for testing. Re-prompts
    on an unparseable or illegal move.
    """

    def __init__(
        self,
        reader: Callable[[str], str] = input,
        writer: Callable[[str], None] = print,
    ) -> None:
        self._read = reader
        self._write = writer

    def next_move(self, board: chess.Board) -> chess.Move:
        while True:
            raw = self._read(f"{_color(board)} to move (UCI or SAN): ").strip()
            move = self._parse(board, raw)
            if move is None:
                self._write(f"Could not parse {raw!r} as a move. Try again.")
                continue
            try:
                return validate_move(board, move)
            except IllegalMoveError as exc:
                self._write(str(exc))

    @staticmethod
    def _parse(board: chess.Board, raw: str) -> chess.Move | None:
        for parse in (lambda s: chess.Move.from_uci(s), board.parse_san):
            try:
                return parse(raw)
            except ValueError:
                continue
        return None


def _color(board: chess.Board) -> str:
    return "White" if board.turn == chess.WHITE else "Black"

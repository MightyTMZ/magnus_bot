"""A move source backed by a UCI engine (Stockfish) via python-chess.

The Stockfish binary is NOT bundled. Provide its path explicitly or via the ``STOCKFISH_PATH``
environment variable; :meth:`StockfishSource.next_move` raises a clear error if it is missing.
Tests use :class:`~magnus_bot.sources.random_source.RandomLegalSource` instead, so this module
is never exercised without a real engine installed.
"""

from __future__ import annotations

import os
import shutil

import chess
import chess.engine

from magnus_bot.sources.move_source import validate_move

DEFAULT_THINK_TIME = 0.1  # seconds per move


class StockfishSource:
    """Query a UCI engine for the best move.

    Use as a context manager (``with StockfishSource() as src: ...``) or call :meth:`close`
    to release the engine subprocess.
    """

    def __init__(self, engine_path: str | None = None, think_time: float = DEFAULT_THINK_TIME) -> None:
        self.engine_path = engine_path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
        if not self.engine_path:
            raise FileNotFoundError(
                "Stockfish not found. Install it and set STOCKFISH_PATH, or pass engine_path=..."
            )
        self.think_time = think_time
        self._engine: chess.engine.SimpleEngine | None = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self._engine

    def next_move(self, board: chess.Board) -> chess.Move:
        result = self._ensure_engine().play(board, chess.engine.Limit(time=self.think_time))
        if result.move is None:
            raise RuntimeError(f"engine returned no move for {board.fen()}")
        return validate_move(board, result.move)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

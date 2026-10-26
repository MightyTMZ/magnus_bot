"""A move source backed by a UCI engine (Stockfish) via python-chess.

The Stockfish binary is NOT bundled. Provide its path explicitly or via the ``STOCKFISH_PATH``
environment variable; otherwise ``stockfish`` on PATH, then a winget install
(``winget install Stockfish.Stockfish``), is used. The constructor raises a clear error if none is found.
Tests use :class:`~magnus_bot.sources.random_source.RandomLegalSource` instead, so this module
is never exercised without a real engine installed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import chess
import chess.engine

from magnus_bot.sources.move_source import validate_move

DEFAULT_THINK_TIME = 0.1  # seconds per move


def find_stockfish() -> str | None:
    """STOCKFISH_PATH, else ``stockfish`` on PATH, else winget's install (its PATH alias only
    reaches shells started after the install)."""
    found = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    exes = sorted(packages.glob("Stockfish.Stockfish_*/**/stockfish*.exe")) if packages.is_dir() else []
    return str(exes[0]) if exes else None


class StockfishSource:
    """Query a UCI engine for the best move.

    Use as a context manager (``with StockfishSource() as src: ...``) or call :meth:`close`
    to release the engine subprocess.
    """

    def __init__(self, engine_path: str | None = None, think_time: float = DEFAULT_THINK_TIME,
                 elo: int | None = None) -> None:
        self.engine_path = engine_path or find_stockfish()
        if not self.engine_path:
            raise FileNotFoundError(
                "Stockfish not found. Install it (winget install Stockfish.Stockfish) and set "
                "STOCKFISH_PATH, or pass engine_path=..."
            )
        self.think_time = think_time
        self.elo = elo  # None: full strength; else UCI_Elo (Stockfish allows 1320-3190)
        self._engine: chess.engine.SimpleEngine | None = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            if self.elo is not None:
                self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        return self._engine

    def next_move(self, board: chess.Board) -> chess.Move:
        result = self._ensure_engine().play(board, chess.engine.Limit(time=self.think_time))
        if result.move is None:
            raise RuntimeError(f"engine returned no move for {board.fen()}")
        return validate_move(board, result.move)

    def candidates(self, board: chess.Board, n: int = 5) -> list[chess.Move]:
        """The engine's top ``n`` moves, best first: fallbacks for when its pick cannot be played."""
        infos = self._ensure_engine().analyse(board, chess.engine.Limit(time=self.think_time), multipv=n)
        return [validate_move(board, info["pv"][0]) for info in infos if info.get("pv")]

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

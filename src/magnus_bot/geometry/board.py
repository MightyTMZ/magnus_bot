"""Chess-board squares and their coordinates in the board frame B.

Conventions (see docs/frames.md):
  * file index f = 0..7 for a..h; rank index r = 0..7 for ranks 1..8.
  * square (f, r) has its center at ((f + 0.5) * s, (r + 0.5) * s), with s the square size.
  * a1 is a DARK square; h1 is light.

The (f, r) convention matches python-chess exactly: ``chess.square_file`` / ``chess.square_rank``
return these same indices, so conversions are a straight pass-through.
"""

from __future__ import annotations

import chess
import numpy as np

FILES = "abcdefgh"
RANKS = "12345678"

BOARD_SIZE = 8


def square_name(file: int, rank: int) -> str:
    """(file, rank) indices -> algebraic name, e.g. (3, 0) -> 'd1'."""
    _check(file, rank)
    return f"{FILES[file]}{RANKS[rank]}"


def parse_square(name: str) -> tuple[int, int]:
    """Algebraic name -> (file, rank), e.g. 'd1' -> (3, 0)."""
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in FILES or name[1] not in RANKS:
        raise ValueError(f"not a square name: {name!r}")
    return FILES.index(name[0]), RANKS.index(name[1])


def is_dark(file: int, rank: int) -> bool:
    """True if (file, rank) is a dark square. a1 (0, 0) is dark, so even file+rank is dark."""
    _check(file, rank)
    return (file + rank) % 2 == 0


def square_color(file: int, rank: int) -> str:
    """'dark' or 'light' for the square at (file, rank)."""
    return "dark" if is_dark(file, rank) else "light"


def square_center(file: int, rank: int, square_size: float) -> np.ndarray:
    """Center of square (file, rank) in frame B as a 2-vector [x, y] (meters)."""
    _check(file, rank)
    return np.array([(file + 0.5) * square_size, (rank + 0.5) * square_size], dtype=float)


def coords_to_square(xy: np.ndarray, square_size: float) -> tuple[int, int]:
    """Board-frame [x, y] -> (file, rank) of the square containing it. Raises if off-board."""
    xy = np.asarray(xy, dtype=float)
    file = int(np.floor(xy[0] / square_size))
    rank = int(np.floor(xy[1] / square_size))
    _check(file, rank)
    return file, rank


# --- python-chess interop --------------------------------------------------

def chess_square_to_fr(square: int) -> tuple[int, int]:
    """python-chess square index -> (file, rank)."""
    return chess.square_file(square), chess.square_rank(square)


def fr_to_chess_square(file: int, rank: int) -> int:
    """(file, rank) -> python-chess square index."""
    _check(file, rank)
    return chess.square(file, rank)


def chess_square_center(square: int, square_size: float) -> np.ndarray:
    """Center of a python-chess square in frame B."""
    return square_center(*chess_square_to_fr(square), square_size)


def _check(file: int, rank: int) -> None:
    if not (0 <= file < BOARD_SIZE and 0 <= rank < BOARD_SIZE):
        raise ValueError(f"square index out of range: file={file}, rank={rank}")

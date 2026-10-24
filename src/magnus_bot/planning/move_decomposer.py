"""Decompose a chess move into an ordered list of physical piece transfers.

A :class:`Transfer` moves one piece from a source to a destination, where each end is either a
board square or an off-board location (a capture bin, the handoff pad, or the promotion stash).
The orchestrator executes the transfers in order; order matters (e.g. the victim leaves a square
before the capturing piece arrives).

Rules (see CLAUDE.md "Design decisions"):
  * normal move: src -> dst.
  * capture: victim dst -> capturing side's bin, THEN mover src -> dst.
  * castling: king first, then rook.
  * en passant: victim pawn (not on dst) -> bin, THEN pawn src -> dst.
  * promotion: remove pawn src -> bin, THEN promoted piece stash -> dst (preceded by the victim
    transfer if the promotion is also a capture).

Moves are UCI-canonical; python-chess owns move semantics. ``board`` is the position BEFORE the
move (so captures/castling/en passant can be detected).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

# Off-board location kinds.
SQUARE = "square"
BIN = "bin"
PAD = "pad"
STASH = "stash"


@dataclass(frozen=True)
class Location:
    """Where a piece is picked from or placed. ``square`` is set iff ``kind == "square"``.

    For a bin, ``owner`` is the capturing side's color ("white"/"black"); execution maps that
    to a physical bin. For the pad/stash, ``owner`` is unused.
    """

    kind: str
    square: tuple[int, int] | None = None
    owner: str | None = None

    @classmethod
    def of_square(cls, sq: int) -> Location:
        return cls(SQUARE, square=(chess.square_file(sq), chess.square_rank(sq)))

    @classmethod
    def bin_for(cls, color: bool) -> Location:
        return cls(BIN, owner="white" if color == chess.WHITE else "black")

    @classmethod
    def pad(cls) -> Location:
        return cls(PAD)

    @classmethod
    def stash(cls) -> Location:
        return cls(STASH)


@dataclass(frozen=True)
class Transfer:
    """Move one piece from ``src`` to ``dst``. ``piece_symbol`` and ``reason`` are for logging."""

    src: Location
    dst: Location
    piece_symbol: str
    reason: str


def decompose_move(board: chess.Board, move: chess.Move) -> list[Transfer]:
    """Ordered transfers realizing ``move`` in ``board`` (the pre-move position)."""
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move.uci()} in {board.fen()}")

    mover = board.piece_at(move.from_square)
    assert mover is not None  # legal move always has a piece on from_square
    transfers: list[Transfer] = []

    # 1. Remove any captured piece first, into the capturing side's bin.
    if board.is_en_passant(move):
        victim_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        transfers.append(_capture(board, victim_sq, board.turn))
    elif board.is_capture(move):  # ordinary capture (victim stands on the destination square)
        transfers.append(_capture(board, move.to_square, board.turn))

    # 2. Move the piece(s).
    if board.is_castling(move):
        transfers += _castle(board, move)
    elif move.promotion is not None:
        # The pawn leaves the board; a promoted piece comes from the stash onto the destination.
        transfers.append(
            Transfer(Location.of_square(move.from_square), Location.bin_for(board.turn),
                     mover.symbol(), "promote-remove")
        )
        promoted = chess.Piece(move.promotion, board.turn)
        transfers.append(
            Transfer(Location.stash(), Location.of_square(move.to_square),
                     promoted.symbol(), "promote-place")
        )
    else:
        transfers.append(
            Transfer(Location.of_square(move.from_square), Location.of_square(move.to_square),
                     mover.symbol(), "move")
        )

    return transfers


def _capture(board: chess.Board, victim_sq: int, capturing_side: bool) -> Transfer:
    victim = board.piece_at(victim_sq)
    assert victim is not None
    return Transfer(
        Location.of_square(victim_sq),
        Location.bin_for(capturing_side),
        victim.symbol(),
        "capture-victim",
    )


def _castle(board: chess.Board, move: chess.Move) -> list[Transfer]:
    """King first, then rook. Rook squares derived from the king's direction."""
    rank = chess.square_rank(move.from_square)
    kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
    rook_from = chess.square(7 if kingside else 0, rank)
    rook_to = chess.square(5 if kingside else 3, rank)
    king = board.piece_at(move.from_square)
    rook = board.piece_at(rook_from)
    assert king is not None and rook is not None
    return [
        Transfer(Location.of_square(move.from_square), Location.of_square(move.to_square),
                 king.symbol(), "castle-king"),
        Transfer(Location.of_square(rook_from), Location.of_square(rook_to),
                 rook.symbol(), "castle-rook"),
    ]

"""STUB (M4): read the human's move from the physical board via the camera.

This source infers the move the human just made by diffing the board state seen by the
:class:`~magnus_bot.vision.camera.Camera` against the last known position. It depends on the
vision stack (camera + homography + square occupancy) which is interface-only for now.

Until M4, use :class:`~magnus_bot.sources.cli_source.CLISource` to enter the human's move.
"""

from __future__ import annotations

import chess

from magnus_bot.vision.camera import Camera


class BoardSource:
    """Infer the side-to-move's move from the physical board. Not implemented until M4."""

    def __init__(self, camera: Camera) -> None:
        self._camera = camera

    def next_move(self, board: chess.Board) -> chess.Move:
        # TODO(M4): capture frame -> warp via homography -> per-square occupancy/color ->
        # diff against `board` to recover the move (incl. castling/en passant/promotion) ->
        # disambiguate with board.legal_moves -> validate_move(board, move).
        raise NotImplementedError("BoardSource (vision move input) lands in milestone M4")

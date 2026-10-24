"""Move decomposition: normal, capture, castling, en passant, promotion."""

import chess

from magnus_bot.planning.move_decomposer import BIN, SQUARE, STASH, decompose_move


def _reasons(transfers):
    return [t.reason for t in transfers]


def test_normal_move_is_single_transfer():
    board = chess.Board()
    transfers = decompose_move(board, chess.Move.from_uci("e2e4"))
    assert _reasons(transfers) == ["move"]
    assert transfers[0].src.kind == SQUARE and transfers[0].dst.kind == SQUARE


def test_capture_removes_victim_first():
    # White pawn on e4 captures black pawn on d5.
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    transfers = decompose_move(board, chess.Move.from_uci("e4d5"))
    assert _reasons(transfers) == ["capture-victim", "move"]
    # Victim goes to the capturing (white) side's bin, before the mover arrives.
    assert transfers[0].dst.kind == BIN and transfers[0].dst.owner == "white"
    assert transfers[0].src.square == (3, 4)  # d5


def test_castling_moves_king_then_rook():
    # White kingside castling available.
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    transfers = decompose_move(board, chess.Move.from_uci("e1g1"))
    assert _reasons(transfers) == ["castle-king", "castle-rook"]
    king, rook = transfers
    assert king.src.square == (4, 0) and king.dst.square == (6, 0)  # e1 -> g1
    assert rook.src.square == (7, 0) and rook.dst.square == (5, 0)  # h1 -> f1


def test_en_passant_victim_not_on_destination():
    # White pawn e5 takes en passant on f6; the pawn to remove sits on f5, not on f6.
    board = chess.Board("4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1")
    transfers = decompose_move(board, chess.Move.from_uci("e5f6"))
    assert _reasons(transfers) == ["capture-victim", "move"]
    assert transfers[0].src.square == (5, 4)  # f5: victim pawn, NOT the destination f6
    assert transfers[1].dst.square == (5, 5)  # f6


def test_promotion_removes_pawn_and_places_from_stash():
    # White pawn on a7 promotes on a8 (empty).
    board = chess.Board("1n6/P7/8/8/8/8/8/k6K w - - 0 1")
    transfers = decompose_move(board, chess.Move.from_uci("a7a8q"))
    assert _reasons(transfers) == ["promote-remove", "promote-place"]
    assert transfers[0].dst.kind == BIN  # pawn to bin
    assert transfers[1].src.kind == STASH and transfers[1].piece_symbol == "Q"


def test_promotion_with_capture_removes_victim_then_pawn_then_places():
    # White pawn a7 captures the knight on b8 and promotes.
    board = chess.Board("1n6/P7/8/8/8/8/8/k6K w - - 0 1")
    transfers = decompose_move(board, chess.Move.from_uci("a7b8q"))
    assert _reasons(transfers) == ["capture-victim", "promote-remove", "promote-place"]

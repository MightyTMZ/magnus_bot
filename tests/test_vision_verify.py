"""Transfer verification: grading the camera's occupancy against the planned change."""

import chess
import pytest

from magnus_bot.vision.verify import (
    Outcome,
    check_board_move,
    check_transfer,
    expected_occupancy,
    relocate,
)

# A rook going g3 -> g4 with two bystanders that must not move.
BEFORE = {"g3": "W", "f3": "W", "h4": "B"}
EXPECTED = {"g4": "W", "f3": "W", "h4": "B"}


def test_expected_occupancy_of_the_start_position():
    occ = expected_occupancy(chess.Board())
    assert len(occ) == 32
    assert occ["d1"] == "W" and occ["d8"] == "B"  # queens on their own color's d-file square
    assert "e4" not in occ


def test_relocate_moves_the_piece_and_refuses_an_empty_source():
    assert relocate(BEFORE, "g3", "g4") == EXPECTED
    with pytest.raises(ValueError, match="no piece on e5"):
        relocate(BEFORE, "e5", "e6")


def test_matching_board_is_success():
    check = check_transfer(BEFORE, EXPECTED, dict(EXPECTED))
    assert check.outcome is Outcome.SUCCESS
    assert check.ok and not check.retryable
    assert check.describe() == "success"


def test_unchanged_board_is_not_picked_and_retryable():
    check = check_transfer(BEFORE, EXPECTED, dict(BEFORE))
    assert check.outcome is Outcome.NOT_PICKED
    assert check.retryable and not check.ok


def test_piece_gone_from_both_squares_is_lost():
    check = check_transfer(BEFORE, EXPECTED, {"f3": "W", "h4": "B"})
    assert check.outcome is Outcome.LOST
    assert check.missing == {"g4": "W"}


def test_piece_landing_on_another_square_is_misplaced():
    check = check_transfer(BEFORE, EXPECTED, {"g5": "W", "f3": "W", "h4": "B"})
    assert check.outcome is Outcome.MISPLACED
    assert check.unexpected == {"g5": "W"}


def test_correct_move_but_knocked_neighbour_is_disturbed():
    check = check_transfer(BEFORE, EXPECTED, {"g4": "W", "h4": "B"})  # f3 knocked off
    assert check.outcome is Outcome.DISTURBED
    assert check.missing == {"f3": "W"}


def test_missed_grasp_that_knocks_a_neighbour_is_disturbed_not_retryable():
    check = check_transfer(BEFORE, EXPECTED, {"g3": "W", "h4": "B"})
    assert check.outcome is Outcome.DISTURBED
    assert not check.retryable


def test_piece_on_both_squares_is_unclear():
    check = check_transfer(BEFORE, EXPECTED, {"g3": "W", "g4": "W", "f3": "W", "h4": "B"})
    assert check.outcome is Outcome.UNCLEAR


def test_wrong_color_on_destination_is_reported():
    check = check_transfer(BEFORE, EXPECTED, {"g4": "B", "f3": "W", "h4": "B"})
    assert check.outcome is Outcome.UNCLEAR  # something IS on g4, so not "lost"
    assert check.wrong_color == {"g4": ("W", "B")}
    assert "wrong color g4(W->B)" in check.describe()


def test_unknown_color_matches_either_color():
    assert check_transfer(BEFORE, EXPECTED, {"g4": "?", "f3": "W", "h4": "B"}).ok
    before = {"g3": "?"}
    assert check_transfer(before, relocate(before, "g3", "g4"), {"g4": "W"}).ok


def test_board_move_checks_against_python_chess():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy()
    after.push(move)
    assert check_board_move(board, move, expected_occupancy(after)).ok
    assert check_board_move(board, move, expected_occupancy(board)).outcome is Outcome.NOT_PICKED


def test_castling_counts_every_square_it_changes():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle = chess.Move.from_uci("e1g1")
    after = board.copy()
    after.push(castle)
    assert check_board_move(board, castle, expected_occupancy(after)).ok
    # King moved but the rook stayed on h1: a square the move should change did not.
    king_only = relocate(expected_occupancy(board), "e1", "g1")
    assert not check_board_move(board, castle, king_only).ok

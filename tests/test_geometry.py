"""Frame and board-geometry facts. These pin the coordinate system: fix the frames to satisfy
them, never the other way round (CLAUDE.md). The square size ``s`` and gap ``d`` here are
arbitrary test values, not measurements.
"""

import chess
import numpy as np

from magnus_bot.geometry import board as bg
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.geometry.transforms import perp_left

S = 0.05  # arbitrary test square size (meters)
D = 0.08  # arbitrary test edge gap


def test_a1_is_dark_h1_is_light():
    assert bg.is_dark(*bg.parse_square("a1")) is True
    assert bg.is_dark(*bg.parse_square("h1")) is False
    assert bg.square_color(0, 0) == "dark"
    assert bg.square_color(7, 0) == "light"


def test_square_name_roundtrip_and_center():
    assert bg.square_name(3, 0) == "d1"
    assert bg.parse_square("d1") == (3, 0)
    assert np.allclose(bg.square_center(3, 0, S), [3.5 * S, 0.5 * S])


def test_standard_start_queens_positions_and_coords():
    board = chess.Board()
    # Black queen on d8, white queen on d1.
    assert board.piece_at(chess.D1) == chess.Piece(chess.QUEEN, chess.WHITE)
    assert board.piece_at(chess.D8) == chess.Piece(chess.QUEEN, chess.BLACK)
    # ...at the right board-frame coordinates.
    assert np.allclose(bg.chess_square_center(chess.D1, S), [3.5 * S, 0.5 * S])
    assert np.allclose(bg.chess_square_center(chess.D8, S), [3.5 * S, 7.5 * S])


def test_perp_left_is_plus_90_degrees():
    assert np.allclose(perp_left([1, 0]), [0, 1])
    assert np.allclose(perp_left([-1, 0]), [0, -1])


def test_arm_base_positions():
    p = arm_placements(S, D)
    assert np.allclose(p["follower_1"].base_xy, [8 * S + D, 4 * S])
    assert np.allclose(p["follower_2"].base_xy, [-D, 4 * S])
    assert np.allclose(p["follower_1"].facing, [-1, 0])
    assert np.allclose(p["follower_2"].facing, [1, 0])


def test_follower_1_left_is_h1_right_is_h8():
    f1 = arm_placements(S, D)["follower_1"]
    h1 = bg.square_center(7, 0, S)
    h8 = bg.square_center(7, 7, S)
    # h1 projects further along the arm's "left" axis than h8 => h1 is to its left.
    assert (h1 - f1.base_xy) @ f1.left > (h8 - f1.base_xy) @ f1.left


def test_follower_2_left_is_a8_right_is_a1():
    f2 = arm_placements(S, D)["follower_2"]
    a1 = bg.square_center(0, 0, S)
    a8 = bg.square_center(0, 7, S)
    assert (a8 - f2.base_xy) @ f2.left > (a1 - f2.base_xy) @ f2.left


def test_board_arm_roundtrip():
    f1 = arm_placements(S, D)["follower_1"]
    pt = bg.square_center(4, 4, S)
    assert np.allclose(f1.arm_to_board(f1.board_to_arm(pt)), pt)

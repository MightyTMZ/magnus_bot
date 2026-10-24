"""Approach-corridor clearance: the arm must not sweep through a neighbouring piece.

Explicit geometry (not configs/*.yaml) so edits to placeholder config can't silently move these.
The integration cases pin a physical fact: reaching across the board to a near-centre square sweeps
the arm over the squares between it and the base, so a tall piece there blocks the approach.
"""

import math

import chess
import numpy as np

from magnus_bot.geometry import board as bg
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.kinematics.so101_ik import SO101Kinematics
from magnus_bot.motion import collision as col

S = 0.057
D = 0.067
Z_GRASP = 0.028
Z_CLEAR = 0.140
KING_R = 0.034 / 2
KING_H = 0.099


def _target_arm(placement, file, rank):
    cx, cy = bg.square_center(file, rank, S)
    fwd, left = placement.board_to_arm(np.array([cx, cy]))
    return np.array([fwd, left, Z_GRASP])


def _obstacle(file, rank, radius=KING_R, height=KING_H):
    return col.PieceObstacle(center=bg.square_center(file, rank, S), radius=radius, height=height)


# --- geometry primitives ---------------------------------------------------

def test_segment_distance_known_cases():
    a0, a1 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    assert math.isclose(col._segment_distance(a0, a1, np.array([0.0, 0, 1]), np.array([1.0, 0, 1])), 1.0)
    assert col._segment_distance(a0, a1, np.array([0.5, -1, 0]), np.array([0.5, 1, 0])) < 1e-9
    d = col._segment_distance(a0, a1, np.array([0.5, 2, 3]), np.array([0.5, 2, 4]))
    assert math.isclose(d, math.hypot(2, 3))


def test_first_collision_respects_radius_and_height():
    seg = [np.array([np.array([0.0, 0.0, 0.05]), np.array([0.2, 0.0, 0.05])])]
    tall = col.PieceObstacle(center=np.array([0.1, 0.0]), radius=0.02, height=0.10)
    short = col.PieceObstacle(center=np.array([0.1, 0.0]), radius=0.02, height=0.03)
    far = col.PieceObstacle(center=np.array([0.1, 0.5]), radius=0.02, height=0.10)
    assert col.first_collision(seg, [tall], margin=0.0) is tall   # arm inside a tall piece
    assert col.first_collision(seg, [short], margin=0.0) is None   # clears a short one
    assert col.first_collision(seg, [far], margin=0.0) is None     # clears a distant one


def test_piece_obstacles_from_board():
    board = chess.Board()  # standard start: 32 pieces
    obstacles = col.piece_obstacles(board, _Config(), exclude=(4, 1))  # drop e2
    assert len(obstacles) == 31
    e2_center = bg.square_center(4, 1, S)
    assert not any(np.allclose(o.center, e2_center) for o in obstacles)
    a1 = next(o for o in obstacles if np.allclose(o.center, bg.square_center(0, 0, S)))
    assert math.isclose(a1.height, 0.050) and math.isclose(a1.radius, 0.028 / 2)


# --- integration: sweep over the board -------------------------------------

def test_near_square_is_clear_even_with_a_full_back_rank():
    """h4 sits right in front of follower_1 -> straight down, nothing between it and the base."""
    kin = SO101Kinematics()
    pl = arm_placements(S, D)["follower_1"]
    obstacles = [_obstacle(f, 0) for f in range(8)]  # full rank 1, none between h4 and the base
    solved = col.min_clear_pitch(kin, pl, _target_arm(pl, 7, 3), obstacles, Z_CLEAR)
    assert solved is not None
    assert solved[0] == 0.0  # grasped top-down


def test_tall_piece_between_target_and_base_blocks_the_approach():
    kin = SO101Kinematics()
    pl = arm_placements(S, D)["follower_1"]
    e4 = _target_arm(pl, 4, 3)
    assert col.min_clear_pitch(kin, pl, e4, [], Z_CLEAR) is not None          # e4 reachable empty
    assert col.min_clear_pitch(kin, pl, e4, [_obstacle(5, 3)], Z_CLEAR) is None  # king on f4 blocks
    assert col.min_clear_pitch(kin, pl, e4, [_obstacle(0, 7)], Z_CLEAR) is not None  # a8 off the line


class _Config:
    """Minimal stand-in exposing the board fields piece_obstacles reads."""

    class board:  # noqa: N801 - mimics config.board attribute access
        square_size = S
        piece_heights = {"pawn": 0.04, "knight": 0.055, "bishop": 0.06,
                         "rook": 0.05, "queen": 0.075, "king": 0.099}
        piece_base_diameters = {"pawn": 0.024, "knight": 0.028, "bishop": 0.028,
                               "rook": 0.028, "queen": 0.032, "king": 0.034}

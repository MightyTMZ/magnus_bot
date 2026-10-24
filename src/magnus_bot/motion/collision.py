"""Approach-corridor clearance: will the arm clip a neighbouring piece on its way to a square?

The reachability map (:mod:`magnus_bot.kinematics.reach`) answers "can the fingertip get there",
treating the arm as a point. That is not enough: a tilted grasp leans the whole arm sideways, so
the forearm/wrist can sweep over an adjacent occupied square and knock the piece on it over. This
module models that.

Model (deliberately simple; tighten as the rig is measured):
  * The arm is the 4-point polyline from :meth:`SO101Kinematics.link_points` -> three line
    segments (upper arm, forearm, wrist-to-tip), converted into the board frame.
  * Each standing piece is a vertical cylinder: centred on its square, ``radius`` from the base
    diameter, ``height`` from the piece height, rising from the table (z = 0).
  * The descent is vertical at a fixed pitch (hover straight down onto the square), so we sample
    the arm posture at several heights from the grasp point up to ``z_clear`` and require every
    sample to clear every obstacle. The piece being grasped (the target square) is excluded.

Collision depends on the LIVE board (occupancy changes every move), so this runs at execution
time with the current position -- it is NOT baked into the static reach map. Not modelled yet
(TODO): gripper jaw width (folded into ``margin`` for now), the horizontal traverse between
squares while carrying a piece, and pitch that varies through the descent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import ArmPlacement
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable

# Default clearance margin (m) added to every piece radius. Stands in for arm-link thickness,
# gripper jaw width, and position error until those are modelled/measured.
DEFAULT_MARGIN = 0.010


@dataclass(frozen=True)
class PieceObstacle:
    """A standing piece as a vertical cylinder in the board frame (meters)."""

    center: np.ndarray  # [x, y] of the square centre
    radius: float       # base_diameter / 2
    height: float       # table (z=0) to the top of the piece


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between 3D segments p1->q1 and p2->q2 (clamped; Ericson RTCD)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    closest1 = p1 + d1 * s
    closest2 = p2 + d2 * t
    return float(np.linalg.norm(closest1 - closest2))


def arm_segments_board(points_arm: np.ndarray, placement: ArmPlacement) -> list[np.ndarray]:
    """Convert an arm-frame link polyline into board-frame 3D segments [(p0, p1), ...].

    ``points_arm`` is the [shoulder, elbow, wrist, tip] array from
    :meth:`SO101Kinematics.link_points` (arm-frame x=forward, y=left, z up). z is shared with the
    board frame; only (x, y) is mapped through the placement.
    """
    board_pts = []
    for px, py, pz in points_arm:
        bx, by = placement.arm_to_board(np.array([px, py]))
        board_pts.append(np.array([bx, by, pz], dtype=float))
    return [np.array([board_pts[i], board_pts[i + 1]]) for i in range(len(board_pts) - 1)]


def first_collision(
    segments: list[np.ndarray], obstacles: list[PieceObstacle], margin: float = DEFAULT_MARGIN
) -> PieceObstacle | None:
    """The first obstacle any arm segment intrudes on (within radius + margin), else ``None``."""
    for obs in obstacles:
        axis_bottom = np.array([obs.center[0], obs.center[1], 0.0])
        axis_top = np.array([obs.center[0], obs.center[1], obs.height])
        threshold = obs.radius + margin
        for seg in segments:
            if _segment_distance(seg[0], seg[1], axis_bottom, axis_top) < threshold:
                return obs
    return None


def approach_clears(
    kinematics: SO101Kinematics,
    placement: ArmPlacement,
    target_arm_xyz: np.ndarray,
    pitch: float,
    obstacles: list[PieceObstacle],
    z_clear: float,
    *,
    elbow: str = "up",
    roll: float = 0.0,
    margin: float = DEFAULT_MARGIN,
    samples: int = 8,
) -> bool:
    """True iff the vertical descent onto ``target_arm_xyz`` at ``pitch`` clears every obstacle.

    ``target_arm_xyz`` is the grasp point in the ARM frame (x=forward, y=left, z=grasp height).
    Samples the arm posture at ``samples`` heights from the grasp point up to ``z_clear``; if any
    height is unreachable at this pitch the descent is treated as not clear (conservative).
    """
    x, y, z_grasp = (float(v) for v in target_arm_xyz)
    for z in np.linspace(z_grasp, z_clear, samples):
        try:
            joints = kinematics.inverse(np.array([x, y, z]), roll=roll, elbow=elbow, pitch=pitch)
        except Unreachable:
            return False
        segments = arm_segments_board(kinematics.link_points(joints), placement)
        if first_collision(segments, obstacles, margin) is not None:
            return False
    return True


def min_clear_pitch(
    kinematics: SO101Kinematics,
    placement: ArmPlacement,
    target_arm_xyz: np.ndarray,
    obstacles: list[PieceObstacle],
    z_clear: float,
    *,
    elbow: str = "up",
    roll: float = 0.0,
    margin: float = DEFAULT_MARGIN,
    max_pitch: float = math.radians(60.0),
    step: float = math.radians(2.0),
    samples: int = 8,
) -> tuple[float, dict[str, float]] | None:
    """Smallest-tilt grasp of ``target_arm_xyz`` whose descent also clears ``obstacles``.

    Every candidate pitch must both solve and pass :func:`approach_clears` against the live
    obstacles. Returns ``(pitch, grasp_joints)`` or ``None`` if no tilt in ``[0, max_pitch]`` both
    reaches and stays clear -- in which case this arm cannot safely take the square and the caller
    must try the other arm or a handoff. NOTE: this checks clearance only; combine with the
    reachability map (joint limits) for a fully admissible pose.
    """
    pitch = 0.0
    while pitch <= max_pitch + 1e-9:
        try:
            joints = kinematics.inverse(target_arm_xyz, roll=roll, elbow=elbow, pitch=pitch)
        except Unreachable:
            pitch += step
            continue
        if approach_clears(
            kinematics, placement, target_arm_xyz, pitch, obstacles, z_clear,
            elbow=elbow, roll=roll, margin=margin, samples=samples,
        ):
            return pitch, joints
        pitch += step
    return None


def piece_obstacles(board, config, exclude: tuple[int, int] | None = None) -> list[PieceObstacle]:
    """Build obstacle cylinders for every occupied square of a python-chess ``board``.

    ``exclude`` (file, rank) drops one square -- the piece being grasped. Piece dimensions come
    from ``config.board`` (PLACEHOLDER until measured); an unknown piece type is skipped.
    """
    import chess

    s = config.board.square_size
    heights = config.board.piece_heights
    diameters = config.board.piece_base_diameters
    obstacles: list[PieceObstacle] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        file, rank = chess.square_file(square), chess.square_rank(square)
        if exclude is not None and (file, rank) == exclude:
            continue
        name = chess.piece_name(piece.piece_type)  # "pawn".."king"
        if name not in heights or name not in diameters:
            continue
        obstacles.append(
            PieceObstacle(
                center=board_geom.square_center(file, rank, s),
                radius=diameters[name] / 2.0,
                height=heights[name],
            )
        )
    return obstacles

"""Per-arm reachability: which squares (and off-board poses) an arm can grasp.

The :class:`ReachabilityMap` is a concrete, injectable data structure (so the arm assigner and
its tests do not need working IK). :func:`compute_reachability` builds one from IK + the joint
limits, offline via scripts/reach_report.py; the result is cached and fed to the assigner.

A point counts as reachable when the top-down IK solves at its height AND (if given) at the
traverse height above it, with every joint inside its URDF limit. That is exactly what the
pick/place primitive needs (descend from z_clear, grasp, lift back to z_clear).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import ArmPlacement
from magnus_bot.kinematics.conventions import model_to_urdf
from magnus_bot.kinematics.model import JointLimit
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable, tilted_tip_target

Square = tuple[int, int]  # (file, rank)

# Pitch sweep for the "lean as little as possible" policy (radians).
DEFAULT_PITCH_STEP = math.radians(2.0)


@dataclass(frozen=True)
class GraspPose:
    """How an arm should grasp a reachable point: the solved approach, not just "yes it reaches".

    ``pitch`` is the tool tilt from straight-down (rad): 0 is top-down, larger leans further for
    points the arm cannot reach top-down. ``roll`` is the jaw orientation, ``elbow`` the IK branch.
    The executor feeds these straight to :meth:`SO101Kinematics.inverse`.
    """

    pitch: float
    roll: float = 0.0
    elbow: str = "up"


@dataclass(frozen=True)
class ReachabilityMap:
    """Which board squares each arm can reach at grasp height, plus pad reachability.

    ``reachable[arm_id]`` is the set of (file, rank) squares that arm can grasp. ``poses`` holds
    the chosen :class:`GraspPose` per (arm, square) -- the approach the executor must command,
    since centre squares are only reachable tilted. ``pad_reach`` is the arm ids that reach the pad.
    """

    reachable: dict[str, set[Square]]
    pad_reach: set[str] = field(default_factory=set)
    poses: dict[str, dict[Square, GraspPose]] = field(default_factory=dict, compare=False)

    def can_reach(self, arm_id: str, file: int, rank: int) -> bool:
        return (file, rank) in self.reachable.get(arm_id, set())

    def arms_reaching(self, file: int, rank: int) -> set[str]:
        """Ids of all arms that can reach square (file, rank)."""
        return {arm_id for arm_id, squares in self.reachable.items() if (file, rank) in squares}

    def can_reach_pad(self, arm_id: str) -> bool:
        return arm_id in self.pad_reach

    def pose(self, arm_id: str, file: int, rank: int) -> GraspPose | None:
        """The grasp pose ``arm_id`` uses for square (file, rank), or ``None`` if unreachable."""
        return self.poses.get(arm_id, {}).get((file, rank))

    def to_dict(self) -> dict:
        """JSON-friendly form (sorted, so cached maps diff cleanly)."""
        return {
            "reachable": {a: sorted(list(sq) for sq in s) for a, s in sorted(self.reachable.items())},
            "pad_reach": sorted(self.pad_reach),
            "poses": {
                a: sorted([f, r, round(p.pitch, 6), round(p.roll, 6), p.elbow] for (f, r), p in d.items())
                for a, d in sorted(self.poses.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> ReachabilityMap:
        poses = {
            a: {(int(f), int(r)): GraspPose(float(pitch), float(roll), str(elbow))
                for f, r, pitch, roll, elbow in rows}
            for a, rows in data.get("poses", {}).items()
        }
        return cls(
            reachable={a: {(int(f), int(r)) for f, r in sq} for a, sq in data["reachable"].items()},
            pad_reach=set(data.get("pad_reach", ())),
            poses=poses,
        )


def reach_pose(
    placement: ArmPlacement,
    kinematics: SO101Kinematics,
    xyz: np.ndarray,
    limits: Mapping[str, JointLimit],
    z_clear: float | None = None,
    max_pitch: float = 0.0,
    to_urdf: Callable[[dict[str, float]], dict[str, float]] = model_to_urdf,
    grip_depth: float | None = None,
) -> GraspPose | None:
    """Smallest-tilt grasp of board-frame ``xyz`` (also clearing z_clear), or ``None`` if unreached.

    Sweeps pitch 0..``max_pitch`` and returns the first that solves top-to-bottom (every height)
    with all joints inside their URDF limit -- the "lean as little as possible" policy.
    ``max_pitch=0`` is a pure top-down check (the default, so existing callers are unchanged).
    ``grip_depth`` is where the stem sits in the jaws (see :func:`tilted_tip_target`).
    """
    x, y = placement.board_to_arm(np.asarray(xyz[:2], dtype=float))
    heights = [float(xyz[2])] if z_clear is None else [float(xyz[2]), z_clear]
    pitch = 0.0
    while pitch <= max_pitch + 1e-9:
        if all(_in_limits(kinematics, x, y, z, pitch, limits, to_urdf, grip_depth) for z in heights):
            return GraspPose(pitch=pitch)
        pitch += DEFAULT_PITCH_STEP
    return None


def _in_limits(kinematics, x, y, z, pitch, limits, to_urdf, grip_depth=None) -> bool:
    try:
        target = tilted_tip_target(np.array([x, y, z]), pitch, kinematics.model.tip_to_pad, grip_depth)
        joints = to_urdf(kinematics.inverse(target, pitch=pitch))
    except Unreachable:
        return False
    return all(limits[n].lo <= v <= limits[n].hi for n, v in joints.items())


def can_reach_point(
    placement: ArmPlacement,
    kinematics: SO101Kinematics,
    xyz: np.ndarray,
    limits: Mapping[str, JointLimit],
    z_clear: float | None = None,
    to_urdf: Callable[[dict[str, float]], dict[str, float]] = model_to_urdf,
    max_pitch: float = 0.0,
) -> bool:
    """True if ``placement``'s arm can grasp board-frame ``xyz`` (and hover at z_clear).

    Top-down by default; pass ``max_pitch`` > 0 to allow a tilted approach for far squares.
    """
    return reach_pose(placement, kinematics, xyz, limits, z_clear, max_pitch, to_urdf) is not None


def compute_reachability(
    placements: dict[str, ArmPlacement],
    kinematics: SO101Kinematics,
    z_grasp: float,
    square_size: float,
    z_clear: float | None = None,
    pad_xyz: np.ndarray | None = None,
    limits: Mapping[str, JointLimit] | None = None,
    max_pitch: float = 0.0,
    grip_depth: float | None = None,
) -> ReachabilityMap:
    """Every square center at ``z_grasp`` (and ``z_clear``) per arm, plus the pad if given.

    ``limits`` default to the kinematics model's (URDF) limits. Pass ``z_clear`` for the map the
    assigner should use; leave it None to see raw grasp-height reach. ``max_pitch=0`` is top-down
    (leaves the d/e centre unreachable with the measured rig); pass max_pitch > 0 to let the arms
    tilt into the centre, recording the chosen per-square pitch in ``poses``.
    """
    limits = limits or kinematics.model.limits
    reachable: dict[str, set[Square]] = {}
    poses: dict[str, dict[Square, GraspPose]] = {}
    for arm_id, placement in placements.items():
        squares: set[Square] = set()
        arm_poses: dict[Square, GraspPose] = {}
        for f in range(board_geom.BOARD_SIZE):
            for r in range(board_geom.BOARD_SIZE):
                xyz = np.append(board_geom.square_center(f, r, square_size), z_grasp)
                pose = reach_pose(placement, kinematics, xyz, limits, z_clear, max_pitch,
                                  grip_depth=grip_depth)
                if pose is not None:
                    squares.add((f, r))
                    arm_poses[(f, r)] = pose
        reachable[arm_id] = squares
        poses[arm_id] = arm_poses
    pad_reach = set()
    if pad_xyz is not None:
        pad_reach = {
            arm_id
            for arm_id, placement in placements.items()
            if can_reach_point(placement, kinematics, pad_xyz, limits, z_clear, max_pitch=max_pitch)
        }
    return ReachabilityMap(reachable=reachable, pad_reach=pad_reach, poses=poses)

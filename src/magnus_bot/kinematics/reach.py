"""Per-arm reachability: which squares (and off-board poses) an arm can grasp.

The :class:`ReachabilityMap` is a concrete, injectable data structure (so the arm assigner and
its tests do not need working IK). :func:`compute_reachability` builds one from IK + the joint
limits, offline via scripts/reach_report.py; the result is cached and fed to the assigner.

A point counts as reachable when the top-down IK solves at its height AND (if given) at the
traverse height above it, with every joint inside its URDF limit. That is exactly what the
pick/place primitive needs (descend from z_clear, grasp, lift back to z_clear).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import ArmPlacement
from magnus_bot.kinematics.conventions import model_to_urdf
from magnus_bot.kinematics.model import JointLimit
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable

Square = tuple[int, int]  # (file, rank)


@dataclass(frozen=True)
class ReachabilityMap:
    """Which board squares each arm can reach at grasp height, plus pad reachability.

    ``reachable[arm_id]`` is the set of (file, rank) squares that arm can grasp. ``pad_reach``
    is the set of arm ids that can reach the shared handoff pad.
    """

    reachable: dict[str, set[Square]]
    pad_reach: set[str] = field(default_factory=set)

    def can_reach(self, arm_id: str, file: int, rank: int) -> bool:
        return (file, rank) in self.reachable.get(arm_id, set())

    def arms_reaching(self, file: int, rank: int) -> set[str]:
        """Ids of all arms that can reach square (file, rank)."""
        return {arm_id for arm_id, squares in self.reachable.items() if (file, rank) in squares}

    def can_reach_pad(self, arm_id: str) -> bool:
        return arm_id in self.pad_reach

    def to_dict(self) -> dict:
        """JSON-friendly form (sorted, so cached maps diff cleanly)."""
        return {
            "reachable": {a: sorted(list(sq) for sq in s) for a, s in sorted(self.reachable.items())},
            "pad_reach": sorted(self.pad_reach),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> ReachabilityMap:
        return cls(
            reachable={a: {(int(f), int(r)) for f, r in sq} for a, sq in data["reachable"].items()},
            pad_reach=set(data.get("pad_reach", ())),
        )


def can_reach_point(
    placement: ArmPlacement,
    kinematics: SO101Kinematics,
    xyz: np.ndarray,
    limits: Mapping[str, JointLimit],
    z_clear: float | None = None,
    to_urdf: Callable[[dict[str, float]], dict[str, float]] = model_to_urdf,
) -> bool:
    """True if ``placement``'s arm can grasp board-frame ``xyz`` top-down (and hover at z_clear)."""
    x, y = placement.board_to_arm(np.asarray(xyz[:2], dtype=float))
    heights = [float(xyz[2])] if z_clear is None else [float(xyz[2]), z_clear]
    for z in heights:
        try:
            joints = to_urdf(kinematics.inverse(np.array([x, y, z])))
        except Unreachable:
            return False
        if not all(limits[n].lo <= v <= limits[n].hi for n, v in joints.items()):
            return False
    return True


def compute_reachability(
    placements: dict[str, ArmPlacement],
    kinematics: SO101Kinematics,
    z_grasp: float,
    square_size: float,
    z_clear: float | None = None,
    pad_xyz: np.ndarray | None = None,
    limits: Mapping[str, JointLimit] | None = None,
) -> ReachabilityMap:
    """Every square center at ``z_grasp`` (and ``z_clear``) per arm, plus the pad if given.

    ``limits`` default to the kinematics model's (URDF) limits. Pass ``z_clear`` for the map the
    assigner should use; leave it None to see raw grasp-height reach.
    """
    limits = limits or kinematics.model.limits
    reachable: dict[str, set[Square]] = {}
    for arm_id, placement in placements.items():
        reachable[arm_id] = {
            (f, r)
            for f in range(board_geom.BOARD_SIZE)
            for r in range(board_geom.BOARD_SIZE)
            if can_reach_point(
                placement,
                kinematics,
                np.append(board_geom.square_center(f, r, square_size), z_grasp),
                limits,
                z_clear,
            )
        }
    pad_reach = set()
    if pad_xyz is not None:
        pad_reach = {
            arm_id
            for arm_id, placement in placements.items()
            if can_reach_point(placement, kinematics, pad_xyz, limits, z_clear)
        }
    return ReachabilityMap(reachable=reachable, pad_reach=pad_reach)

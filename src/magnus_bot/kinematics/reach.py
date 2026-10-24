"""Per-arm reachability: which squares (and off-board poses) an arm can grasp.

The :class:`ReachabilityMap` is a concrete, injectable data structure (so the arm assigner and
its tests do not need working IK). Building one from IK + joint limits is the stub part (M1),
done offline by scripts/reach_report.py; the result is cached and fed to the assigner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from magnus_bot.geometry.frames import ArmPlacement
from magnus_bot.kinematics.so101_ik import SO101Kinematics

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


def compute_reachability(
    placements: dict[str, ArmPlacement],
    kinematics: SO101Kinematics,
    z_grasp: float,
    square_size: float,
) -> ReachabilityMap:
    """STUB (M1): sample every square, ask IK for an in-limits solution at grasp height.

    TODO(M1): for each arm and each (file, rank), map the square center into the arm frame and
    call ``kinematics.inverse(...)``; the square is reachable iff a solution exists. Evaluate
    candidate pad poses the same way. scripts/reach_report.py wraps this and recommends a pad.
    """
    raise NotImplementedError("reachability computation lands in milestone M1")

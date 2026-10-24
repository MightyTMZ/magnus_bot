"""STUB (M1): analytic inverse kinematics for a top-down grasp.

Approach (documented now, implemented in M1):
  * base yaw = atan2(y, x) of the target in the arm frame (shoulder_pan).
  * reduce to a planar 2-link problem in the vertical plane through that yaw: shoulder_lift and
    elbow_flex place the wrist above the target at ``z`` + wrist offset.
  * wrist_flex is fixed for a top-down (vertical) grasp; wrist_roll orients the gripper jaws.
  * clamp every joint to :class:`~magnus_bot.kinematics.model.JointLimit`; raise if unreachable.

A numeric solver (placo/ikpy, the ``ik`` extra) is only ever a test cross-check against this
analytic solution, never on the control path.

All angles radians, positions meters, in the arm frame A_i (origin at the base yaw axis).
"""

from __future__ import annotations

import numpy as np

from magnus_bot.kinematics.model import ARM_JOINTS, SO101Model


class Unreachable(ValueError):
    """Raised when a target lies outside the arm's dexterous workspace or joint limits."""


class SO101Kinematics:
    """Analytic FK/IK for one SO-101 arm. Angles in radians, positions in the arm frame."""

    def __init__(self, model: SO101Model | None = None) -> None:
        self.model = model or SO101Model()

    def forward(self, joints: dict[str, float]) -> np.ndarray:
        """Fingertip position [x, y, z] in the arm frame for the given joint angles (radians).

        TODO(M1): compose the link transforms (base_height, upper_arm, forearm, wrist_to_tip)
        down the chain in ARM_JOINTS order.
        """
        raise NotImplementedError("SO101 forward kinematics lands in milestone M1")

    def inverse(self, target_xyz: np.ndarray, roll: float = 0.0) -> dict[str, float]:
        """Joint angles (radians) placing the fingertip at ``target_xyz`` for a top-down grasp.

        ``roll`` sets wrist_roll (gripper jaw orientation). Raises :class:`Unreachable` if no
        in-limits solution exists. TODO(M1): implement per the module docstring.
        """
        raise NotImplementedError("SO101 inverse kinematics lands in milestone M1")

    def _arm_joint_names(self) -> tuple[str, ...]:
        return ARM_JOINTS

"""SO-101 joint model: names, order, limits, and link lengths.

Joint names and order are exactly LeRobot's SO101Follower bus (see
lerobot/robots/so_follower/so_follower.py):

    shoulder_pan (1), shoulder_lift (2), elbow_flex (3), wrist_flex (4), wrist_roll (5), gripper (6)

The first five are revolute arm joints; ``gripper`` is the end effector (percent 0..100 at the
hardware edge, not a length). Code works in SI (radians); conversion to servo degrees/percent
happens only in :mod:`magnus_bot.hardware`.

Link lengths and joint limits below are PLACEHOLDER until read from the SO-101 URDF and the
per-arm calibration. The URDF-zero <-> servo-calibration offset for each joint is the known
pitfall documented in docs/joint_conventions.md; do not trust these numbers for live motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Order matches LeRobot's bus definition. ``gripper`` is last and handled separately from IK.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

ARM_JOINTS: tuple[str, ...] = JOINT_NAMES[:5]  # the revolute joints IK solves for


@dataclass(frozen=True)
class JointLimit:
    """Soft limit for one joint, in radians (percent for the gripper). PLACEHOLDER values."""

    lo: float
    hi: float


@dataclass(frozen=True)
class SO101Model:
    """Link lengths (meters) and joint limits for the analytic IK.

    Link lengths are derived from the SO-ARM100 URDF (hardware/urdf/so101_new_calib.urdf) by
    composing the joint transforms at the zero pose (see docs/joint_conventions.md). The
    shoulder_lift / elbow_flex / wrist_flex axes are parallel (world +y at zero pose), so the
    arm is a base yaw (shoulder_pan) plus a planar 2-link (upper_arm, forearm) with the wrist.

    Joint LIMITS below are still PLACEHOLDER. TODO(M1): derive them per-arm from the calibration
    (range_min/range_max) via docs/joint_conventions.md. TODO(M1): validate wrist_to_tip's
    effective top-down vertical offset against placo FK (lerobot.model.kinematics.RobotKinematics).
    """

    # Link lengths along the arm chain (meters), from so101_new_calib.urdf at the zero pose.
    base_height: float = 0.1166       # table/base to the shoulder_lift axis
    upper_arm: float = 0.1160         # shoulder_lift -> elbow_flex
    forearm: float = 0.1350           # elbow_flex -> wrist_flex
    wrist_to_tip: float = 0.1605      # wrist_flex -> gripper_frame_link (grasp point)

    limits: dict[str, JointLimit] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.limits is None:
            # PLACEHOLDER symmetric limits; replace with calibration-derived values (M1).
            object.__setattr__(
                self,
                "limits",
                {
                    "shoulder_pan": JointLimit(-math.pi / 2, math.pi / 2),
                    "shoulder_lift": JointLimit(-math.pi / 2, math.pi / 2),
                    "elbow_flex": JointLimit(-math.pi / 2, math.pi / 2),
                    "wrist_flex": JointLimit(-math.pi / 2, math.pi / 2),
                    "wrist_roll": JointLimit(-math.pi, math.pi),
                    "gripper": JointLimit(0.0, 100.0),
                },
            )

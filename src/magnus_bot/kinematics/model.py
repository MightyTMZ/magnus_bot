"""SO-101 joint model: names, order, limits, and link lengths.

Joint names and order are exactly LeRobot's SO101Follower bus (see
lerobot/robots/so_follower/so_follower.py):

    shoulder_pan (1), shoulder_lift (2), elbow_flex (3), wrist_flex (4), wrist_roll (5), gripper (6)

The first five are revolute arm joints; ``gripper`` is the end effector (percent 0..100 at the
hardware edge, not a length). Code works in SI (radians); conversion to servo degrees/percent
happens only in :mod:`magnus_bot.hardware`.

Link lengths and joint limits below come from the SO-101 URDF; per-arm calibration limits are
still to come. The URDF-zero <-> servo-calibration offset for each joint is the known
pitfall documented in docs/joint_conventions.md; do not trust these numbers for live motion.
"""

from __future__ import annotations

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

# Peak joint speed at speed_scale = 1.0, in rad/s (percent/s for the gripper). PLACEHOLDER:
# deliberately well under the STS3215's ~4.7 rad/s no-load speed; tune on the bench (M1).
MAX_JOINT_VELOCITY: dict[str, float] = {
    "shoulder_pan": 1.5,
    "shoulder_lift": 1.5,
    "elbow_flex": 1.5,
    "wrist_flex": 2.0,
    "wrist_roll": 2.0,
    "gripper": 100.0,
}


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
    The shoulder_lift axis sits ``shoulder_forward`` ahead of the yaw axis; the fingertip lies in
    the vertical plane through the yaw axis, so ``atan2`` still gives the exact pan.
    Validated against the URDF chain in tests/test_conventions.py (exact at wrist_roll = 0).

    Joint LIMITS are the URDF's, in URDF-zero radians (see kinematics/conventions.py for the
    IK-model <-> URDF mapping). TODO(M1): tighten per-arm from the calibration
    (range_min/range_max) once the URDF <-> servo check is done (docs/joint_conventions.md).
    """

    # Lengths in the arm's vertical plane (meters), from so101_new_calib.urdf at the zero pose.
    base_height: float = 0.1166       # table/base to the shoulder_lift axis
    shoulder_forward: float = 0.0304  # yaw axis -> shoulder_lift axis, horizontal
    upper_arm: float = 0.1160         # shoulder_lift -> elbow_flex
    forearm: float = 0.1350           # elbow_flex -> wrist_flex
    # wrist_flex -> gripper_frame_link (grasp point), in-plane. The full 3-D length is 0.1605;
    # the other 1.83 cm runs along the pitch axis and only cancels the shoulder's lateral offset.
    wrist_to_tip: float = 0.1594
    # Fingertip (the IK target) -> centre of the jaw pads, back up the tool axis. MEASURED 2026-10-24
    # (board.yaml derives each piece's fingertip grasp_height as its neck height minus this).
    tip_to_pad: float = 0.023

    limits: dict[str, JointLimit] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.limits is None:
            # URDF joint limits (URDF-zero convention, radians); gripper in percent.
            object.__setattr__(
                self,
                "limits",
                {
                    "shoulder_pan": JointLimit(-1.91986, 1.91986),
                    "shoulder_lift": JointLimit(-1.74533, 1.74533),
                    "elbow_flex": JointLimit(-1.69, 1.69),
                    "wrist_flex": JointLimit(-1.65806, 1.65806),
                    "wrist_roll": JointLimit(-2.74385, 2.84121),
                    "gripper": JointLimit(0.0, 100.0),
                },
            )

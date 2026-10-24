"""IK-model <-> URDF-zero joint angle mapping.

The analytic IK (:mod:`magnus_bot.kinematics.so101_ik`) measures angles from +z; the URDF
(hardware/urdf/so101_new_calib.urdf) measures them from its own zero pose. Per joint:

    model = sign * urdf + offset        urdf = sign * (model - offset)       (sign = +/-1)

Derived from the URDF zero pose (docs/joint_conventions.md) and pinned against the URDF chain in
tests/test_conventions.py. ``offset`` is the model angle of each link at the URDF zero pose:
upper arm 0.2438 rad from +z, forearm 1.5323, wrist -> tip line 1.6204. shoulder_pan's URDF axis
points down (-z), hence its sign. wrist_roll is passed through: the IK leaves it at 0.

URDF-zero -> servo angles (what LeRobot commands) is a separate, hardware-verified step.
"""

from __future__ import annotations

from collections.abc import Mapping

# joint: (sign, offset rad)
MODEL_FROM_URDF: dict[str, tuple[int, float]] = {
    "shoulder_pan": (-1, 0.0),
    "shoulder_lift": (1, 0.243783),
    "elbow_flex": (1, 1.288485),
    "wrist_flex": (1, 0.088104),
    "wrist_roll": (1, 0.0),
}


def model_to_urdf(joints: Mapping[str, float]) -> dict[str, float]:
    """IK-model angles -> URDF-zero angles. Joints not in the table (gripper) pass through."""
    out = dict(joints)
    for name, (sign, offset) in MODEL_FROM_URDF.items():
        if name in out:
            out[name] = sign * (out[name] - offset)
    return out


def urdf_to_model(joints: Mapping[str, float]) -> dict[str, float]:
    """URDF-zero angles -> IK-model angles. Joints not in the table (gripper) pass through."""
    out = dict(joints)
    for name, (sign, offset) in MODEL_FROM_URDF.items():
        if name in out:
            out[name] = sign * out[name] + offset
    return out

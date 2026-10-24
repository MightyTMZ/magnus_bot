"""IK-model <-> URDF-zero joint angle mapping.

The analytic IK (:mod:`magnus_bot.kinematics.so101_ik`) measures angles from +z; the URDF
(hardware/urdf/so101_new_calib.urdf) measures them from its own zero pose. Per joint:

    model = sign * urdf + offset        urdf = sign * (model - offset)       (sign = +/-1)

Derived from the URDF zero pose (docs/joint_conventions.md) and pinned against the URDF chain in
tests/test_conventions.py. ``offset`` is the model angle of each link at the URDF zero pose:
upper arm 0.2438 rad from +z, forearm 1.5323, wrist -> tip line 1.6204. shoulder_pan's URDF axis
points down (-z), hence its sign. wrist_roll is passed through: the IK leaves it at 0.

URDF-zero -> servo angles (what LeRobot commands) is per arm and measured on the rig:
:class:`ServoMap`, fitted by :func:`fit_servo_map` (scripts/joint_zero_check.py).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from magnus_bot.kinematics.model import JointLimit

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


# --- URDF zero <-> servo (per arm, measured on the rig) -----------------------------------

ARM_JOINTS = tuple(MODEL_FROM_URDF)  # the five revolute joints, chain order

# A nudge smaller than this (rad, ~3 deg) is too small to read a sign from reliably.
MIN_NUDGE = 0.05


@dataclass(frozen=True)
class ServoMap:
    """One arm's ``servo = sign * urdf + offset`` per joint (radians, like ``LeRobotArm``).

    Fitted on the rig by scripts/joint_zero_check.py and stored in hardware/servo_maps/.
    """

    arm_id: str
    joints: dict[str, tuple[int, float]]  # joint: (sign, offset rad)

    def urdf_to_servo(self, joints: Mapping[str, float]) -> dict[str, float]:
        out = dict(joints)
        for name, (sign, offset) in self.joints.items():
            if name in out:
                out[name] = sign * out[name] + offset
        return out

    def servo_to_urdf(self, joints: Mapping[str, float]) -> dict[str, float]:
        out = dict(joints)
        for name, (sign, offset) in self.joints.items():
            if name in out:
                out[name] = sign * (out[name] - offset)
        return out

    def model_to_servo(self, joints: Mapping[str, float]) -> dict[str, float]:
        """IK model angles -> servo angles: the executor's live ``to_command``."""
        return self.urdf_to_servo(model_to_urdf(joints))

    def servo_limits(self, urdf_limits: Mapping[str, JointLimit]) -> dict[str, JointLimit]:
        """URDF-convention limits expressed in servo angles (a sign flip swaps the ends)."""
        out = dict(urdf_limits)
        for name, (sign, offset) in self.joints.items():
            lo, hi = sorted((sign * urdf_limits[name].lo + offset, sign * urdf_limits[name].hi + offset))
            out[name] = JointLimit(lo, hi)
        return out

    def to_dict(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "joints": {
                n: {"sign": s, "offset_rad": o, "offset_deg": round(math.degrees(o), 2)}
                for n, (s, o) in self.joints.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> ServoMap:
        joints = {n: (int(j["sign"]), float(j["offset_rad"])) for n, j in data["joints"].items()}
        if set(joints) != set(ARM_JOINTS) or any(s not in (1, -1) for s, _ in joints.values()):
            raise ValueError(f"servo map needs sign +/-1 for exactly {ARM_JOINTS}")
        return cls(arm_id=str(data["arm_id"]), joints=joints)


def load_servo_map(path: str | Path) -> ServoMap:
    return ServoMap.from_dict(json.loads(Path(path).read_text()))


class NudgeError(ValueError):
    """A nudge reading cannot be trusted (too small, or a different joint moved most)."""


def fit_servo_map(
    arm_id: str,
    zero: Mapping[str, float],
    nudged: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, float] | None = None,
) -> ServoMap:
    """Fit sign + offset per joint from servo readings (radians).

    ``zero``: readings with the arm in the ``reference`` pose (URDF angles; default URDF zero).
    ``nudged[j]``: readings after moving ONLY joint ``j`` some way in its URDF-positive direction
    from that pose. The sign is the direction the reading moved; the offset makes the reference
    pose read as ``reference``. Raises :class:`NudgeError` if a nudge is too small, or if another
    joint's reading changed more than ``j``'s (a scrambled servo ID or a bumped joint).
    """
    reference = reference or {}
    joints = {}
    for name in ARM_JOINTS:
        reading = nudged[name]
        deltas = {n: reading[n] - zero[n] for n in ARM_JOINTS}
        biggest = max(deltas, key=lambda n: abs(deltas[n]))
        if biggest != name:
            raise NudgeError(
                f"nudging {name} moved {biggest} most ({math.degrees(deltas[biggest]):+.1f} deg vs "
                f"{math.degrees(deltas[name]):+.1f} deg): scrambled servo ID or the wrong joint moved"
            )
        if abs(deltas[name]) < MIN_NUDGE:
            raise NudgeError(f"{name} only moved {math.degrees(deltas[name]):+.1f} deg; nudge >= 3 deg")
        sign = 1 if deltas[name] > 0 else -1
        joints[name] = (sign, zero[name] - sign * reference.get(name, 0.0))
    return ServoMap(arm_id=arm_id, joints=joints)

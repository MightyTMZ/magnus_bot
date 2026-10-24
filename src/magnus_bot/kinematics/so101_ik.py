"""Analytic inverse kinematics for a top-down grasp, plus the matching forward kinematics.

Approach (per CLAUDE.md): base yaw via ``atan2``; a planar 2-link (upper_arm, forearm) in the
vertical plane through that yaw; the wrist is driven so the tool points straight down for a
top-down grasp; ``wrist_roll`` orients the jaws. A numeric solver (placo/ikpy, the ``ik`` extra)
is only ever a cross-check (see tests/test_ik.py), never the control path.

CONVENTION: this module works in an internal *model* convention — angles measured from +z (up),
in radians, in the arm frame A_i (origin on the base yaw axis, +x forward, +y left, +z up). The
shoulder_lift / elbow_flex / wrist_flex axes are parallel (verified from the URDF), so the arm is
a yaw plus a planar 3-link (upper_arm, forearm, wrist_to_tip). :mod:`kinematics.conventions`
maps these model angles to URDF-zero angles, where joint limits apply; the URDF-zero <-> servo
step (docs/joint_conventions.md) is still unverified on hardware, so do NOT feed either
directly to an arm yet.

The planar model includes the shoulder_lift axis's forward offset from the yaw axis. Mapped to
URDF angles (kinematics/conventions.py) it matches the URDF chain to <0.1 mm at wrist_roll = 0
(tests/test_conventions.py); non-zero roll moves the tip up to 7.9 mm (it sits off the roll axis).
"""

from __future__ import annotations

import math

import numpy as np

from magnus_bot.kinematics.model import SO101Model

# IK joint keys this solver produces (model convention, radians). Order matches the arm chain.
IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class Unreachable(ValueError):
    """Raised when a target lies outside the arm's dexterous workspace (planar 2-link reach)."""


class SO101Kinematics:
    """Analytic FK/IK for one SO-101 arm (model convention; see module docstring)."""

    def __init__(self, model: SO101Model | None = None) -> None:
        self.model = model or SO101Model()

    def forward(self, joints: dict[str, float]) -> np.ndarray:
        """Fingertip position [x, y, z] (arm frame, meters) for model-convention joint angles."""
        pan = joints["shoulder_pan"]
        a0 = joints["shoulder_lift"]
        a1 = a0 + joints["elbow_flex"]
        a2 = a1 + joints["wrist_flex"]
        m = self.model
        r = (
            m.shoulder_forward
            + m.upper_arm * math.sin(a0)
            + m.forearm * math.sin(a1)
            + m.wrist_to_tip * math.sin(a2)
        )
        z = (
            m.base_height
            + m.upper_arm * math.cos(a0)
            + m.forearm * math.cos(a1)
            + m.wrist_to_tip * math.cos(a2)
        )
        return np.array([r * math.cos(pan), r * math.sin(pan), z], dtype=float)

    def inverse(self, target_xyz: np.ndarray, roll: float = 0.0, elbow: str = "up") -> dict[str, float]:
        """Model-convention joint angles placing the fingertip at ``target_xyz`` for a top-down grasp.

        ``roll`` sets wrist_roll (jaw orientation). ``elbow`` selects the "up" or "down" branch.
        Raises :class:`Unreachable` if the planar 2-link cannot reach the implied wrist point.
        """
        if elbow not in ("up", "down"):
            raise ValueError(f"elbow must be 'up' or 'down', got {elbow!r}")
        x, y, z = (float(v) for v in target_xyz)
        m = self.model

        pan = math.atan2(y, x)
        r = math.hypot(x, y)

        # Tool points straight down (a2 = pi), so the wrist sits wrist_to_tip ABOVE the tip.
        dr = r - m.shoulder_forward
        dz = (z + m.wrist_to_tip) - m.base_height
        reach = math.hypot(dr, dz)
        if reach > m.upper_arm + m.forearm or reach < abs(m.upper_arm - m.forearm):
            raise Unreachable(
                f"target {target_xyz} needs wrist reach {reach:.4f} m, outside "
                f"[{abs(m.upper_arm - m.forearm):.4f}, {m.upper_arm + m.forearm:.4f}]"
            )

        cos_elbow = (reach**2 - m.upper_arm**2 - m.forearm**2) / (2 * m.upper_arm * m.forearm)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))  # guard float drift at the reach boundary
        q2 = math.acos(cos_elbow)
        # Angles grow away from +z, so q2 > 0 bends the elbow ABOVE the shoulder->wrist line.
        if elbow == "down":
            q2 = -q2

        beta = math.atan2(dr, dz)  # direction from shoulder to wrist, measured from +z
        a0 = beta - math.atan2(m.forearm * math.sin(q2), m.upper_arm + m.forearm * math.cos(q2))
        a1 = a0 + q2
        wrist_flex = math.pi - a1  # so a2 = a1 + wrist_flex = pi (tool vertical, pointing down)

        return {
            "shoulder_pan": pan,
            "shoulder_lift": a0,
            "elbow_flex": q2,
            "wrist_flex": wrist_flex,
            "wrist_roll": roll,
        }

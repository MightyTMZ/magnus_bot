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


def tilted_tip_target(
    top_down_tip: np.ndarray, pitch: float, tip_to_pad: float, grip_depth: float | None = None
) -> np.ndarray:
    """Fingertip target (arm frame) that holds the stem at the right spot in the jaws.

    Grasp heights are fingertip targets for a top-down grasp with the pads ``tip_to_pad`` straight
    above, so the GRIP POINT on the piece is ``top_down_tip + tip_to_pad`` up. The stem is held
    ``grip_depth`` back from the tip along the tool (default ``tip_to_pad``: the pads; less = nearer
    the tips, which saves reach on far squares). Tilting the tool by ``pitch`` keeps that point on
    the grip point: the tip sits ``grip_depth`` beyond it, out along the reach and down. Aiming the
    tip itself at the square instead closed the jaws 10-13 mm in front of the stem (2026-10-24 live).
    """
    depth = tip_to_pad if grip_depth is None else grip_depth
    if pitch == 0.0 and depth == tip_to_pad:
        return np.asarray(top_down_tip, dtype=float)
    x, y, z = (float(v) for v in top_down_tip)
    r = math.hypot(x, y)
    out = depth * math.sin(pitch)
    return np.array([x + out * x / r, y + out * y / r, z + tip_to_pad - depth * math.cos(pitch)])


class Unreachable(ValueError):
    """Raised when a target lies outside the arm's dexterous workspace (planar 2-link reach)."""


class SO101Kinematics:
    """Analytic FK/IK for one SO-101 arm (model convention; see module docstring)."""

    def __init__(self, model: SO101Model | None = None) -> None:
        self.model = model or SO101Model()

    def forward(self, joints: dict[str, float]) -> np.ndarray:
        """Fingertip position [x, y, z] (arm frame, meters) for model-convention joint angles."""
        return self.link_points(joints)[-1]

    def link_points(self, joints: dict[str, float]) -> np.ndarray:
        """The arm as a 4-point polyline [shoulder, elbow, wrist, tip] (arm frame, meters).

        Same model as :meth:`forward` (the tip is the last point), but it also exposes the
        intermediate joint positions so callers can test the whole arm for clearance, not just
        where the fingertip lands (see :mod:`magnus_bot.motion.collision`). The "shoulder" point is
        the shoulder_lift axis, ``shoulder_forward`` ahead of the yaw axis at ``base_height``.
        """
        pan = joints["shoulder_pan"]
        a0 = joints["shoulder_lift"]
        a1 = a0 + joints["elbow_flex"]
        a2 = a1 + joints["wrist_flex"]
        m = self.model
        r_sh = m.shoulder_forward
        z_sh = m.base_height
        r_el = r_sh + m.upper_arm * math.sin(a0)
        z_el = z_sh + m.upper_arm * math.cos(a0)
        r_wr = r_el + m.forearm * math.sin(a1)
        z_wr = z_el + m.forearm * math.cos(a1)
        r_tip = r_wr + m.wrist_to_tip * math.sin(a2)
        z_tip = z_wr + m.wrist_to_tip * math.cos(a2)
        cos_p, sin_p = math.cos(pan), math.sin(pan)
        return np.array(
            [
                [r_sh * cos_p, r_sh * sin_p, z_sh],
                [r_el * cos_p, r_el * sin_p, z_el],
                [r_wr * cos_p, r_wr * sin_p, z_wr],
                [r_tip * cos_p, r_tip * sin_p, z_tip],
            ],
            dtype=float,
        )

    def inverse(
        self,
        target_xyz: np.ndarray,
        roll: float = 0.0,
        elbow: str = "up",
        pitch: float = 0.0,
    ) -> dict[str, float]:
        """Model-convention joint angles placing the fingertip at ``target_xyz`` for a tilted grasp.

        ``pitch`` is the approach tilt of the tool away from straight-down, in radians, in the
        vertical plane the arm reaches along: ``0`` is a pure top-down grasp (tool axis a2 = pi);
        positive values tilt the tool so the fingertip reaches further out (the long wrist_to_tip
        link then adds horizontal reach). The board is NOT fully reachable top-down with the
        measured rig -- the d/e files fall outside both arms' top-down workspace -- so centre
        squares need a small positive pitch (reach.py picks the smallest that works per square).

        ``roll`` sets wrist_roll (jaw orientation). ``elbow`` selects the "up" or "down" branch.
        Raises :class:`Unreachable` if the planar 2-link cannot reach the implied wrist point.
        """
        if elbow not in ("up", "down"):
            raise ValueError(f"elbow must be 'up' or 'down', got {elbow!r}")
        x, y, z = (float(v) for v in target_xyz)
        m = self.model

        pan = math.atan2(y, x)
        r = math.hypot(x, y)

        # Tool axis a2 = pi - pitch (pitch=0 -> straight down). The wrist sits back along that axis
        # from the tip: wrist_to_tip*cos(pitch) above it and wrist_to_tip*sin(pitch) toward the
        # base. Tilting pulls the wrist radially inward, so the 2-link needs less reach while the
        # tip still lands at r -- that is what buys the extra horizontal range.
        dr = r - m.shoulder_forward - m.wrist_to_tip * math.sin(pitch)
        dz = (z + m.wrist_to_tip * math.cos(pitch)) - m.base_height
        reach = math.hypot(dr, dz)
        if reach > m.upper_arm + m.forearm or reach < abs(m.upper_arm - m.forearm):
            raise Unreachable(
                f"target {target_xyz} at pitch {pitch:.3f} rad needs wrist reach {reach:.4f} m, "
                f"outside [{abs(m.upper_arm - m.forearm):.4f}, {m.upper_arm + m.forearm:.4f}]"
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
        wrist_flex = math.pi - pitch - a1  # so a2 = a1 + wrist_flex = pi - pitch (tool tilt)

        return {
            "shoulder_pan": pan,
            "shoulder_lift": a0,
            "elbow_flex": q2,
            "wrist_flex": wrist_flex,
            "wrist_roll": roll,
        }

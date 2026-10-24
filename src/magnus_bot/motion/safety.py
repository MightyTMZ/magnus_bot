"""Command-path safety: clamp every joint target to its limit and bound the speed scale.

These helpers run on EVERY command regardless of mode (sim or live). Safety rule 2
(CLAUDE.md): clamp every joint target to configured limits; speed scale comes from config.
On a fault the policy is HOLD (safety rule 3): stop commanding and hold position; never cut
torque unless the operator says so.
"""

from __future__ import annotations

from magnus_bot.kinematics.model import JointLimit

# Fault policy identifiers (see executor): hold position, do not disable torque.
FAULT_HOLD = "hold"


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    if lo > hi:
        raise ValueError(f"invalid limit: lo={lo} > hi={hi}")
    return max(lo, min(hi, value))


def clamp_joint(name: str, value: float, limits: dict[str, JointLimit]) -> float:
    """Clamp one joint target to its configured limit. Unknown joints are rejected."""
    if name not in limits:
        raise KeyError(f"no limit configured for joint {name!r}")
    lim = limits[name]
    return clamp(value, lim.lo, lim.hi)


def clamp_joints(targets: dict[str, float], limits: dict[str, JointLimit]) -> dict[str, float]:
    """Clamp a full joint-target dict to limits, returning a new dict."""
    return {name: clamp_joint(name, value, limits) for name, value in targets.items()}


def clamp_speed_scale(scale: float) -> float:
    """Bound a trajectory speed scale to (0, 1]. Zero or negative is invalid."""
    if scale <= 0.0:
        raise ValueError(f"speed_scale must be > 0, got {scale}")
    return min(scale, 1.0)

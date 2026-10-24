"""Assign each transfer to an arm, using a reachability map rather than a hard-coded split.

If one arm can reach BOTH endpoints of a transfer, it does it alone. Otherwise the transfer
needs a handoff: arm A (which reaches the source and the pad) picks and places on the pad; arm B
(which reaches the pad and the destination) picks from the pad and places. See
:mod:`magnus_bot.planning.handoff` for the step expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.planning.move_decomposer import BIN, PAD, SQUARE, STASH, Location, Transfer

SINGLE = "single"
HANDOFF = "handoff"


@dataclass(frozen=True)
class Assignment:
    """How one transfer is carried out.

    ``mode == "single"``: ``arm`` does the whole transfer.
    ``mode == "handoff"``: ``pick_arm`` (A) moves src -> pad, ``place_arm`` (B) moves pad -> dst.
    """

    transfer: Transfer
    mode: str
    arm: str | None = None
    pick_arm: str | None = None
    place_arm: str | None = None


class Unassignable(RuntimeError):
    """Raised when no arm (or arm pair) can carry out a transfer with the given reachability."""


def arms_reaching(loc: Location, rmap: ReachabilityMap, all_arms: set[str]) -> set[str]:
    """Set of arm ids that can reach a location.

    Board squares and the pad come from ``rmap``. Each arm drops captures into its own bin, so a
    BIN end does not constrain the choice; the STASH is positioned for both arms.
    TODO(M2): replace the BIN/STASH assumption with measured off-board reachability.
    """
    if loc.kind == SQUARE:
        assert loc.square is not None
        return rmap.arms_reaching(*loc.square) & all_arms
    if loc.kind == PAD:
        return set(rmap.pad_reach) & all_arms
    if loc.kind in (BIN, STASH):
        return set(all_arms)
    raise ValueError(f"unknown location kind: {loc.kind!r}")


def assign_transfer(transfer: Transfer, rmap: ReachabilityMap, arms: set[str]) -> Assignment:
    """Decide single-arm vs. handoff for one transfer. Deterministic tie-break by sorted id."""
    src_arms = arms_reaching(transfer.src, rmap, arms)
    dst_arms = arms_reaching(transfer.dst, rmap, arms)

    both = src_arms & dst_arms
    if both:
        return Assignment(transfer, SINGLE, arm=min(both))

    # Handoff: A reaches src + pad, B reaches pad + dst.
    pad = set(rmap.pad_reach) & arms
    pick_candidates = src_arms & pad
    place_candidates = dst_arms & pad
    if not pick_candidates or not place_candidates:
        raise Unassignable(
            f"cannot assign transfer {transfer.reason} "
            f"({transfer.src.kind}->{transfer.dst.kind}): "
            f"src reachable by {sorted(src_arms)}, dst by {sorted(dst_arms)}, pad by {sorted(pad)}"
        )

    pick_arm = min(pick_candidates)
    # Prefer a different arm for the place leg (a handoff that stays on one arm is pointless).
    place_options = place_candidates - {pick_arm} or place_candidates
    place_arm = min(place_options)
    return Assignment(transfer, HANDOFF, pick_arm=pick_arm, place_arm=place_arm)


def assign_transfers(
    transfers: list[Transfer], rmap: ReachabilityMap, arms: set[str]
) -> list[Assignment]:
    """Assign a whole move's transfers in order."""
    return [assign_transfer(t, rmap, arms) for t in transfers]

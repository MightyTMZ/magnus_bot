"""Expand a handoff assignment into two single-arm legs mediated by the pad.

A cross-arm transfer src -> dst becomes:
    leg A (pick_arm):  src -> pad,  then retract
    leg B (place_arm): pad -> dst,  then retract

The two arms share the pad, so execution MUST interlock: arm B does not descend onto the pad
until arm A has released and retracted clear. One arm moves at a time (see CLAUDE.md
"Concurrency"); the other holds its rest pose. Each leg is itself a single-arm transfer and is
expanded into motion primitives by :mod:`magnus_bot.motion.primitives`.
"""

from __future__ import annotations

from dataclasses import dataclass

from magnus_bot.planning.assigner import HANDOFF, Assignment
from magnus_bot.planning.move_decomposer import Location, Transfer


@dataclass(frozen=True)
class HandoffLeg:
    """One arm's single-arm transfer within a handoff, with its execution ``order`` (0 then 1)."""

    arm: str
    transfer: Transfer
    order: int


def expand_handoff(assignment: Assignment) -> list[HandoffLeg]:
    """Two ordered legs (pick to pad, then pad to place) for a handoff assignment."""
    if assignment.mode != HANDOFF:
        raise ValueError(f"expand_handoff needs a handoff assignment, got {assignment.mode!r}")

    t = assignment.transfer
    pad = Location.pad()
    to_pad = Transfer(t.src, pad, t.piece_symbol, f"{t.reason}:to-pad")
    from_pad = Transfer(pad, t.dst, t.piece_symbol, f"{t.reason}:from-pad")
    return [
        HandoffLeg(arm=assignment.pick_arm, transfer=to_pad, order=0),  # type: ignore[arg-type]
        HandoffLeg(arm=assignment.place_arm, transfer=from_pad, order=1),  # type: ignore[arg-type]
    ]

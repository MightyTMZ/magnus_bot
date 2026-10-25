"""Did a physical transfer do what was planned? Compare the camera's occupancy to the plan.

Everything works on occupancy maps (square -> ``"W"``/``"B"``/``"?"``) so one check serves both
the grasp-tuning loop, whose baseline is whatever the camera saw before the attempt (other
pieces included), and post-move verification against python-chess (:func:`check_board_move`).

Limits: occupancy cannot tell a standing piece from one lying on its square, so a piece knocked
over in place still reads as present. ``"?"`` (occupied, color unknown) matches either color.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import chess

from magnus_bot.vision.observation import Occupancy

UNKNOWN = "?"


class Outcome(str, Enum):
    SUCCESS = "success"  # the board matches the plan exactly
    NOT_PICKED = "not_picked"  # nothing changed: the grasp missed or slipped before the lift
    LOST = "lost"  # the piece left its square and is seen nowhere (still in the gripper? off the board?)
    MISPLACED = "misplaced"  # the piece left its square and turned up on another one
    DISTURBED = "disturbed"  # a square the move should not touch changed (neighbour knocked)
    UNCLEAR = "unclear"  # none of the above


@dataclass(frozen=True)
class TransferCheck:
    """The verdict on one transfer, with the squares that disagree with the plan."""

    outcome: Outcome
    missing: dict[str, str] = field(default_factory=dict)  # expected occupied, seen empty
    unexpected: dict[str, str] = field(default_factory=dict)  # expected empty, seen occupied
    wrong_color: dict[str, tuple[str, str]] = field(default_factory=dict)  # square -> (expected, seen)

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.SUCCESS

    @property
    def retryable(self) -> bool:
        """The board is exactly as before, so the same attempt can simply be repeated."""
        return self.outcome is Outcome.NOT_PICKED

    def describe(self) -> str:
        parts = [self.outcome.value]
        if self.missing:
            parts.append("missing " + ", ".join(f"{s}({c})" for s, c in sorted(self.missing.items())))
        if self.unexpected:
            parts.append("unexpected " + ", ".join(f"{s}({c})" for s, c in sorted(self.unexpected.items())))
        if self.wrong_color:
            parts.append(
                "wrong color " + ", ".join(f"{s}({e}->{o})" for s, (e, o) in sorted(self.wrong_color.items()))
            )
        return "; ".join(parts)


def expected_occupancy(board: chess.Board) -> Occupancy:
    """Occupancy the camera should report for ``board``."""
    return {
        chess.square_name(sq): "W" if piece.color == chess.WHITE else "B"
        for sq, piece in board.piece_map().items()
    }


def relocate(occupancy: Occupancy, src: str, dst: str) -> Occupancy:
    """``occupancy`` after the piece on ``src`` moves to ``dst`` (replacing anything on ``dst``)."""
    if src not in occupancy:
        raise ValueError(f"no piece on {src} to move")
    after = dict(occupancy)
    after[dst] = after.pop(src)
    return after


def neighborhood(squares: Iterable[str], radius: int = 1) -> set[str]:
    """Every square within ``radius`` king moves of any of ``squares``."""
    out = set()
    for name in squares:
        index = chess.parse_square(name)
        f, r = chess.square_file(index), chess.square_rank(index)
        for df in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if 0 <= f + df < 8 and 0 <= r + dr < 8:
                    out.add(chess.square_name(chess.square(f + df, r + dr)))
    return out


def restrict(occupancy: Occupancy, zone: set[str]) -> Occupancy:
    """Only the squares in ``zone``."""
    return {s: c for s, c in occupancy.items() if s in zone}


def changes_outside(before: Occupancy, after: Occupancy, zone: set[str]) -> dict[str, list[str | None]]:
    """Squares outside ``zone`` whose reading changed: detection flicker, not something the arm did."""
    return {s: [before.get(s), after.get(s)] for s in sorted(set(before) | set(after))
            if s not in zone and before.get(s) != after.get(s)}


def _matches(expected: str | None, seen: str | None) -> bool:
    if expected is None or seen is None:
        return expected is None and seen is None
    return UNKNOWN in (expected, seen) or expected == seen


def _agree(a: Occupancy, b: Occupancy, squares: Iterable[str]) -> bool:
    return all(_matches(a.get(s), b.get(s)) for s in squares)


def check_transfer(before: Occupancy, expected: Occupancy, observed: Occupancy) -> TransferCheck:
    """Classify ``observed`` against the planned change from ``before`` to ``expected``."""
    squares = set(before) | set(expected) | set(observed)
    missing = {s: c for s, c in expected.items() if s not in observed}
    unexpected = {s: c for s, c in observed.items() if s not in expected}
    wrong = {
        s: (c, observed[s]) for s, c in expected.items() if s in observed and not _matches(c, observed[s])
    }

    def verdict(outcome: Outcome) -> TransferCheck:
        return TransferCheck(outcome, missing, unexpected, wrong)

    if _agree(expected, observed, squares):
        return verdict(Outcome.SUCCESS)
    if _agree(before, observed, squares):
        return verdict(Outcome.NOT_PICKED)

    moved = {s for s in squares if not _matches(before.get(s), expected.get(s))}
    vacated = {s for s in moved if s not in expected}  # squares the plan empties (the source)
    left = all(s not in observed for s in vacated)  # ...and they really are empty now
    arrived_nowhere = all(s not in observed for s in moved - vacated)  # destination still empty
    changed = {s for s in squares - moved if not _matches(expected.get(s), observed.get(s))}

    if changed:
        only_new = all(s not in before and s in observed for s in changed)
        if left and arrived_nowhere and only_new:
            return verdict(Outcome.MISPLACED)
        return verdict(Outcome.DISTURBED)
    if left and arrived_nowhere:
        return verdict(Outcome.LOST)
    return verdict(Outcome.UNCLEAR)  # e.g. a wrong-colored piece on the destination


def check_board_move(board: chess.Board, move: chess.Move, observed: Occupancy) -> TransferCheck:
    """Check the camera's view after ``move`` was played on the physical ``board``.

    Every square the move changes counts, so castling and en passant need no special case.
    """
    after = board.copy(stack=False)
    after.push(move)
    return check_transfer(expected_occupancy(board), expected_occupancy(after), observed)

"""Closed-loop grasp tuning: try a transfer, let the camera grade it, log it, repeat.

One piece shuttles between two squares (g3 -> g4, then g4 -> g3, ...), so each success resets
the next attempt and both directions get tested. After every attempt the arm is back at rest and
the board reader is asked for a fresh, stable board, which :func:`check_transfer` grades:

* ``success``      the piece now sits on the other square; carry on from there.
* ``not_picked``   the board is unchanged; retry. After ``max_misses`` in a row the setting is
                   given up as bad and the sweep moves on (a search tries bad settings on
                   purpose). After ``max_failed_settings`` give-ups in a row, a setting that
                   worked before is retried once: if it still works the failures were just bad
                   settings; if not (piece lying down? gripper?), the operator is asked.
* anything else    stop. The arm already holds at rest; a human fixes the board, then resumes or
                   quits through ``prompt`` (with no ``prompt``, :class:`TuningStopped` is raised).

:func:`run_sweep` tries a fixed list of settings; :func:`refine_search` searches a tilt range
(then optionally a grasp-height range) coarse to fine; :func:`best_setting` picks the most robust
winner. Hardware stays outside: the caller's ``make_shuttle`` builds the thing that moves the piece
for each :class:`Setting`. Motion faults are deliberately not caught:
:class:`~magnus_bot.orchestration.executor.ExecutionFault` propagates with the arms holding, and
so does :class:`~magnus_bot.vision.observation.VisionUnavailable`.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from magnus_bot.vision.observation import BoardObservation, BoardReader, Occupancy
from magnus_bot.vision.verify import (
    Outcome,
    TransferCheck,
    changes_outside,
    check_transfer,
    neighborhood,
    relocate,
    restrict,
)


@dataclass(frozen=True)
class Setting:
    """One point of the sweep: tool tilt, a grasp-height offset and a sideways grasp offset."""

    pitch_deg: float
    z_offset_mm: float = 0.0
    side_mm: float = 0.0  # + = toward the arm's left (the jaws open to one side of the aim point)

    def label(self) -> str:
        side = f", side {self.side_mm:+g} mm" if self.side_mm else ""
        return f"pitch {self.pitch_deg:g} deg, z {self.z_offset_mm:+g} mm{side}"


class SettingInfeasible(Exception):
    """A setting cannot be planned (unreachable square, joint limit). Raised by ``make_shuttle``."""


class TuningStopped(Exception):
    """The loop needed a human and had no way to ask, or the operator quit."""

    def __init__(self, reason: str, check: TransferCheck | None = None) -> None:
        super().__init__(reason)
        self.check = check


class Shuttle(Protocol):
    def transfer(self, src: str, dst: str) -> None:
        """Move the piece from ``src`` to ``dst`` and bring the arm back to rest."""
        ...


@dataclass
class SettingResult:
    """Outcome counts for one setting, per direction (UCI)."""

    setting: Setting
    skipped: str | None = None
    gave_up: bool = False  # stopped early after max_misses in a row
    outcomes: dict[str, Counter[Outcome]] = field(default_factory=dict)

    def add(self, uci: str, outcome: Outcome) -> None:
        self.outcomes.setdefault(uci, Counter())[outcome] += 1

    @property
    def attempts(self) -> int:
        return sum(sum(c.values()) for c in self.outcomes.values())

    @property
    def successes(self) -> int:
        return sum(c[Outcome.SUCCESS] for c in self.outcomes.values())

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def score(self) -> float:
        """Success rate counting one extra success and one extra failure, so 6/6 outranks 2/2."""
        return (self.successes + 1) / (self.attempts + 2)


def locate(occupancy: Occupancy, a: str, b: str) -> str | None:
    """Which of squares ``a`` / ``b`` holds the piece, if exactly one of them does."""
    if (a in occupancy) == (b in occupancy):
        return None
    return a if a in occupancy else b


def run_sweep(
    settings: Iterable[Setting],
    home: str,
    away: str,
    trials: int,
    make_shuttle: Callable[[Setting], Shuttle],
    reader: BoardReader,
    *,
    log: Callable[[dict], None] = lambda record: None,
    prompt: Callable[[str], str] | None = None,
    shuttle_back: bool = True,
    max_misses: int = 3,
    max_failed_settings: int = 3,
    vision_timeout: float = 8.0,
    watch_radius: int = 1,
    results: list[SettingResult] | None = None,
    clock: Callable[[], float] = time.time,
) -> list[SettingResult]:
    """Run up to ``trials`` graded attempts per setting.

    The piece starts on ``home``. With ``shuttle_back`` it then alternates direction; without,
    every attempt is ``home -> away`` and the operator puts the piece back after each success.
    Only squares within ``watch_radius`` king moves of the two squares are graded: the arm cannot
    disturb a piece farther away, so a change there is detection flicker (logged, not failed).
    Results are appended to ``results`` as they come, so pass your own list to keep them if the
    sweep stops early.
    """
    zone = neighborhood([home, away], watch_radius)
    results = [] if results is None else results

    def ask(message: str, check: TransferCheck | None = None) -> None:
        log({"event": "intervention", "time": clock(), "reason": message})
        if prompt is None:
            raise TuningStopped(message, check)
        answer = prompt(f"{message}, then press Enter to continue (q to quit): ")
        if answer.strip().lower().startswith("q"):
            raise TuningStopped(f"operator quit: {message}", check)

    def attempt(
        shuttle: Shuttle, before: BoardObservation
    ) -> tuple[str, str, BoardObservation, TransferCheck]:
        """Move the piece to the other square and grade the result against ``before``."""
        src = locate(before.occupancy, home, away)
        assert src is not None, "caller checks that exactly one of home/away holds the piece"
        dst = away if src == home else home
        shuttle.transfer(src, dst)
        after = reader.read(fresh=True, timeout=vision_timeout)
        near_before = restrict(before.occupancy, zone)
        check = check_transfer(near_before, relocate(near_before, src, dst), restrict(after.occupancy, zone))
        return src, dst, after, check

    def probe(before: BoardObservation, streak: int) -> BoardObservation | None:
        """Several settings in a row kept missing: bad settings, or a broken setup (piece lying down,
        gripper)? Retry the best setting that has worked. Success -> carry on unattended (returns the
        new board); failure, or nothing has worked yet -> ask the operator (returns None)."""
        worked = best_setting([r for r in results if r.successes])
        if worked is None:
            ask(f"{streak} settings in a row never picked the piece up and none has worked yet "
                "(lying down? gripper?); check it")
            return None
        src, dst, after, check = attempt(make_shuttle(worked.setting), before)
        log({"event": "probe", "time": clock(), **_fields(worked.setting), "uci": src + dst,
             "outcome": check.outcome.value, "detail": check.describe()})
        if check.ok:
            return after
        ask(f"{worked.setting.label()} worked before but now: {check.describe()} "
            "(lying down? gripper?); fix the board", check)
        return None

    before: BoardObservation | None = None
    failed_streak = 0  # settings given up since the last success
    for setting in settings:
        result = SettingResult(setting)
        results.append(result)
        try:
            shuttle = make_shuttle(setting)
        except SettingInfeasible as e:
            result.skipped = str(e)
            log({"event": "skip", "time": clock(), **_fields(setting), "reason": str(e)})
            continue

        misses = 0
        while result.attempts < trials:
            if before is None:
                before = reader.read(fresh=True, timeout=vision_timeout)
            at = locate(before.occupancy, home, away)
            if at is None or (not shuttle_back and at != home):
                where = home if not shuttle_back else f"{home} or {away}"
                ask(f"put the piece on {where}, with the other square empty")
                before = None
                continue

            started = clock()
            src, dst, after, check = attempt(shuttle, before)
            result.add(src + dst, check.outcome)
            log({
                "event": "attempt",
                "time": started,
                "duration_s": round(clock() - started, 2),
                **_fields(setting),
                "uci": src + dst,
                "attempt": result.attempts,
                "outcome": check.outcome.value,
                "detail": check.describe(),
                "piece_on_dst": after.pieces.get(dst),
                "far_flicker": changes_outside(before.occupancy, after.occupancy, zone),
                "frames": [before.frame_id, after.frame_id],
            })
            before = after

            if check.ok:
                misses = failed_streak = 0
                continue
            if not check.retryable:
                ask(f"{src}{dst} at {setting.label()}: {check.describe()}; fix the board", check)
                before = None
                misses = 0
                continue
            misses += 1
            if misses < max_misses:
                continue
            result.gave_up = True
            failed_streak += 1
            log({"event": "give_up", "time": clock(), **_fields(setting), "misses": misses})
            if failed_streak >= max_failed_settings:
                before = probe(before, failed_streak)
                failed_streak = 0
            break
    return results


def refine_search(
    home: str,
    away: str,
    trials: int,
    make_shuttle: Callable[[Setting], Shuttle],
    reader: BoardReader,
    *,
    pitch_range: tuple[float, float],
    z_range: tuple[float, float] | None = None,
    coarse_points: int = 5,
    min_step: tuple[float, float] = (1.0, 1.0),
    side_mm: float = 0.0,
    results: list[SettingResult] | None = None,
    log: Callable[[dict], None] = lambda record: None,
    clock: Callable[[], float] = time.time,
    **sweep_kwargs,
) -> SettingResult | None:
    """Search a tilt range, then (if given) a grasp-height range, coarse to fine; return the winner.

    Along each axis: try ``coarse_points`` evenly spaced values, middle first so the extremes (the
    likely failures) come last; then keep halving the step around that axis' winner
    (:func:`best_setting` over the settings on the axis' line) while the step stays >= ``min_step``
    (degrees, mm). Tilt is searched at z offset 0 (clamped into ``z_range``); grasp height is then
    searched at the winning tilt. Each setting is tried once; every result is appended to
    ``results`` as it comes. ``sweep_kwargs`` go to :func:`run_sweep`. Returns None when no
    setting ever succeeded.
    """
    if coarse_points < 2:
        raise ValueError("coarse_points must be at least 2")
    results = [] if results is None else results

    def run(settings: list[Setting], axis: str, step: float) -> None:
        tried = {r.setting for r in results}
        new = list(dict.fromkeys(s for s in settings if s not in tried))
        log({"event": "search_round", "time": clock(), "axis": axis, "step": round(step, 3),
             "settings": [[s.pitch_deg, s.z_offset_mm] for s in new]})
        run_sweep(new, home, away, trials, make_shuttle, reader, log=log, clock=clock,
                  results=results, **sweep_kwargs)

    def search_axis(axis: int, lo: float, hi: float, start: Setting) -> SettingResult | None:
        name = ("pitch", "z")[axis]
        if hi < lo:
            raise ValueError(f"{name} range is empty: {lo} > {hi}")
        step = (hi - lo) / (coarse_points - 1)
        coarse = [_with(start, axis, lo + i * step) for i in range(coarse_points)]
        middle = (lo + hi) / 2
        run(sorted(coarse, key=lambda s: (abs(_coords(s)[axis] - middle), _coords(s)[axis])), name, step)
        fixed = _coords(start)[1 - axis]
        while True:
            line = [r for r in results if _coords(r.setting)[1 - axis] == fixed]
            best = best_setting(line, scale=min_step)
            if best is None or best.successes == 0 or step / 2 < min_step[axis]:
                return best
            step /= 2
            here = _coords(best.setting)[axis]
            near = [_with(best.setting, axis, here + d) for d in (-step, step)]
            run([s for s in near if lo - 1e-9 <= _coords(s)[axis] <= hi + 1e-9], name, step)

    z0 = 0.0 if z_range is None else min(max(0.0, z_range[0]), z_range[1])
    best = search_axis(0, *pitch_range, Setting(pitch_range[0], z0, side_mm))
    if best is None or best.successes == 0:
        return None
    if z_range is not None:
        best = search_axis(1, *z_range, best.setting)
    return best


def _fields(setting: Setting) -> dict:
    return {"pitch_deg": setting.pitch_deg, "z_offset_mm": setting.z_offset_mm, "side_mm": setting.side_mm}


def _coords(setting: Setting) -> tuple[float, float]:
    return setting.pitch_deg, setting.z_offset_mm


def _with(setting: Setting, axis: int, value: float) -> Setting:
    """``setting`` with coordinate ``axis`` (0 tilt, 1 z) set to ``value``, rounded for clean keys."""
    value = round(value, 2)
    field_name = "pitch_deg" if axis == 0 else "z_offset_mm"
    return dataclasses.replace(setting, **{field_name: value})


def best_setting(
    results: Iterable[SettingResult], scale: Sequence[float] = (1.0, 1.0)
) -> SettingResult | None:
    """The most robust winner among the settings tried.

    Highest :attr:`SettingResult.score` first. Among equals, the one farthest from any worse
    setting (a winner at the edge of the region that works fails at the first wobble), then the
    smaller tilt. ``scale`` is the (degrees, mm) that count as one unit of distance.
    """
    tried = [r for r in results if r.attempts]
    if not tried:
        return None
    top = max(r.score for r in tried)
    tied = [r for r in tried if math.isclose(r.score, top)]
    return max(
        tied,
        key=lambda r: (_margin(r, tried, scale), -abs(r.setting.pitch_deg), -abs(r.setting.z_offset_mm)),
    )


def _margin(r: SettingResult, tried: list[SettingResult], scale: Sequence[float]) -> float:
    """How far ``r`` is from trouble, in ``scale`` units.

    Along each axis, among the settings sharing ``r``'s other coordinate, look outward on both
    sides: the margin on a side is the distance to the first worse-scoring setting, or to the last
    setting tried if none is worse (nothing is known beyond it); no setting at all on a side means
    a margin of 0 there. The result is the smallest side over the axes with any other setting; an
    isolated setting has no known margin (0).
    """
    here = _coords(r.setting)
    sides = []
    for axis in (0, 1):
        line = [q for q in tried if q is not r and _coords(q.setting)[1 - axis] == here[1 - axis]]
        if not line:
            continue
        for direction in (-1, 1):
            beyond = sorted(
                (q for q in line if (_coords(q.setting)[axis] - here[axis]) * direction > 0),
                key=lambda q: abs(_coords(q.setting)[axis] - here[axis]),
            )
            if not beyond:
                sides.append(0.0)
                continue
            worse = next((q for q in beyond if _worse(q, r)), None)
            edge = worse if worse is not None else beyond[-1]
            sides.append(abs(_coords(edge.setting)[axis] - here[axis]) / scale[axis])
    return min(sides) if sides else 0.0


def _worse(a: SettingResult, b: SettingResult) -> bool:
    return a.score < b.score and not math.isclose(a.score, b.score)


def summary_lines(
    results: Iterable[SettingResult],
    scale: Sequence[float] = (1.0, 1.0),
    best: SettingResult | None = None,
) -> list[str]:
    """A human-readable table of the sweep (by tilt, then z), ending with the best setting
    (``best`` if given, e.g. a search's winner, else :func:`best_setting`)."""
    results = sorted(results, key=lambda r: _coords(r.setting))
    lines = []
    for r in results:
        label = f"{r.setting.label():<26}"
        if r.skipped:
            lines.append(f"{label} skipped: {r.skipped}")
        elif not r.attempts:
            lines.append(f"{label} not run")
        else:
            per = "  ".join(
                f"{uci} {c[Outcome.SUCCESS]}/{sum(c.values())}" for uci, c in sorted(r.outcomes.items())
            )
            note = "  (gave up: kept missing)" if r.gave_up else ""
            lines.append(f"{label} {r.successes}/{r.attempts} ok ({r.success_rate:.0%})  [{per}]{note}")
    best = best if best is not None else best_setting(results, scale)
    if best is not None:
        lines.append(f"best: {best.setting.label()} ({best.successes}/{best.attempts} ok)")
    return lines

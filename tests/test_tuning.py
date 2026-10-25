"""Closed-loop grasp tuning: the sweep's state machine, driven by a scripted fake world."""

import pytest

from magnus_bot.orchestration.tuning import (
    Setting,
    SettingInfeasible,
    SettingResult,
    TuningStopped,
    best_setting,
    locate,
    refine_search,
    run_sweep,
    summary_lines,
)
from magnus_bot.vision.observation import BoardObservation
from magnus_bot.vision.verify import Outcome


class ScriptedWorld:
    """A board, and a shuttle whose physical results follow a script.

    Each transfer takes the next effect: "ok" moves the piece, "miss" leaves the board alone,
    "drop" loses the piece, "stray" drops it on h8, "knock" moves it but knocks a1 off. When the
    script runs out every transfer is "ok".
    """

    def __init__(self, occupancy, script=()):
        self.occupancy = dict(occupancy)
        self.script = list(script)
        self.transfers = []
        self.frame = 0

    def transfer(self, src, dst):  # Shuttle
        self.transfers.append(src + dst)
        effect = self.script.pop(0) if self.script else "ok"
        if effect == "miss":
            return
        piece = self.occupancy.pop(src)
        if effect in ("ok", "knock"):
            self.occupancy[dst] = piece
        if effect == "knock":
            self.occupancy.pop("a1", None)
        if effect == "stray":
            self.occupancy["h8"] = piece

    def read(self, fresh=True, timeout=8.0):  # BoardReader
        self.frame += 1
        return BoardObservation(occupancy=dict(self.occupancy), frame_id=self.frame)


class SettingWorld:
    """A board, and shuttles whose grasps work only at settings ``good`` accepts (others miss).

    After ``break_after`` successful transfers everything misses: the piece got knocked flat.
    """

    def __init__(self, occupancy, good, break_after=None):
        self.occupancy = dict(occupancy)
        self.good = good
        self.break_after = break_after
        self.successes = 0
        self.transfers = []  # (setting, uci)
        self.frame = 0

    def shuttle(self, setting):  # make_shuttle
        world = self

        class Shuttle:
            def transfer(self, src, dst):
                world.transfers.append((setting, src + dst))
                broken = world.break_after is not None and world.successes >= world.break_after
                if broken or not world.good(setting):
                    return
                world.occupancy[dst] = world.occupancy.pop(src)
                world.successes += 1

        return Shuttle()

    def read(self, fresh=True, timeout=8.0):  # BoardReader
        self.frame += 1
        return BoardObservation(occupancy=dict(self.occupancy), frame_id=self.frame)


def pitches(world):
    return [s.pitch_deg for s, _ in world.transfers]


def sweep(world, trials=4, **kwargs):
    """One setting (25 deg), g3 <-> g4; returns (results, log records)."""
    log = []
    results = run_sweep([Setting(25)], "g3", "g4", trials, lambda s: world, world, log=log.append, **kwargs)
    return results, log


def result(pitch, ok, total):
    r = SettingResult(Setting(pitch))
    for i in range(total):
        r.add("g3g4", Outcome.SUCCESS if i < ok else Outcome.NOT_PICKED)
    return r


def test_locate_needs_exactly_one_occupied_square():
    assert locate({"g3": "W"}, "g3", "g4") == "g3"
    assert locate({"g4": "W", "a1": "B"}, "g3", "g4") == "g4"
    assert locate({"g3": "W", "g4": "W"}, "g3", "g4") is None
    assert locate({}, "g3", "g4") is None


def test_piece_shuttles_back_and_forth_and_every_attempt_is_logged():
    world = ScriptedWorld({"g3": "W", "a1": "B"})
    results, log = sweep(world, trials=4)
    assert world.transfers == ["g3g4", "g4g3", "g3g4", "g4g3"]
    assert (results[0].successes, results[0].attempts) == (4, 4)
    attempts = [e for e in log if e["event"] == "attempt"]
    assert [e["outcome"] for e in attempts] == ["success"] * 4
    assert attempts[0]["uci"] == "g3g4" and attempts[0]["pitch_deg"] == 25


def test_miss_is_retried_from_the_same_square():
    world = ScriptedWorld({"g3": "W"}, ["miss", "ok"])
    results, _ = sweep(world, trials=3)
    assert world.transfers == ["g3g4", "g3g4", "g4g3"]
    assert results[0].outcomes["g3g4"][Outcome.NOT_PICKED] == 1
    assert results[0].successes == 2


def test_a_setting_that_keeps_missing_is_given_up_and_the_sweep_moves_on():
    world = SettingWorld({"g3": "W"}, good=lambda s: s.pitch_deg == 30)
    log = []
    results = run_sweep([Setting(25), Setting(30)], "g3", "g4", 4, world.shuttle, world, log=log.append)
    assert results[0].gave_up and results[0].attempts == 3  # max_misses, not all 4 trials
    assert results[1].successes == 4 and not results[1].gave_up
    assert [e["event"] for e in log].count("give_up") == 1


def test_failing_streak_probes_a_setting_that_worked_and_carries_on_unattended():
    world = SettingWorld({"g3": "W"}, good=lambda s: s.pitch_deg in (30, 31))
    log = []
    settings = [Setting(p) for p in (30, 20, 22, 24, 31)]
    results = run_sweep(settings, "g3", "g4", 2, world.shuttle, world, max_misses=2, log=log.append)
    probes = [e for e in log if e["event"] == "probe"]
    assert [(p["pitch_deg"], p["outcome"]) for p in probes] == [(30, "success")]
    assert results[-1].successes == 2  # no prompt was needed to get here
    assert pitches(world) == [30, 30, 20, 20, 22, 22, 24, 24, 30, 31, 31]


def test_a_known_good_setting_failing_in_the_probe_asks_the_operator():
    world = SettingWorld({"g3": "W"}, good=lambda s: s.pitch_deg == 30, break_after=2)
    settings = [Setting(p) for p in (30, 20, 22, 24)]
    with pytest.raises(TuningStopped, match="pitch 30 deg, z \\+0 mm worked before but now: not_picked"):
        run_sweep(settings, "g3", "g4", 2, world.shuttle, world, max_misses=2)


def test_failing_streak_with_nothing_that_ever_worked_asks_the_operator():
    world = SettingWorld({"g3": "W"}, good=lambda s: False)
    results = []
    with pytest.raises(TuningStopped, match="none has worked yet"):
        run_sweep([Setting(p) for p in (20, 22, 24)], "g3", "g4", 2, world.shuttle, world,
                  max_misses=2, results=results)
    assert [r.gave_up for r in results] == [True, True, True]  # partial results survive


def test_lost_piece_waits_for_the_operator_then_resumes():
    world = ScriptedWorld({"g3": "W", "a1": "B"}, ["drop"])
    asked = []

    def operator(message):
        asked.append(message)
        world.occupancy["g3"] = "W"  # puts the rook back
        return ""

    results, log = sweep(world, trials=3, prompt=operator)
    assert "lost" in asked[0]
    assert world.transfers == ["g3g4", "g3g4", "g4g3"]
    assert results[0].outcomes["g3g4"][Outcome.LOST] == 1
    assert [e["event"] for e in log].count("intervention") == 1


def test_knocked_bystander_stops_even_though_the_move_worked():
    world = ScriptedWorld({"g3": "W", "a1": "B"}, ["knock"])

    def operator(message):
        world.occupancy["a1"] = "B"  # stands the neighbour back up
        return ""

    results, _ = sweep(world, trials=2, prompt=operator)
    assert results[0].outcomes["g3g4"][Outcome.DISTURBED] == 1
    assert world.transfers == ["g3g4", "g4g3"]  # resumes from where the rook actually is


def test_operator_can_quit():
    world = ScriptedWorld({"g3": "W"}, ["stray"])
    with pytest.raises(TuningStopped, match="operator quit") as excinfo:
        sweep(world, prompt=lambda message: "q")
    assert excinfo.value.check.outcome is Outcome.MISPLACED


def test_missing_piece_is_asked_for_before_any_motion():
    world = ScriptedWorld({})
    asked = []

    def operator(message):
        asked.append(message)
        world.occupancy["g3"] = "W"
        return ""

    sweep(world, trials=1, prompt=operator)
    assert "put the piece on g3 or g4" in asked[0]
    assert world.transfers == ["g3g4"]


def test_missing_piece_without_a_prompt_stops():
    world = ScriptedWorld({})
    with pytest.raises(TuningStopped, match="put the piece on g3 or g4"):
        sweep(world)
    assert world.transfers == []


def test_infeasible_setting_is_skipped_and_logged():
    world = ScriptedWorld({"g3": "W"})

    def make(setting):
        if setting.pitch_deg > 40:
            raise SettingInfeasible("joint limit")
        return world

    log = []
    results = run_sweep([Setting(60), Setting(30)], "g3", "g4", 2, make, world, log=log.append)
    assert results[0].skipped == "joint limit" and results[0].attempts == 0
    assert results[1].successes == 2
    assert log[0]["event"] == "skip" and log[0]["pitch_deg"] == 60


def test_without_shuttle_back_the_operator_resets_after_each_success():
    world = ScriptedWorld({"g3": "W"})
    asked = []

    def operator(message):
        asked.append(message)
        world.occupancy = {"g3": "W"}
        return ""

    sweep(world, trials=2, shuttle_back=False, prompt=operator)
    assert world.transfers == ["g3g4", "g3g4"]
    assert "put the piece on g3," in asked[0]


def test_best_setting_prefers_success_rate_then_robustness():
    # 35 is the last setting tried upward (nothing known past it); 25 has a known margin of 5.
    assert best_setting([result(35, 4, 4), result(25, 4, 4), result(20, 2, 4)]).setting.pitch_deg == 25
    assert best_setting([SettingResult(Setting(20), skipped="joint limit")]) is None


def test_best_setting_picks_the_middle_of_the_working_range_not_its_edge():
    tried = [result(25, 0, 3), result(27.5, 3, 3), result(30, 3, 3), result(32.5, 3, 3), result(35, 0, 3)]
    assert best_setting(tried).setting.pitch_deg == 30


def test_best_setting_does_not_trust_the_edge_of_what_was_tried():
    assert best_setting([result(25, 3, 3), result(30, 3, 3), result(35, 3, 3)]).setting.pitch_deg == 30


def test_more_evidence_outranks_less():
    assert best_setting([result(20, 2, 2), result(40, 6, 6)]).setting.pitch_deg == 40


def test_equal_settings_fall_back_to_the_smaller_tilt():
    assert best_setting([result(20, 3, 3), result(25, 3, 3)]).setting.pitch_deg == 20


def test_search_converges_inside_the_working_window():
    world = SettingWorld({"g3": "W"}, good=lambda s: 27 <= s.pitch_deg <= 34)
    log, results = [], []
    best = refine_search("g3", "g4", 2, world.shuttle, world, pitch_range=(20, 40),
                         max_misses=2, log=log.append, results=results)
    assert best.setting == Setting(30)  # the middle of what worked (27.5..32.5), not an edge
    rounds = [e["settings"] for e in log if e["event"] == "search_round"]
    assert rounds == [
        [[30, 0], [25, 0], [35, 0], [20, 0], [40, 0]],  # coarse, middle first
        [[27.5, 0], [32.5, 0]],
        [[28.75, 0], [31.25, 0]],
    ]
    assert len(results) == 9


def test_search_then_tunes_grasp_height_at_the_winning_tilt():
    world = SettingWorld({"g3": "W"}, good=lambda s: 27 <= s.pitch_deg <= 34 and -1 <= s.z_offset_mm <= 5)
    best = refine_search("g3", "g4", 2, world.shuttle, world, pitch_range=(20, 40), z_range=(-4, 8),
                         max_misses=2)
    assert best.setting == Setting(30, 2)


def test_search_where_nothing_works_returns_none():
    world = SettingWorld({"g3": "W"}, good=lambda s: False)
    assert refine_search("g3", "g4", 2, world.shuttle, world, pitch_range=(20, 40),
                         max_misses=2, max_failed_settings=10) is None


def test_search_rejects_a_single_coarse_point():
    world = SettingWorld({"g3": "W"}, good=lambda s: True)
    with pytest.raises(ValueError, match="coarse_points"):
        refine_search("g3", "g4", 2, world.shuttle, world, pitch_range=(20, 40), coarse_points=1)


def test_summary_lines_show_rates_skips_give_ups_and_the_best_setting():
    gave_up = result(20, 0, 3)
    gave_up.gave_up = True
    lines = summary_lines([result(25, 3, 4), SettingResult(Setting(60), skipped="joint limit"), gave_up])
    assert "gave up" in lines[0]  # sorted by tilt: 20 first
    assert "3/4 ok (75%)" in lines[1] and "[g3g4 3/4]" in lines[1]
    assert "skipped: joint limit" in lines[2]
    assert lines[-1] == "best: pitch 25 deg, z +0 mm (3/4 ok)"


def test_summary_lines_can_name_a_given_winner():
    lines = summary_lines([result(25, 3, 4), result(30, 2, 4)], best=result(30, 2, 4))
    assert lines[-1].startswith("best: pitch 30 deg")

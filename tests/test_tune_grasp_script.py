"""scripts/tune_grasp.py end to end: sim sweeps, the vision link check, and the live preflight gates.

The live tests never reach hardware: each one is refused before the arm is imported or touched.
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

from conftest import BOARD_JSON
from magnus_bot.config import load_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_grasp.py"


@pytest.fixture(scope="module")
def tune():
    spec = importlib.util.spec_from_file_location("tune_grasp", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attempts(log_path):
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    return [r for r in records if r["event"] == "attempt"]


def dead_url():
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def no_prompt(message):
    raise AssertionError(f"should have stopped before prompting: {message}")


def test_sim_sweep_runs_the_real_planner_end_to_end(tune, tmp_path, capsys):
    log = tmp_path / "sim.jsonl"
    assert tune.main(["--pitches", "25", "--trials", "4", "--log", str(log)]) == 0
    rows = attempts(log)
    assert [r["uci"] for r in rows] == ["g3g4", "g4g3", "g3g4", "g4g3"]
    assert {r["outcome"] for r in rows} == {"success"}
    assert all(r["live"] is False and r["arm"] == "follower_1" for r in rows)
    assert "best: pitch 25 deg, z +0 mm (4/4 ok)" in capsys.readouterr().out


def test_too_steep_a_tilt_is_skipped_before_any_motion(tune, tmp_path, capsys):
    log = tmp_path / "sim.jsonl"
    assert tune.main(["--pitches", "85", "--trials", "2", "--log", str(log)]) == 0
    assert "skipped: cannot plan" in capsys.readouterr().out
    assert attempts(log) == []


def test_grasp_heights_that_would_scrape_the_board_are_refused(tune, tmp_path, capsys):
    log = tmp_path / "sim.jsonl"
    assert tune.main(["--pitches", "25", "--z-offsets=-4,0", "--trials", "2", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "z -4 mm" in out and "jaw tips would scrape" in out
    assert {r["z_offset_mm"] for r in attempts(log)} == {0.0}  # only the safe height ran


def test_sim_misses_are_retried(tune, tmp_path):
    log = tmp_path / "sim.jsonl"
    argv = ["--pitches", "25", "--trials", "6", "--sim-miss-rate", "0.5", "--max-misses", "20"]
    assert tune.main([*argv, "--log", str(log)]) == 0
    outcomes = [r["outcome"] for r in attempts(log)]
    assert len(outcomes) == 6 and "not_picked" in outcomes and "success" in outcomes


def test_sim_run_where_nothing_ever_works_stops(tune, tmp_path, capsys):
    argv = ["--pitches", "25,30,35", "--sim-miss-rate", "1.0", "--max-misses", "2"]
    assert tune.main([*argv, "--log", str(tmp_path / "sim.jsonl")]) == 2
    err = capsys.readouterr().err
    assert "3 settings in a row never picked the piece up and none has worked yet" in err


def test_sim_search_finds_confirms_and_saves_the_working_window(tune, tmp_path, config_dir, capsys):
    cfg = config_dir(measured=False)
    argv = ["--search", "--pitch-range", "20,40", "--sim-good-pitch", "27,34", "--trials", "3",
            "--save", "--config-dir", str(cfg), "--log", str(tmp_path / "sim.jsonl")]
    assert tune.main(argv) == 0
    assert "saved rook: pitch 30 deg, z +0 mm, confirmed 10/10" in capsys.readouterr().out
    tuned = load_config(cfg).grasp_tuning["rook"]
    assert (tuned.pitch_deg, tuned.z_offset_mm, tuned.confirmed) == (30.0, 0.0, "10/10")
    assert tuned.tuned_on == "follower_1 g3g4"


def test_sim_save_needs_an_explicit_config_dir(tune, tmp_path):
    with pytest.raises(SystemExit):  # never write simulated results into the real configs/
        tune.main(["--search", "--save", "--log", str(tmp_path / "sim.jsonl")])


def test_winner_that_fails_confirmation_is_not_saved(tune, tmp_path, config_dir, capsys):
    cfg = config_dir(measured=False)
    argv = ["--pitches", "25", "--trials", "2", "--save", "--min-success", "1.01",
            "--config-dir", str(cfg), "--log", str(tmp_path / "sim.jsonl")]
    assert tune.main(argv) == 5
    assert "not saved" in capsys.readouterr().err
    assert not (cfg / "grasp.yaml").exists()


def test_check_vision_prints_the_board(tune, vision_server, capsys):
    assert tune.main(["--check-vision", "--vision-url", vision_server.url]) == 0
    out = capsys.readouterr().out
    assert "vision server OK" in out and "g3: white-rook" in out
    assert "3  . . . . . . W ." in out


def test_check_vision_reports_a_dead_server(tune, capsys):
    assert tune.main(["--check-vision", "--vision-url", dead_url()]) == 1
    assert "not reachable" in capsys.readouterr().err


def test_live_refuses_an_unmeasured_rig(tune, config_dir, capsys):
    assert tune.main(["--live", "--config-dir", str(config_dir(measured=False))], prompt=no_prompt) == 1
    assert "live refused" in capsys.readouterr().err


def test_live_stops_before_hardware_when_vision_is_down(tune, config_dir, capsys):
    argv = ["--live", "--config-dir", str(config_dir(measured=True)), "--vision-url", dead_url()]
    assert tune.main(argv, prompt=no_prompt) == 1
    assert "vision:" in capsys.readouterr().err


def test_live_needs_the_piece_on_the_first_square(tune, config_dir, vision_server, capsys):
    vision_server.reply = (200, {**BOARD_JSON, "occupancy": {"g4": "W"}, "pieces": {}})
    argv = ["--live", "--config-dir", str(config_dir(measured=True)), "--vision-url", vision_server.url]
    assert tune.main(argv, prompt=no_prompt) == 1
    assert "put the rook on g3 and clear g4 first" in capsys.readouterr().err


def test_live_needs_a_typed_yes(tune, config_dir, vision_server, capsys):
    argv = ["--live", "--config-dir", str(config_dir(measured=True)), "--vision-url", vision_server.url]
    assert tune.main(argv, prompt=lambda message: "no") == 1
    assert "aborted" in capsys.readouterr().out

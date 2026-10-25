"""scripts/reach_probe.py end to end: sim diagnoses, and the live gates (refused before any motion)."""

import importlib.util
import json
import re
import socket
from pathlib import Path

import pytest

from conftest import BOARD_JSON

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reach_probe.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("reach_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def dead_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def no_prompt(message):
    raise AssertionError(f"should have stopped before prompting: {message}")


def test_sim_probe_of_a_correct_rig_finds_nothing_wrong(probe, tmp_path, capsys):
    log = tmp_path / "probe.jsonl"
    assert probe.main(["--squares", "g3,h4", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "motors OK" in out and "geometry OK" in out
    rows = records(log)
    assert [r["square"] for r in rows] == ["g3", "h4"]
    assert rows[0]["camera_mm"] == pytest.approx([0.0, 0.0], abs=0.1)
    assert {"commanded", "reached", "camera_squares", "edge_gap_m"} <= set(rows[0])


def test_sim_probe_spots_a_geometry_error(probe, tmp_path, capsys):
    argv = ["--squares", "g3", "--sim-camera-offset-mm=-20,0", "--log", str(tmp_path / "p.jsonl")]
    assert probe.main(argv) == 0
    assert "GEOMETRY: on average the tip lands -20 mm along the reach" in capsys.readouterr().out


def test_sim_probe_spots_motor_sag(probe, tmp_path, capsys):
    argv = ["--squares", "g3", "--sim-sag-deg", "3", "--log", str(tmp_path / "p.jsonl")]
    assert probe.main(argv) == 0
    assert "MOTORS: shoulder_lift ended +3.0 deg" in capsys.readouterr().out


def test_sim_probe_removes_camera_parallax(probe, tmp_path, capsys):
    argv = ["--squares", "g3,h4", "--sim-camera-offset-mm=-25,0", "--sim-parallax-mm", "2,11",
            "--log", str(tmp_path / "p.jsonl")]
    assert probe.main(argv) == 0
    out = capsys.readouterr().out
    assert "parallax removed" in out
    # The sideways mean is ~0 either side of zero (float noise), so its printed sign is not pinned.
    assert re.search(r"GEOMETRY: on average the tip lands -25 mm along the reach and [+-]0 mm sideways", out)
    row = records(tmp_path / "p.jsonl")[0]
    assert row["z2_m"] == 0.04 and row["camera_squares_z2"] is not None


def test_ruler_heights_are_asked_at_each_hold_and_reported(probe, tmp_path, capsys):
    gaps = iter(["6", "5.5"])  # the operator measured 6 and 5.5 mm where 10 was commanded
    argv = ["--squares", "g3,h4", "--ask-height", "--log", str(tmp_path / "p.jsonl")]
    assert probe.main(argv, prompt=lambda message: next(gaps)) == 0
    out = capsys.readouterr().out
    assert "height (ruler): tip is -4.0 mm" in out
    assert "HEIGHT: by ruler the tip sits -4.2 mm from the commanded height" in out
    assert [r["height_error_mm"] for r in records(tmp_path / "p.jsonl")] == [-4.0, -4.5]


def test_single_height_probe_still_works(probe, tmp_path, capsys):
    assert probe.main(["--squares", "g3", "--z2", "0", "--log", str(tmp_path / "p.jsonl")]) == 0
    assert "parallax removed" not in capsys.readouterr().out


def test_second_height_must_sit_between_the_hold_and_the_hover(probe):
    with pytest.raises(SystemExit):
        probe.main(["--z2", "0.08"])


def test_unreachable_squares_are_skipped_and_the_rest_probed(probe, tmp_path, capsys):
    log = tmp_path / "p.jsonl"
    assert probe.main(["--squares", "a8,g3", "--log", str(log)]) == 0
    assert "skipping a8" in capsys.readouterr().err
    assert [r["square"] for r in records(log)] == ["g3"]


def test_a_hold_height_that_would_touch_the_board_is_refused(probe):
    with pytest.raises(SystemExit):
        probe.main(["--z", "0.001"])


def test_live_refuses_an_unmeasured_rig(probe, config_dir, capsys):
    assert probe.main(["--live", "--config-dir", str(config_dir(measured=False))], prompt=no_prompt) == 1
    assert "live refused" in capsys.readouterr().err


def test_live_stops_when_vision_is_down(probe, config_dir, capsys, lerobot_calibration):
    argv = ["--live", "--config-dir", str(config_dir(measured=True)), "--vision-url", dead_url()]
    assert probe.main(argv, prompt=no_prompt) == 1
    assert "vision:" in capsys.readouterr().err


def test_live_refuses_to_lower_the_jaws_onto_a_piece(probe, config_dir, vision_server, capsys,
                                                     lerobot_calibration):
    vision_server.reply = (200, BOARD_JSON)  # a rook on g3
    argv = ["--live", "--squares", "g3", "--config-dir", str(config_dir(measured=True)),
            "--vision-url", vision_server.url]
    assert probe.main(argv, prompt=no_prompt) == 1
    assert "clear g3 first" in capsys.readouterr().err


def test_live_needs_a_typed_yes(probe, config_dir, vision_server, capsys, lerobot_calibration):
    vision_server.reply = (200, {**BOARD_JSON, "occupancy": {}, "pieces": {}})
    argv = ["--live", "--squares", "g3", "--config-dir", str(config_dir(measured=True)),
            "--vision-url", vision_server.url]
    assert probe.main(argv, prompt=lambda message: "no") == 1
    assert "aborted" in capsys.readouterr().out

"""Planning inside the servos' calibrated ranges (2026-10-25: the elbow stalled at its 96.3 deg stop)."""

import importlib.util
import math
from pathlib import Path

import pytest

from conftest import F1_CALIBRATION
from magnus_bot.kinematics.conventions import calibrated_range, load_servo_map, real_urdf_limits
from magnus_bot.kinematics.model import SO101Model

ROOT = Path(__file__).resolve().parents[1]
SMAP = load_servo_map(ROOT / "hardware" / "servo_maps" / "follower_1.json")


def test_calibrated_range_matches_lerobot_degrees():
    # LeRobot degrees mode: (ticks - mid) * 360 / 4095; the elbow's range_max 3065, mid 1970.
    elbow = calibrated_range(F1_CALIBRATION)["elbow_flex"]
    assert math.degrees(elbow.hi) == pytest.approx(96.26, abs=0.01)
    assert elbow.lo == pytest.approx(-elbow.hi)


def test_real_limits_narrow_the_urdf_ones_in_urdf_angles(lerobot_calibration):
    urdf = SO101Model().limits
    real = real_urdf_limits(SMAP, urdf, "follower_1")
    sign, offset = SMAP.joints["elbow_flex"]
    assert sign == 1
    assert real["elbow_flex"].hi == pytest.approx(math.radians(96.26) - offset, abs=1e-3)  # the servo stop
    assert real["elbow_flex"].hi < urdf["elbow_flex"].hi
    for name, limit in real.items():
        assert urdf[name].lo <= limit.lo <= limit.hi <= urdf[name].hi


def test_no_calibration_file_means_no_real_limits():
    assert real_urdf_limits(SMAP, SO101Model().limits, "follower_1") is None  # conftest hides the cache


@pytest.fixture(scope="module")
def tune():
    spec = importlib.util.spec_from_file_location("tune_grasp", ROOT / "scripts" / "tune_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sim_tuning_refuses_what_the_real_elbow_cannot_reach(tune, lerobot_calibration, tmp_path, capsys):
    # 60 deg with the stem at the pads asks the elbow ~98 deg at g4: past its 96.3 deg stop. A 25 deg
    # grasp with the stem 10 mm from the tips stays inside it.
    log = tmp_path / "t.jsonl"
    tune.main(["--uci", "g4g5", "--pitches", "60", "--z-offsets=-9", "--trials", "2", "--log", str(log)])
    out = capsys.readouterr().out
    assert "joint limits: calibrated servo ranges" in out
    assert "skipped: cannot plan" in out
    assert tune.main(["--uci", "g4g5", "--pitches", "25", "--z-offsets=-9", "--grip-depth-mm", "10",
                      "--trials", "2", "--log", str(log)]) == 0
    assert "2/2 ok" in capsys.readouterr().out

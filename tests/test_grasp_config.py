"""Tuned grasps: saving to configs/grasp.yaml, loading them back, and first_move.py using them."""

import importlib.util
import math
from pathlib import Path

import pytest

from magnus_bot.config import TunedGrasp, load_config, save_tuned_grasp

FIRST_MOVE = Path(__file__).resolve().parents[1] / "scripts" / "first_move.py"


def test_untuned_piece_uses_the_measured_height_and_the_default_tilt(config_dir):
    config = load_config(config_dir(measured=False))
    assert config.grasp_tuning == {}
    pitch, z = config.grasp_for("rook", default_pitch=0.3)
    assert pitch == 0.3
    assert z == config.board.grasp_height["rook"]


def test_saved_grasp_round_trips_and_applies_the_offset(config_dir):
    cfg = config_dir(measured=False)
    path = save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=30.0, z_offset_mm=2.0, confirmed="10/10"))
    assert path.read_text().startswith("# Grasp settings found by closed-loop tuning")
    config = load_config(cfg)
    assert config.grasp_tuning["rook"].confirmed == "10/10"
    pitch, z = config.grasp_for("rook")
    assert pitch == pytest.approx(math.radians(30.0))
    assert z == pytest.approx(config.board.grasp_height["rook"] + 0.002)


def test_saving_one_piece_keeps_the_others(config_dir):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=30.0))
    save_tuned_grasp(cfg, "king", TunedGrasp(pitch_deg=10.0, z_offset_mm=-1.0))
    assert set(load_config(cfg).grasp_tuning) == {"rook", "king"}


def test_unknown_pieces_are_rejected(config_dir):
    cfg = config_dir(measured=False)
    with pytest.raises(ValueError, match="unknown piece"):
        save_tuned_grasp(cfg, "dragon", TunedGrasp(pitch_deg=30.0))
    (cfg / "grasp.yaml").write_text("dragon:\n  pitch_deg: 30\n")
    with pytest.raises(ValueError, match="unknown piece"):
        load_config(cfg)


@pytest.fixture(scope="module")
def first_move():
    spec = importlib.util.spec_from_file_location("first_move", FIRST_MOVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_move_defaults_to_the_tuned_grasp(first_move, config_dir, capsys):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=30.0, z_offset_mm=2.0))
    assert first_move.main(["--config-dir", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "30 deg tilt (tuned)" in out
    measured = load_config(cfg).board.grasp_height["rook"]
    assert f"grasp z {(measured + 0.002) * 1000:.1f} mm" in out  # measured height + the tuned 2 mm


def test_first_move_without_tuning_uses_the_old_default(first_move, config_dir, capsys):
    assert first_move.main(["--config-dir", str(config_dir(measured=False))]) == 0
    assert "20 deg tilt (default)" in capsys.readouterr().out


def test_explicit_pitch_overrides_the_tuned_tilt(first_move, config_dir, capsys):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=30.0))
    assert first_move.main(["--config-dir", str(cfg), "--pitch", "25"]) == 0
    assert "25 deg tilt (--pitch)" in capsys.readouterr().out

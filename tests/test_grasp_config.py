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
    assert "30 deg tilt (rook default)" in out
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


def test_grip_depth_is_saved_and_read_back(config_dir):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=50.0, z_offset_mm=-6.0, grip_depth_mm=12.0))
    config = load_config(cfg)
    assert config.grip_depth_for("rook") == pytest.approx(0.012)
    assert config.grip_depth_for("king") is None  # untuned: stem at the pads


def test_square_overrides_are_saved_beside_the_default(config_dir):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=60.0, z_offset_mm=-9.0))
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=20.0, z_offset_mm=-12.0, grip_depth_mm=10.0),
                     squares=["h3", "h4"])
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=61.0, z_offset_mm=-9.0))  # default again
    config = load_config(cfg)
    assert config.grasp_tuning["rook"].pitch_deg == 61.0
    assert set(config.square_grasps["rook"]) == {"h3", "h4"}  # overrides survive a default re-save
    assert config.square_grasps["rook"]["h4"].grip_depth_mm == 10.0


def test_a_move_touching_an_override_square_tries_the_override_first(config_dir):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=60.0, z_offset_mm=-9.0))
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=20.0, z_offset_mm=-12.0, grip_depth_mm=10.0),
                     squares=["h4"])
    config = load_config(cfg)
    base = config.board.grasp_height["rook"]
    g = config.grasp_choices("rook", ["g3", "g4"])
    assert [c.label for c in g] == ["rook default"]
    h = config.grasp_choices("rook", ["g4", "h4"])  # one grasp for both ends: the override wins
    assert [c.label for c in h] == ["rook @ h4", "rook default"]
    assert h[0].pitch == pytest.approx(math.radians(20)) and h[0].z == pytest.approx(base - 0.012)
    assert h[0].grip_depth == pytest.approx(0.010)
    assert config.grasp_choices("king", ["g4", "g5"]) == []  # untuned piece


def test_each_piece_closes_the_gripper_to_its_own_hold(config_dir):
    import dataclasses

    config = load_config(config_dir(measured=False))
    assert config.gripper_closed_for("rook") == config.board.gripper_closed_pct["rook"]
    assert config.gripper_closed_for("pawn") == config.board.gripper_closed_pct["pawn"]  # 3.1, measured
    unmeasured = dataclasses.replace(config, board=dataclasses.replace(
        config.board, gripper_closed_pct={**config.board.gripper_closed_pct, "pawn": None}))
    assert unmeasured.gripper_closed_for("pawn") == config.arms.gripper_closed  # falls back to the default


def test_side_offset_is_saved_with_the_grasp(config_dir):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=60.0, z_offset_mm=-9.0, side_offset_mm=5.0))
    (choice,) = load_config(cfg).grasp_choices("rook", ["f4", "f5"])
    assert choice.side == pytest.approx(0.005)


def test_every_grasp_carries_the_arms_side_offset(config_dir):
    config = load_config(config_dir(measured=False))
    assert config.arms.grasp_side_offset["follower_1"] == pytest.approx(0.003)

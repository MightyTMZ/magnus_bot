"""scripts/play_moves.py: real moves with the tuned grasps (sim pass, easing, live gates)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from magnus_bot.config import TunedGrasp, save_tuned_grasp

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "play_moves.py"


@pytest.fixture(scope="module")
def play():
    spec = importlib.util.spec_from_file_location("play_moves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["play_moves"] = module  # dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tuned(config_dir):
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "pawn", TunedGrasp(pitch_deg=45.0, z_offset_mm=-2.0))
    return cfg


def test_robot_moves_play_in_a_row_with_the_human_passing(play, tuned, capsys):
    assert play.main(["--moves", "g2g3,f2f3", "--config-dir", str(tuned)]) == 0
    out = capsys.readouterr().out
    assert "SIM OK g2g3 (g3)" in out and "SIM OK f2f3 (f3)" in out and "pawn default" in out


def test_a_grasp_that_cannot_reach_a_square_is_eased_steeper(play, tuned, lerobot_calibration, capsys):
    # h4 sits beside follower_1's base: 45 deg would push the elbow past its servo stop.
    assert play.main(["--moves", "h2h4", "--config-dir", str(tuned)]) == 0
    assert "pawn default, eased to" in capsys.readouterr().out


def test_an_untuned_piece_says_so(play, tuned, capsys):
    assert play.main(["--moves", "g1f3", "--config-dir", str(tuned)]) == 1
    assert "the knight has no tuned grasp yet" in capsys.readouterr().err


def test_illegal_moves_are_refused_before_anything(play, tuned):
    with pytest.raises(SystemExit):
        play.main(["--moves", "g2g5", "--config-dir", str(tuned)])


def test_a_whole_game_plays_the_robot_side_and_lets_the_human_play_theirs(play, config_dir, capsys):
    # Scholar's mate: two handoffs through the pad (e3) and a capture into follower_1's bin.
    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "pawn", TunedGrasp(pitch_deg=45.0, z_offset_mm=-2.0))
    save_tuned_grasp(cfg, "bishop", TunedGrasp(pitch_deg=40.0, z_offset_mm=0.0))
    save_tuned_grasp(cfg, "queen", TunedGrasp(pitch_deg=45.0, z_offset_mm=-2.0))
    game = "e2e4,e7e5,f1c4,b8c6,d1h5,g8f6,h5f7"
    assert play.main(["--moves", game, "--config-dir", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "human  e7e5 (e5)" in out and "human  g8f6 (Nf6)" in out
    assert "SIM OK f1c4 (Bc4): follower_1 move:to-pad" in out and "follower_2 move:from-pad c4" in out
    assert "SIM OK h5f7 (Qxf7#): follower_1 capture-victim f7" in out
    assert "all 4 robot move(s)" in out


def test_live_refuses_a_move_that_needs_an_arm_not_allowed(play, config_dir, capsys):
    cfg = config_dir(measured=True)
    save_tuned_grasp(cfg, "bishop", TunedGrasp(pitch_deg=40.0, z_offset_mm=0.0))
    save_tuned_grasp(cfg, "pawn", TunedGrasp(pitch_deg=45.0, z_offset_mm=-2.0))
    argv = ["--live", "--moves", "e2e4,f1c4", "--config-dir", str(cfg)]  # Bc4 is a handoff
    assert play.main(argv, prompt=lambda m: "no") == 1
    assert "needs follower_1, follower_2; --arms allows only follower_1" in capsys.readouterr().err

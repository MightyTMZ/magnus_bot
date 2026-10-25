"""scripts/play_moves.py: real moves with the tuned grasps (sim pass, easing, live gates)."""

import importlib.util
from pathlib import Path

import pytest

from magnus_bot.config import TunedGrasp, save_tuned_grasp

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "play_moves.py"


@pytest.fixture(scope="module")
def play():
    spec = importlib.util.spec_from_file_location("play_moves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
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


def test_live_refuses_moves_off_the_board(play, config_dir, capsys):
    cfg = config_dir(measured=True)
    save_tuned_grasp(cfg, "pawn", TunedGrasp(pitch_deg=45.0, z_offset_mm=-2.0))
    fen = "rnbqkbnr/ppppppp1/8/8/8/7p/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # g2xh3 captures
    assert play.main(["--live", "--moves", "g2h3", "--fen", fen, "--config-dir", str(cfg)],
                     prompt=lambda m: "no") == 1
    assert "leaves the board" in capsys.readouterr().err

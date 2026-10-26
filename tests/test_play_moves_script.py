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


def test_the_saved_scholars_mate_plans_with_both_arms(play, config_dir, capsys):
    cfg = config_dir(measured=False)
    for piece, pitch in (("pawn", 45.0), ("bishop", 40.0), ("queen", 45.0)):
        save_tuned_grasp(cfg, piece, TunedGrasp(pitch_deg=pitch, z_offset_mm=0.0))
    assert play.main(["--game", "scholars-mate", "--config-dir", str(cfg)]) == 0
    assert "SIM OK h5f7 (Qxf7#)" in capsys.readouterr().out


class _Board:
    """A vision reader that shows a sequence of boards: what the camera sees over time."""

    def __init__(self, frames):
        self.frames = list(frames)

    def read(self, fresh=True, timeout=8.0, near=None):
        from magnus_bot.vision.observation import BoardObservation

        return BoardObservation(occupancy=self.frames.pop(0) if len(self.frames) > 1 else self.frames[0])


def test_the_humans_move_is_taken_once_the_camera_sees_it_twice(play, monkeypatch):
    import types

    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    board.push_uci("e2e4")
    move = chess.Move.from_uci("e7e5")
    before = expected_occupancy(board)
    after = board.copy()
    after.push(move)
    done = expected_occupancy(after)
    hand = {k: v for k, v in before.items() if k != "e7"}  # pawn lifted, not set down yet
    reader = _Board([before, hand, done, done])
    args = types.SimpleNamespace(human_timeout=5.0, vision_timeout=1.0)
    check = play.wait_for_human(reader, board, move, "e5", args)
    assert check is not None and check.outcome.value == "success"
    assert reader.frames == [done]  # it waited for the second matching look


def test_waiting_for_the_human_gives_up_after_the_timeout(play, monkeypatch):
    import types

    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    board.push_uci("e2e4")
    reader = _Board([expected_occupancy(board)])  # nobody moves
    args = types.SimpleNamespace(human_timeout=0.05, vision_timeout=1.0)
    check = play.wait_for_human(reader, board, chess.Move.from_uci("e7e5"), "e5", args)
    assert check is not None and check.outcome.value != "success"


def test_the_next_button_passes_for_the_human(play, monkeypatch):
    import types

    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    board.push_uci("e2e4")
    reader = _Board([expected_occupancy(board)])  # nobody moves
    args = types.SimpleNamespace(human_timeout=5.0, vision_timeout=1.0)
    presses = iter([True, False, False, True])  # a stale press (dropped), two looks, then N
    check = play.wait_for_human(reader, board, chess.Move.from_uci("e7e5"), "e5", args,
                                skip=lambda: next(presses))
    assert check is play.PASSED
    assert next(presses, None) is None  # it passed on the fresh press, not the stale one


def test_a_missing_piece_after_a_robot_move_only_warns_until_fixed(play, monkeypatch, capsys):
    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy()
    after.push(move)
    done = expected_occupancy(after)
    lost = {k: v for k, v in done.items() if k != "e4"}  # the pawn is nowhere to be seen
    reader = _Board([lost, lost, done])
    ok, seen = play.wait_for_robot_move(reader, board, move, set(), 1.0, grace=5.0)
    assert ok and seen == done
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1 and "check e4" in err  # warned once, the game went on


def test_the_game_stops_once_the_board_stays_wrong_for_the_grace(play, monkeypatch, capsys):
    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    reader = _Board([expected_occupancy(board)])  # the pawn never moved
    ok, _ = play.wait_for_robot_move(reader, board, move, set(), 1.0, grace=0.05)
    assert not ok
    assert "not corrected within 0.05 s" in capsys.readouterr().err


def test_any_legal_move_is_seen_through_a_steady_camera_misread(play, monkeypatch):
    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    board.push_uci("e2e4")
    misread = dict(expected_occupancy(board), d4="W")  # the camera always sees a ghost on d4
    after = board.copy()
    after.push_uci("g8f6")
    played = dict(expected_occupancy(after), d4="W")
    reader = _Board([misread, played, played])
    assert play.wait_for_any_move(reader, board, 5.0, 1.0).uci() == "g8f6"


def test_a_board_no_legal_move_explains_warns_then_stops_after_the_grace(play, monkeypatch, capsys):
    import chess

    from magnus_bot.vision.verify import expected_occupancy

    monkeypatch.setattr(play, "HUMAN_POLL", 0.0)
    board = chess.Board()
    board.push_uci("e2e4")
    start = expected_occupancy(board)
    knocked = {k: v for k, v in start.items() if k != "a8"}  # a rook vanished: no legal move
    reader = _Board([start, knocked])
    assert play.wait_for_any_move(reader, board, 0.0, 1.0, grace=0.05) is None
    err = capsys.readouterr().err
    assert "WARNING: the board shows no legal move (a8 changed)" in err
    assert "not corrected within 0.05 s" in err

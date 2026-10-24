"""Reachability from IK + URDF limits (kinematics/reach.py) and the reach_report script."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from magnus_bot.cli import _full_reach_map
from magnus_bot.config import load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.reach import ReachabilityMap, can_reach_point, compute_reachability
from magnus_bot.kinematics.so101_ik import SO101Kinematics
from magnus_bot.orchestration.executor import Executor, build_move_program


@pytest.fixture(scope="module")
def rig():
    config = load_config()
    return config, arm_placements(config.board.square_size, config.arms.edge_gap), SO101Kinematics()


def _rmap(rig, z_clear=None, pad_xyz=None):
    config, placements, kin = rig
    return compute_reachability(
        placements, kin, config.arms.z_grasp, config.board.square_size, z_clear=z_clear, pad_xyz=pad_xyz
    )


def test_each_arm_reaches_its_own_side_only(rig):
    rmap = _rmap(rig)
    assert rmap.can_reach("follower_1", 7, 3)  # h4, right in front of follower_1
    assert rmap.can_reach("follower_2", 0, 3)  # a4, right in front of follower_2
    assert not rmap.can_reach("follower_1", 0, 3)
    assert not rmap.can_reach("follower_2", 7, 3)


def test_arms_are_mirror_images(rig):
    # The arms face each other across the board, so their maps mirror across the d/e line.
    rmap = _rmap(rig)
    mirrored = {(7 - f, r) for f, r in rmap.reachable["follower_1"]}
    assert mirrored == rmap.reachable["follower_2"]


def test_rig_numbers_top_down(rig):
    # Pins HANDOVER Known issues: REACH (top-down, URDF limits). Update when the geometry changes.
    config = rig[0]
    grasp = _rmap(rig)
    assert len(set().union(*grasp.reachable.values())) == 48
    assert not grasp.reachable["follower_1"] & grasp.reachable["follower_2"]
    hover = _rmap(rig, z_clear=config.arms.z_clear)
    assert all(not squares for squares in hover.reachable.values())


def test_hover_requirement_only_removes_squares(rig):
    grasp, hover = _rmap(rig), _rmap(rig, z_clear=0.06)
    for arm_id in grasp.reachable:
        assert hover.reachable[arm_id] <= grasp.reachable[arm_id]


def test_pad_reach(rig):
    _, placements, kin = rig
    base = placements["follower_1"].base_xy
    near_f1 = np.array([base[0] - 0.10, base[1] - 0.12, 0.02])  # off-board, beside follower_1
    rmap = _rmap(rig, z_clear=0.06, pad_xyz=near_f1)
    assert rmap.pad_reach == {"follower_1"}
    assert can_reach_point(placements["follower_1"], kin, near_f1, kin.model.limits, 0.06)


def test_computed_squares_execute_in_sim(rig):
    # Every square the map calls reachable must actually plan in the executor (same IK + limits).
    config = rig[0]
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=0.06))
    rmap = _rmap(rig, z_clear=0.06)
    arms = {a: SimArm(a) for a in config.arms.arms}
    for arm in arms.values():
        arm.connect()
    executor = Executor(config, arms, _full_reach_map(config), simulate=True)
    for f, r in sorted(rmap.reachable["follower_1"]):
        board = chess.Board(None)
        board.set_piece_at(chess.square(f, r), chess.Piece(chess.KING, chess.WHITE))
        dst = (f, r + 1) if (f, r + 1) in rmap.reachable["follower_1"] else (f, r - 1)
        move = chess.Move(chess.square(f, r), chess.square(*dst))
        executor.plan(build_move_program(board, move, rmap, config))  # raises if not commandable


def test_map_round_trips_through_json():
    rmap = ReachabilityMap(reachable={"follower_1": {(7, 3), (6, 4)}, "follower_2": set()},
                           pad_reach={"follower_1"})
    assert ReachabilityMap.from_dict(json.loads(json.dumps(rmap.to_dict()))) == rmap


def test_pitch_opens_squares_top_down_cannot_reach(rig):
    """Top-down leaves 16 squares unreachable; a tilt opens some of them (the pitch fix)."""
    config, placements, kin = rig
    s, zg = config.board.square_size, config.arms.z_grasp
    top_union = set().union(*compute_reachability(placements, kin, zg, s).reachable.values())
    assert len(top_union) == 48  # matches test_rig_numbers_top_down

    tilt = compute_reachability(placements, kin, zg, s, max_pitch=np.radians(50))
    tilt_union = set().union(*tilt.reachable.values())
    newly = tilt_union - top_union
    assert newly, "a tilted approach should reach squares top-down cannot"
    # every newly-opened square carries a genuinely tilted pose
    for f, r in newly:
        assert tilt.pose(min(tilt.arms_reaching(f, r)), f, r).pitch > 0.0


def test_pitch_leans_only_where_needed(rig):
    config, placements, kin = rig
    tilt = compute_reachability(placements, kin, config.arms.z_grasp, config.board.square_size,
                                max_pitch=np.radians(50))
    assert tilt.pose("follower_1", 7, 3).pitch == 0.0  # h4 in front of follower_1 stays top-down


def test_reach_report_runs_and_writes_the_map(tmp_path, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "reach_report.py"
    spec = importlib.util.spec_from_file_location("reach_report", script)
    reach_report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reach_report)
    out = tmp_path / "reach.json"
    assert reach_report.main(["--z-clear", "0.06", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Grasp + hover at z_clear (0.06 m)" in text and "Pad search" in text
    assert ReachabilityMap.from_dict(json.loads(out.read_text())).reachable["follower_1"]

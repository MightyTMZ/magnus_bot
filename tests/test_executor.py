"""Executor: program building (dry-run) and joint execution (simulate, live gates)."""

import dataclasses
import math

import chess
import numpy as np
import pytest

from magnus_bot.cli import _full_reach_map
from magnus_bot.config import ArmCorrection, load_config
from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import model_to_urdf, urdf_to_model
from magnus_bot.kinematics.model import JointLimit
from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable
from magnus_bot.orchestration.executor import (
    VERIFY_TOLERANCE,
    ExecutionFault,
    Executor,
    JointLimitError,
    build_move_program,
    check_limits,
)
from magnus_bot.orchestration.probe import split_mm


def test_build_program_normal_move_is_one_sequence():
    config = load_config()
    rmap = _full_reach_map(config)
    program = build_move_program(chess.Board(), chess.Move.from_uci("e2e4"), rmap, config)
    assert len(program) == 1
    assert program[0].transfer.reason == "move"
    assert len(program[0].steps) == 8  # full pick/place primitive


def test_build_program_capture_has_victim_then_move():
    config = load_config()
    rmap = _full_reach_map(config)
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    program = build_move_program(board, chess.Move.from_uci("e4d5"), rmap, config)
    assert [p.transfer.reason for p in program] == ["capture-victim", "move"]


def test_executor_dry_run_returns_program_without_moving():
    config = load_config()
    rmap = _full_reach_map(config)
    arms = {a: SimArm(a) for a in config.arms.arms}
    executor = Executor(config, arms, rmap, live=False)
    program = executor.execute_move(chess.Board(), chess.Move.from_uci("e2e4"))
    assert program and all(not arm.command_log for arm in arms.values())  # dry-run commands nothing


# --- joint execution (simulate / live gates) -------------------------------


ROOK_H4 = chess.Board("4k3/8/8/8/7R/8/8/4K3 w - - 0 1")  # h4h5 is within follower_1's reach
H4H5 = chess.Move.from_uci("h4h5")


def _low_clear():
    """Rig config with a 6 cm traverse. The rig's 14 cm is above what a top-down grasp can reach
    within the URDF wrist limit (max ~9.6 cm anywhere; HANDOVER Known issues: REACH)."""
    config = load_config()
    return dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=0.06))


def _sim(config, **kwargs):
    arms = {a: SimArm(a) for a in config.arms.arms}  # URDF limits, like the executor's default
    for arm in arms.values():
        arm.connect()
    return arms, Executor(config, arms, _full_reach_map(config), simulate=True, **kwargs)


def test_simulate_moves_only_the_assigned_arm_and_leaves_it_up():
    from magnus_bot.orchestration.executor import VIA_REACH

    config = _low_clear()
    arms, executor = _sim(config)
    executor.execute_move(ROOK_H4, H4H5)
    assert arms["follower_1"].command_log
    assert not arms["follower_2"].command_log  # one arm at a time; the other is untouched
    # It ends up at travel height near its own board edge, not back down where it started.
    tip = config.arms.correction("follower_1").undo(
        SO101Kinematics().forward(urdf_to_model(arms["follower_1"].read_joints())))
    assert np.hypot(tip[0], tip[1]) == pytest.approx(VIA_REACH, abs=1e-3)
    assert tip[2] == pytest.approx(config.arms.z_clear, abs=1e-3)


def test_after_letting_go_on_the_pad_the_jaws_only_go_up(config_dir):
    # 2026-10-25: after placing the bishop on e3, the wrist swung the open jaws down and back into it.
    from magnus_bot.config import TunedGrasp, save_tuned_grasp

    cfg_dir = config_dir(measured=False)
    save_tuned_grasp(cfg_dir, "bishop", TunedGrasp(pitch_deg=40.0, z_offset_mm=0.0))
    config = load_config(cfg_dir)
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=0.10))
    arms, _ = _sim(config)
    executor = Executor(config, arms, _full_reach_map(config), simulate=True, tuned_grasps=True)
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    # follower_1 reaches f1, follower_2 reaches c4, both reach the pad: Bc4 is a handoff via e3.
    rmap = ReachabilityMap(reachable={"follower_1": {(5, 0)}, "follower_2": {(2, 3)}},
                           pad_reach={"follower_1", "follower_2"})
    program = build_move_program(board, chess.Move.from_uci("f1c4"), rmap, config)
    to_pad = next(s for s in program if s.transfer.reason.endswith("to-pad"))
    traj = executor.plan([to_pad])[0]
    at = {phase: i for i, phase in traj.checkpoints if phase != "dwell"}
    kin, tp = SO101Kinematics(), SO101Kinematics().model.tip_to_pad
    pads = []
    for sample in traj.samples[at["release"]:]:
        pts = kin.link_points(urdf_to_model(sample))
        pads.append(pts[-1] - tp * (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2]))
    heights = [p[2] for p in pads]
    up = next(i for i, h in enumerate(heights) if h >= config.arms.z_clear)
    rising = heights[:up + 1]
    assert all(b >= a - 1e-4 for a, b in zip(rising, rising[1:], strict=False)), "the jaws dipped while low"
    assert min(heights[up:]) >= config.arms.z_clear - 0.003  # and never come back down below travel height


def test_simulated_grip_happens_at_the_source_square():
    config = _low_clear()
    arms, executor = _sim(config)
    executor.execute_move(ROOK_H4, H4H5)
    closed = config.arms.gripper_closed
    grip = next(c for c in arms["follower_1"].command_log if abs(c["gripper"] - closed) < 1e-6)
    # FK of the first fully-closed command, mapped back to the board, is h4 at grasp height.
    tip = SO101Kinematics().forward(urdf_to_model(grip))  # commands are URDF-zero angles
    placement = arm_placements(config.board.square_size, config.arms.edge_gap)["follower_1"]
    # The model tip is commanded off the square by the arm's measured landing error (arms.*_correction):
    # undoing that correction is where the real tip lands, which must be h4 at grasp height.
    real = config.arms.correction("follower_1").undo(tip)
    xy = placement.arm_to_board(real[:2])
    # 1e-4 m (0.1 mm): the model<->URDF round-trip (model_to_urdf in the command, urdf_to_model
    # here) carries ~1e-5 float drift that varies with the arm configuration; far below any
    # physical relevance (and 50x tighter than the grasp tolerance).
    # ...aimed arms.grasp_side_offset to the arm's left of the centre (the jaws open to one side).
    centre = placement.board_to_arm(board_geom.square_center(7, 3, config.board.square_size))
    side = config.arms.grasp_side_offset.get("follower_1", 0.0)
    aim = ArmCorrection(side=side).apply(np.append(centre, config.arms.z_grasp))
    assert xy == pytest.approx(placement.arm_to_board(aim[:2]), abs=1e-4)
    assert real[2] == pytest.approx(config.arms.z_grasp, abs=1e-4)


def test_unreachable_move_aborts_before_any_motion():
    config = load_config()
    arms, executor = _sim(config)
    with pytest.raises(Unreachable, match="follower_1"):
        executor.execute_move(chess.Board(), chess.Move.from_uci("e2e4"))  # e2 is out of reach
    assert all(not arm.command_log for arm in arms.values())


def test_unmapped_model_angles_are_rejected_before_motion():
    # Without model -> URDF mapping, IK angles (wrist_flex ~3.1 rad tool-down) break URDF limits.
    config = load_config()
    arms, executor = _sim(config, to_command=dict)
    with pytest.raises(JointLimitError, match="wrist_flex"):
        executor.execute_move(ROOK_H4, H4H5)
    assert all(not arm.command_log for arm in arms.values())


def test_rig_z_clear_breaks_the_wrist_limit_top_down():
    # Pins the REACH finding: hovering at the rig's z_clear needs wrist_flex > 95 deg.
    config = load_config()
    arms, executor = _sim(config)
    with pytest.raises(JointLimitError, match="hover: wrist_flex"):
        executor.execute_move(ROOK_H4, H4H5)
    assert all(not arm.command_log for arm in arms.values())


def test_check_limits_names_every_offender():
    with pytest.raises(JointLimitError, match="a=2.000.*b=-3.000"):
        check_limits({"a": 2.0, "b": -3.0, "c": 0.0}, {n: JointLimit(-1, 1) for n in "abc"})


class _FailingArm(SimArm):
    def __init__(self, *args, fail_at: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_at, self.sent, self.held = fail_at, 0, False

    def send_joints(self, targets):
        self.sent += 1
        if self.sent == self.fail_at:
            raise OSError("bus timeout")
        return super().send_joints(targets)

    def hold(self):
        self.held = True
        return super().hold()


def test_fault_mid_motion_holds_every_arm():
    config = _low_clear()
    arms = {
        "follower_1": _FailingArm("follower_1", fail_at=10),
        "follower_2": _FailingArm("follower_2", fail_at=-1),
    }
    for arm in arms.values():
        arm.connect()
    executor = Executor(config, arms, _full_reach_map(config), simulate=True)
    with pytest.raises(ExecutionFault, match="bus timeout"):
        executor.execute_move(ROOK_H4, H4H5)
    assert arms["follower_1"].held and arms["follower_2"].held


def test_live_refuses_without_joint_mapping():
    config = dataclasses.replace(load_config(), measured=True)
    with pytest.raises(RuntimeError, match="joint mapping"):
        Executor(config, {}, _full_reach_map(config), live=True)


def test_live_refuses_on_unmeasured_config():
    config = dataclasses.replace(load_config(), measured=False)  # shipped config may be measured now
    with pytest.raises(Exception, match="(?i)measured"):
        Executor(config, {}, _full_reach_map(config), live=True, to_command=dict)


def test_simulate_refuses_real_arms():
    config = load_config()
    with pytest.raises(TypeError):
        Executor(config, {"follower_1": object()}, _full_reach_map(config), simulate=True)


def test_sim_defaults_to_model_to_urdf():
    config = _low_clear()
    _, default = _sim(config)
    _, explicit = _sim(config, to_command=model_to_urdf)
    program = build_move_program(ROOK_H4, H4H5, _full_reach_map(config), config)
    assert default.plan(program)[0].samples == explicit.plan(program)[0].samples


class _RecordingKinematics(SO101Kinematics):
    """Records every IK target the planner asks for."""

    def __init__(self):
        super().__init__()
        self.targets = []

    def inverse(self, target, **kwargs):
        self.targets.append(tuple(target))
        return super().inverse(target, **kwargs)


def _planned_heights(z_correction):
    config = _low_clear()
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_correction=z_correction))
    rmap = _full_reach_map(config)
    board = chess.Board(None)
    board.set_piece_at(chess.H4, chess.Piece(chess.ROOK, chess.WHITE))
    program = build_move_program(board, chess.Move.from_uci("h4h5"), rmap, config)
    kin = _RecordingKinematics()
    arms = {a: SimArm(a) for a in config.arms.arms}
    for arm in arms.values():
        arm.connect()
    Executor(config, arms, rmap, simulate=True, kinematics=kin).plan(program)
    return [z for _, _, z in kin.targets]


def test_arm_z_correction_raises_every_commanded_height():
    plain = _planned_heights({})
    lifted = _planned_heights({"follower_1": 0.005, "follower_2": 0.005})
    assert len(plain) == len(lifted) > 0
    assert all(b - a == pytest.approx(0.005) for a, b in zip(plain, lifted, strict=True))


# --- the arm must arrive before the gripper acts (2026-10-25 h4 drop) ------


class _StallingElbowArm(SimArm):
    """A SimArm whose elbow stops short of every command (like the real elbow near its limit)."""

    def read_joints(self):
        joints = super().read_joints()
        joints["elbow_flex"] -= 0.2  # ~11 deg short
        return joints


def test_trajectories_mark_the_poses_where_the_gripper_acts():
    config = _low_clear()
    arms, executor = _sim(config)
    program = build_move_program(ROOK_H4, H4H5, _full_reach_map(config), config)
    traj = executor.plan(program)[0]
    # Arrive before the gripper acts, pause after it acts, and be up before moving sideways.
    expected = ["grip", "dwell", "lift", "release", "dwell", "retract"]
    assert [phase for _, phase in traj.checkpoints] == expected
    at = {phase: i for i, phase in traj.checkpoints if phase != "dwell"}
    grip_at, release_at = at["grip"], at["release"]
    # Arrival at the pick pose, jaws still open; arrival at the place pose, jaws still closed.
    assert traj.samples[grip_at]["gripper"] == config.arms.gripper_open
    assert traj.samples[grip_at + 1]["gripper"] < config.arms.gripper_open  # closing starts after
    assert traj.samples[release_at]["gripper"] == config.arms.gripper_closed


def test_gripper_never_moves_when_the_arm_has_not_arrived():
    config = _low_clear()
    arms = {"follower_1": _StallingElbowArm("follower_1"), "follower_2": SimArm("follower_2")}
    for arm in arms.values():
        arm.connect()
    executor = Executor(config, arms, _full_reach_map(config), simulate=True)
    with pytest.raises(ExecutionFault, match="before grip.*NOT moved"):
        executor.execute_move(ROOK_H4, H4H5)
    # Stopped at the pick pose: the jaws were never told to close.
    assert all(c["gripper"] >= config.arms.gripper_open for c in arms["follower_1"].command_log[1:])


def test_a_grasp_that_would_drive_the_tips_into_the_table_is_refused():
    config = _low_clear()
    low = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_grasp=0.002))  # 2 mm
    arms, executor = _sim(low)
    with pytest.raises(Unreachable, match="hit the table"):
        executor.execute_move(ROOK_H4, H4H5)
    assert all(not arm.command_log for arm in arms.values())  # refused before anything moved


def test_tuned_grasps_use_the_square_override_at_both_ends(config_dir):
    from magnus_bot.config import TunedGrasp, save_tuned_grasp

    cfg = config_dir(measured=False)
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=60.0, z_offset_mm=-9.0))
    save_tuned_grasp(cfg, "rook", TunedGrasp(pitch_deg=20.0, z_offset_mm=-6.0, grip_depth_mm=10.0),
                     squares=["h4"])
    config = load_config(cfg)
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=0.06))
    arms, _ = _sim(config)
    executor = Executor(config, arms, _full_reach_map(config), simulate=True, tuned_grasps=True)
    board = chess.Board("4k3/8/8/8/6R1/8/8/4K3 w - - 0 1")  # rook on g4
    near = executor.plan(build_move_program(board, chess.Move.from_uci("g4h4"), executor.rmap, config))
    assert near[0].grasp == "rook @ h4"  # g4 -> h4 takes the near-vertical grasp at BOTH ends
    far = executor.plan(build_move_program(board, chess.Move.from_uci("g4g5"), executor.rmap, config))
    assert far[0].grasp == "rook default"


def test_side_offset_aims_to_the_arms_left_of_the_square():
    # follower_1 faces -x, so its left is -y (toward rank 1): a +5 mm offset aims 5 mm toward rank 1.
    config = _low_clear()
    tips = {}
    for side in (0.0, 0.005):
        arms, executor = _sim(config, side_offset=side)
        executor.execute_move(ROOK_H4, H4H5)
        hold = config.gripper_closed_for("rook")
        grip = next(c for c in arms["follower_1"].command_log if c["gripper"] == hold)
        tip = SO101Kinematics().forward(urdf_to_model(grip))
        tips[side] = config.arms.correction("follower_1").undo(tip)  # where the real tip lands
    placement = arm_placements(config.board.square_size, config.arms.edge_gap)["follower_1"]
    before, after = placement.arm_to_board(tips[0.0][:2]), placement.arm_to_board(tips[0.005][:2])
    reach_dir = (before - placement.base_xy) / np.linalg.norm(before - placement.base_xy)
    # 5 mm square to the reach, to the arm's left (h4 is ~15 deg off the arm's centre line).
    assert split_mm(after - before, reach_dir) == pytest.approx((0.0, 5.0), abs=0.3)
    assert after[1] < before[1]  # i.e. toward rank 1


def test_servo_stop_check_picks_an_easier_grasp_but_keeps_one_that_is_barely_over():
    from magnus_bot.config import GraspChoice
    from magnus_bot.orchestration.executor import STOP_SLACK, _eased

    eased = _eased(GraspChoice(math.radians(45), 0.003, None, "pawn default"), tip_to_pad=0.023)
    assert eased[0].label == "pawn default"  # the tuned grasp is always tried first
    assert [round(math.degrees(c.pitch)) for c in eased[1:6]] == [40, 35, 30, 25, 20]
    assert eased[-1].grip_depth == pytest.approx(0.010)  # finally: stem near the tips
    assert math.degrees(STOP_SLACK) < math.degrees(VERIFY_TOLERANCE)  # never past what arrival allows


def test_after_the_drop_off_the_arm_never_comes_back_down_low():
    # 2026-10-25: the swing back down to a low rest pose swept the edge pieces. After the release the
    # arm goes up and then only higher, to its park.
    config = _low_clear()
    arms, executor = _sim(config)
    traj = executor.plan(build_move_program(ROOK_H4, H4H5, _full_reach_map(config), config))[0]
    retract_at = dict((phase, i) for i, phase in traj.checkpoints)["retract"]
    kin, fix = SO101Kinematics(), config.arms.correction("follower_1")
    heights = [fix.undo(kin.forward(urdf_to_model(s)))[2] for s in traj.samples[retract_at:]]
    assert min(heights) >= config.arms.z_clear - 0.003


def test_a_square_nudge_moves_only_that_squares_end_of_the_move():
    config = _low_clear()
    tips = {}
    for nudge in ({}, {"follower_1": {"h5": (0.010, 0.0)}}):
        cfg = dataclasses.replace(config, arms=dataclasses.replace(config.arms, square_nudge=nudge))
        arms, executor = _sim(cfg)
        traj = executor.plan(build_move_program(ROOK_H4, H4H5, _full_reach_map(cfg), cfg))[0]
        at = dict((phase, i) for i, phase in traj.checkpoints)
        fk = SO101Kinematics().forward
        tips[bool(nudge)] = {ph: fk(urdf_to_model(traj.samples[i]))[:2] for ph, i in at.items()}
    further = np.hypot(*tips[True]["release"]) - np.hypot(*tips[False]["release"])
    assert further == pytest.approx(0.010, abs=5e-4)  # h5 (the place) 10 mm further out
    assert tips[True]["grip"] == pytest.approx(tips[False]["grip"], abs=1e-6)  # h4 (the pick) untouched


class _SaggingShoulderArm(SimArm):
    """The lift never quite gets there: the shoulder reads back 4 deg low (past the tolerance)."""

    def read_joints(self):
        joints = super().read_joints()
        joints["shoulder_lift"] += 0.07
        return joints


def test_a_lift_that_settles_a_little_off_does_not_stop_the_move(monkeypatch):
    from magnus_bot.orchestration import executor as ex

    monkeypatch.setattr(ex, "ARRIVE_TIMEOUT", 0.0)
    config = _low_clear()
    arms = {"follower_1": _SaggingShoulderArm("follower_1"), "follower_2": SimArm("follower_2")}
    for arm in arms.values():
        arm.connect()
    executor = Executor(config, arms, _full_reach_map(config), simulate=True)
    calls = []
    real = executor._arrive

    def arrive(arm, target, phase, strict=True):
        calls.append((phase, strict))
        if strict:
            return None  # the gripper gates are not under test here
        return real(arm, target, phase, strict)

    executor._arrive = arrive
    executor._verify = lambda arm, target: None
    executor.execute_move(ROOK_H4, H4H5)  # no fault although the lift reads back 4 deg off
    assert ("lift", False) in calls and ("retract", False) in calls

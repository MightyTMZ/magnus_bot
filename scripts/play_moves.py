"""Play real chess moves with the tuned grasps; the camera grades every move.

Starts from --fen (default: the standard start position, set up on the real board) and plays
--moves in order: the robots' moves, legal in sequence, with the human's side passing between
them (this tests the arm; it does not play the human's moves). Every move takes the piece's grasp from
configs/grasp.yaml (per-square overrides first, e.g. the near-vertical grab on h3-h6), lifts and
travels at 10 cm, and closes the gripper to the piece's hold (board.yaml gripper_closed_pct).

  sim:   <core-venv>/python scripts/play_moves.py --moves g1f3,g2g3
  live:  <lerobot-venv>/python scripts/play_moves.py --live --moves g1f3,g2g3

The whole sequence is planned and run in sim first (safety rule 4); live only starts if that
passes. Live (follower_1 only for now: follower_2 is not calibrated) needs board.yaml
`measured: true`, RF-DETR/vision_server.py running, the real board matching --fen, and a typed
"yes". After each move the camera grades the squares near it (success / not_picked / lost /
misplaced / disturbed / unclear); anything but success stops the run with the arm holding.
Captures, promotions and handoffs go off the board (bins, pad), which is not calibrated yet:
live refuses them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import chess

from magnus_bot.config import load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import load_servo_map, servo_stops, stops_in_urdf
from magnus_bot.kinematics.reach import compute_reachability
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable
from magnus_bot.orchestration.bench import board_diagram, shut_down
from magnus_bot.orchestration.executor import (
    ExecutionFault,
    Executor,
    JointLimitError,
    build_move_program,
    transfer_squares,
)
from magnus_bot.planning.assigner import Unassignable
from magnus_bot.planning.move_decomposer import SQUARE
from magnus_bot.vision.observation import DEFAULT_VISION_URL, HttpBoardReader, VisionUnavailable
from magnus_bot.vision.verify import (
    Outcome,
    check_transfer,
    expected_occupancy,
    neighborhood,
    restrict,
)

ROOT = Path(__file__).resolve().parents[1]
HOVER = 0.10  # lift / travel height (m), as tune_grasp.py
LIVE_ARMS = {"follower_1"}  # follower_2 is not calibrated yet


def pass_turn(board: chess.Board, robot: chess.Color) -> None:
    """Let the human's side pass (a null move) so the robots can play several moves in a row."""
    if board.turn != robot:
        board.push(chess.Move.null())


def parse_moves(board: chess.Board, text: str, robot: chess.Color = chess.WHITE) -> list[chess.Move]:
    """The robots' UCI moves, checked legal in sequence from ``board`` (not changed). The human's
    side passes between them: this script tests the arm, it does not play the human's moves."""
    replay = board.copy(stack=False)
    moves = []
    for uci in (m.strip() for m in text.split(",") if m.strip()):
        move = chess.Move.from_uci(uci)
        pass_turn(replay, robot)
        if move not in replay.legal_moves:
            raise ValueError(f"{uci} is not legal in {replay.fen()}")
        moves.append(move)
        replay.push(move)
    return moves


def reach_map(config, kin):
    """Where each arm can grasp, up to the steepest tuned tilt (the executor picks the real grasp)."""
    steepest = max([math.radians(t.pitch_deg) for t in config.grasp_tuning.values()] or [0.0])
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    return compute_reachability(placements, kin, config.arms.z_grasp, config.board.square_size,
                                z_clear=config.arms.z_clear, max_pitch=steepest,
                                corrections=config.arms.corrections())


def off_board(program) -> list[str]:
    """Transfers that leave the board (capture bins, pad, stash): not calibrated for live yet."""
    return [s.transfer.reason for s in program
            if s.transfer.src.kind != SQUARE or s.transfer.dst.kind != SQUARE]


def grade(before: chess.Board, move: chess.Move, observed) -> object:
    """The camera's verdict on the squares near ``move`` (far flicker ignored, as in tuning)."""
    after = before.copy(stack=False)
    after.push(move)
    was, want = expected_occupancy(before), expected_occupancy(after)
    zone = neighborhood({s for s in set(was) | set(want) if was.get(s) != want.get(s)})
    return check_transfer(restrict(was, zone), restrict(want, zone), restrict(observed, zone))


def main(argv: list[str] | None = None, prompt=input) -> int:
    parser = argparse.ArgumentParser(description="Play real chess moves with the tuned grasps.")
    parser.add_argument("--moves", required=True, help="comma-separated UCI, e.g. g1f3,g2g3")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="the real board before the first move")
    parser.add_argument("--live", action="store_true", help="drive the real arm (gated)")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--vision-timeout", type=float, default=10.0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    board = chess.Board(args.fen)
    config = load_config(args.config_dir)
    robot = chess.WHITE if config.board.robot_color == "white" else chess.BLACK
    try:
        moves = parse_moves(board, args.moves, robot)
    except ValueError as e:
        parser.error(str(e))
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=HOVER))
    kin = SO101Kinematics()
    rmap = reach_map(config, kin)
    # The servos' physical stops (LeRobot calibration), to choose grasps each arm can really reach.
    smap = load_servo_map(ROOT / "hardware" / "servo_maps" / "follower_1.json")
    stops = servo_stops(config.arms.arms["follower_1"].calibration_id)
    if stops is None:
        print("note: no LeRobot calibration found; grasps are chosen on URDF limits only", file=sys.stderr)

    # --- sim pass of the whole sequence (safety rule 4) ---
    sim_arms = {a: SimArm(a) for a in config.arms.arms}
    for arm in sim_arms.values():
        arm.connect()
    sim = Executor(config, sim_arms, rmap, simulate=True, kinematics=kin, tuned_grasps=True,
                   stops=None if stops is None else {"follower_1": stops_in_urdf(smap, stops)})
    replay = board.copy(stack=False)
    plans = []
    for move in moves:
        pass_turn(replay, robot)
        try:
            program = build_move_program(replay, move, rmap, config)
            trajectories = sim.plan(program)
            sim.run(trajectories)
        except (Unreachable, JointLimitError, ExecutionFault, Unassignable, ValueError) as e:
            piece = board.piece_at(move.from_square) or replay.piece_at(move.from_square)
            name = chess.piece_name(piece.piece_type) if piece else "piece"
            hint = (f" (the {name} has no tuned grasp yet: scripts/tune_grasp.py --piece {name} --save)"
                    if name not in config.grasp_tuning else "")
            print(f"{move.uci()}: cannot be played{hint}: {e}", file=sys.stderr)
            return 1
        arms_used = sorted({s.arm_id for s in program})
        grasps = ", ".join(f"{t.arm_id} {'-'.join(transfer_squares(t.transfer)) or t.transfer.reason}: "
                           f"{t.grasp}" for t in trajectories)
        print(f"SIM OK {move.uci()} ({replay.san(move)}): {grasps}")
        plans.append((move, program, arms_used))
        replay.push(move)
    if not args.live:
        print(f"sim: all {len(moves)} move(s) planned and executed.")
        return 0

    # --- live ---
    for move, program, arms_used in plans:
        if set(arms_used) - LIVE_ARMS:
            print(f"live refused: {move.uci()} needs {', '.join(arms_used)}; only "
                  f"{', '.join(sorted(LIVE_ARMS))} is calibrated", file=sys.stderr)
            return 1
        if off_board(program):
            print(f"live refused: {move.uci()} leaves the board ({', '.join(off_board(program))}); capture "
                  "bins / pad are not calibrated yet", file=sys.stderr)
            return 1
    try:
        config.require_measured()
    except RuntimeError as e:
        print(f"live refused: {e}", file=sys.stderr)
        return 1
    reader = HttpBoardReader(args.vision_url)
    try:
        obs = reader.read(fresh=True, timeout=args.vision_timeout)
    except VisionUnavailable as e:
        print(f"vision: {e}", file=sys.stderr)
        return 1
    print(board_diagram(obs.occupancy))
    expected = expected_occupancy(board)
    first = moves[0]
    near = neighborhood({chess.square_name(first.from_square), chess.square_name(first.to_square)})
    off = sorted(s for s in near if expected.get(s) != obs.occupancy.get(s))
    if off:
        print(f"the real board does not match --fen near {first.uci()}: check {', '.join(off)}",
              file=sys.stderr)
        return 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.log or ROOT / "hardware" / "tuning_logs" / f"play_{stamp}.jsonl"
    print(f"\nLIVE: follower_1 plays {', '.join(m.uci() for m in moves)}; lifts to {HOVER * 100:.0f} cm "
          f"between pick and place. Each move is graded; anything but success stops with the arm "
          f"holding.\nlog: {log_path}\nCtrl+C holds the arm. Keep a hand near the power.")
    if prompt("Type 'yes' to start: ").strip().lower() != "yes":
        print("aborted")
        return 1

    from magnus_bot.hardware.lerobot_arm import LeRobotArm

    spec = config.arms.arms["follower_1"]
    arms = {"follower_1": LeRobotArm(arm_id="follower_1", port=spec.port, calibration_id=spec.calibration_id)}
    code = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        try:
            arms["follower_1"].connect()
            live = Executor(config, arms, rmap, live=True, to_command=smap.model_to_servo,
                            limits=smap.servo_limits(kin.model.limits), kinematics=kin, tuned_grasps=True,
                            stops=None if stops is None else {"follower_1": stops})
            for move in moves:
                pass_turn(board, robot)
                san = board.san(move)
                started = time.time()
                trajectories = live.plan(build_move_program(board, move, rmap, config))
                live.run(trajectories)
                check = grade(board, move, reader.read(fresh=True, timeout=args.vision_timeout).occupancy)
                fh.write(json.dumps({"event": "move", "time": started, "uci": move.uci(), "san": san,
                                     "grasp": [t.grasp for t in trajectories],
                                     "outcome": check.outcome.value, "detail": check.describe(),
                                     "duration_s": round(time.time() - started, 1)}) + "\n")
                fh.flush()
                print(f"{move.uci()} ({san}): {check.describe()}")
                if check.outcome is not Outcome.SUCCESS:
                    print("stopped: fix the board, then rerun from the position on the board (--fen)",
                          file=sys.stderr)
                    code = 2
                    break
                board.push(move)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            code = 130
        except (ExecutionFault, VisionUnavailable, Unreachable, JointLimitError) as e:
            print(f"stopped, arm holding: {e}", file=sys.stderr)
            code = 3
        finally:
            shut_down(arms, live=True, prompt=prompt)
    print(f"position now: {board.fen()}\nlog: {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

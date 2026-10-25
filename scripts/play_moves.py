"""Play real chess moves with the tuned grasps; the camera grades every move.

Starts from --fen (default: the standard start position, set up on the real board) and plays
--moves in order. The list may hold the whole game: the robots play their side's moves (White by
default), and for the human's moves the script waits for you to make them on the board, then
checks them with the camera. A list of only robot moves also works: the human's side passes.
Every move takes the piece's grasp from configs/grasp.yaml (per-square overrides first), eased
steeper where it cannot reach, lifts and travels at 10 cm, and closes the gripper to the piece's
hold (board.yaml gripper_closed_pct).

  sim:   <core-venv>/python scripts/play_moves.py --moves e2e4,e7e5,f1c4,b8c6,d1h5,g8f6,h5f7
  live:  <lerobot-venv>/python scripts/play_moves.py --live --arms follower_1,follower_2 --moves ...

Handoffs (neither arm reaches both squares) go through the pad, a board square both arms reach
(configs/handoff.yaml: e3): arm A sets the piece down there, arm B picks it up. Captures first
take the victim to the capturing arm's bin, a spot on the table just off the board.

The whole sequence is planned and run in sim first (safety rule 4); live only starts if that
passes. Live needs board.yaml `measured: true`, RF-DETR/vision_server.py running, the real board
matching --fen, every arm a move needs in --arms, and a typed "yes". After each move the camera
grades the squares near it (success / not_picked / lost / misplaced / disturbed / unclear);
anything but success stops the run with the arms holding. Promotions are refused live (no stash).
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
import numpy as np

from magnus_bot.config import load_config
from magnus_bot.geometry import board as board_geom
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
from magnus_bot.planning.move_decomposer import PAD, STASH
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
DEFAULT_LIVE_ARMS = "follower_1"
HUMAN_TRIES = 3  # how often to re-check the board after asking for the human's move


@dataclasses.dataclass(frozen=True)
class Turn:
    move: chess.Move
    robot: bool  # True: the arms play it; False: the human plays it by hand


def parse_game(board: chess.Board, text: str, robot: chess.Color = chess.WHITE) -> list[Turn]:
    """The moves in order, checked legal in sequence from ``board`` (not changed). A move legal for
    the side to move is that side's; if the human is to move and the move is not theirs, the human
    passes (a null move), so a list of robot moves only tests the arms."""
    replay = board.copy(stack=False)
    turns = []
    for uci in (m.strip() for m in text.split(",") if m.strip()):
        move = chess.Move.from_uci(uci)
        if replay.turn != robot and move not in replay.legal_moves:
            replay.push(chess.Move.null())
        if move not in replay.legal_moves:
            raise ValueError(f"{uci} is not legal in {replay.fen()}")
        turns.append(Turn(move, replay.turn == robot))
        replay.push(move)
    return turns


def reach_map(config, kin):
    """Where each arm can grasp, up to the steepest tuned tilt (the executor picks the real grasp),
    and which arms reach the handoff pad."""
    steepest = max([math.radians(t.pitch_deg) for t in config.grasp_tuning.values()] or [0.0])
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    pad = config.handoff.pad
    return compute_reachability(placements, kin, config.arms.z_grasp, config.board.square_size,
                                z_clear=config.arms.z_clear, max_pitch=steepest,
                                pad_xyz=np.array([pad.x, pad.y, config.arms.z_grasp]),
                                corrections=config.arms.corrections())


def pad_square(config) -> str | None:
    """The board square the handoff pad sits on, if it is on the board."""
    s = config.board.square_size
    f, r = int(config.handoff.pad.x // s), int(config.handoff.pad.y // s)
    if 0 <= f < board_geom.BOARD_SIZE and 0 <= r < board_geom.BOARD_SIZE:
        return chess.square_name(chess.square(f, r))
    return None


def grade(before: chess.Board, move: chess.Move, observed, also: set[str] = frozenset()) -> object:
    """The camera's verdict on the squares near ``move`` (and ``also``, e.g. the pad, which must be
    empty again); far flicker is ignored, as in tuning."""
    after = before.copy(stack=False)
    after.push(move)
    was, want = expected_occupancy(before), expected_occupancy(after)
    zone = neighborhood({s for s in set(was) | set(want) if was.get(s) != want.get(s)}) | set(also)
    return check_transfer(restrict(was, zone), restrict(want, zone), restrict(observed, zone))


def main(argv: list[str] | None = None, prompt=input) -> int:
    parser = argparse.ArgumentParser(description="Play real chess moves with the tuned grasps.")
    parser.add_argument("--moves", required=True,
                        help="comma-separated UCI, the whole game or the robots' moves only")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="the real board before the first move")
    parser.add_argument("--live", action="store_true", help="drive the real arms (gated)")
    parser.add_argument("--arms", default=DEFAULT_LIVE_ARMS,
                        help="arms allowed to move live, comma-separated (e.g. follower_1,follower_2)")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--vision-timeout", type=float, default=10.0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    board = chess.Board(args.fen)
    config = load_config(args.config_dir)
    robot = chess.WHITE if config.board.robot_color == "white" else chess.BLACK
    try:
        turns = parse_game(board, args.moves, robot)
    except ValueError as e:
        parser.error(str(e))
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=HOVER))
    kin = SO101Kinematics()
    rmap = reach_map(config, kin)
    pad = pad_square(config)
    live_arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    if live_arms - set(config.arms.arms):
        parser.error(f"--arms: unknown {sorted(live_arms - set(config.arms.arms))}")
    # Per arm: servo map, and the servos' physical stops (LeRobot calibration) to choose grasps
    # each arm can really reach.
    smaps = {a: load_servo_map(ROOT / "hardware" / "servo_maps" / f"{a}.json") for a in config.arms.arms}
    stops = {a: st for a in config.arms.arms
             if (st := servo_stops(config.arms.arms[a].calibration_id)) is not None}
    if not stops:
        print("note: no LeRobot calibration found; grasps are chosen on URDF limits only", file=sys.stderr)

    # --- sim pass of the whole sequence (safety rule 4: two-arm motion only after a sim pass) ---
    sim_arms = {a: SimArm(a) for a in config.arms.arms}
    for arm in sim_arms.values():
        arm.connect()
    sim = Executor(config, sim_arms, rmap, simulate=True, kinematics=kin, tuned_grasps=True,
                   stops={a: stops_in_urdf(smaps[a], st) for a, st in stops.items()})
    replay = board.copy(stack=False)
    plans = {}  # turn index -> (program, arms used)
    for i, turn in enumerate(turns):
        if turn.move not in replay.legal_moves:
            replay.push(chess.Move.null())
        san = replay.san(turn.move)
        if not turn.robot:
            print(f"human  {turn.move.uci()} ({san})")
            replay.push(turn.move)
            continue
        try:
            program = build_move_program(replay, turn.move, rmap, config)
            trajectories = sim.plan(program)
            sim.run(trajectories)
        except (Unreachable, JointLimitError, ExecutionFault, Unassignable, ValueError) as e:
            piece = replay.piece_at(turn.move.from_square)
            name = chess.piece_name(piece.piece_type) if piece else "piece"
            hint = (f" (the {name} has no tuned grasp yet: scripts/tune_grasp.py --piece {name} --save)"
                    if name not in config.grasp_tuning else "")
            print(f"{turn.move.uci()}: cannot be played{hint}: {e}", file=sys.stderr)
            return 1
        arms_used = sorted({s.arm_id for s in program})
        legs = ", ".join(f"{t.arm_id} {t.transfer.reason} "
                         f"{'-'.join(transfer_squares(t.transfer)) or ''}: {t.grasp}" for t in trajectories)
        print(f"SIM OK {turn.move.uci()} ({san}): {legs}")
        plans[i] = (program, arms_used)
        replay.push(turn.move)
    if not args.live:
        print(f"sim: all {len(plans)} robot move(s) planned and executed.")
        return 0

    # --- live ---
    for i, (program, arms_used) in plans.items():
        uci = turns[i].move.uci()
        if set(arms_used) - live_arms:
            print(f"live refused: {uci} needs {', '.join(arms_used)}; --arms allows only "
                  f"{', '.join(sorted(live_arms))}", file=sys.stderr)
            return 1
        if any(STASH in (s.transfer.src.kind, s.transfer.dst.kind) for s in program):
            print(f"live refused: {uci} is a promotion; the queen stash is not set up", file=sys.stderr)
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
    first = turns[0].move
    near = neighborhood({chess.square_name(first.from_square), chess.square_name(first.to_square)})
    off = sorted(s for s in near if expected.get(s) != obs.occupancy.get(s))
    if off:
        print(f"the real board does not match --fen near {first.uci()}: check {', '.join(off)}",
              file=sys.stderr)
        return 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.log or ROOT / "hardware" / "tuning_logs" / f"play_{stamp}.jsonl"
    used = sorted({a for _, arms_used in plans.values() for a in arms_used})
    print(f"\nLIVE: {' + '.join(used)} play {', '.join(t.move.uci() for t in turns if t.robot)}; you "
          f"play {', '.join(t.move.uci() for t in turns if not t.robot) or 'nothing'}. Each move is "
          f"graded; anything but success stops with the arms holding.\nlog: {log_path}\n"
          "Ctrl+C holds the arms. Keep a hand near the power.")
    if prompt("Type 'yes' to start: ").strip().lower() != "yes":
        print("aborted")
        return 1

    from magnus_bot.hardware.lerobot_arm import LeRobotArm

    arms = {a: LeRobotArm(arm_id=a, port=config.arms.arms[a].port,
                          calibration_id=config.arms.arms[a].calibration_id) for a in used}
    code = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        def log(record):
            fh.write(json.dumps(record) + "\n")
            fh.flush()

        try:
            executors = {}  # one per arm, each with its own servo map; arms move one at a time
            for a, arm in arms.items():
                arm.connect()
                executors[a] = Executor(config, {a: arm}, rmap, live=True, to_command=smaps[a].model_to_servo,
                                        limits=smaps[a].servo_limits(kin.model.limits), kinematics=kin,
                                        tuned_grasps=True, stops={a: stops[a]} if a in stops else None)
            for turn in turns:
                if turn.move not in board.legal_moves:
                    board.push(chess.Move.null())  # the human passes
                san = board.san(turn.move)
                started = time.time()
                if not turn.robot:
                    check = None
                    for _ in range(HUMAN_TRIES):
                        answer = prompt(f"YOUR MOVE: play {turn.move.uci()} ({san}) on the board, then press "
                                        "Enter (q to stop): ").strip().lower()
                        if answer == "q":
                            break
                        check = grade(board, turn.move, reader.read(fresh=True, timeout=args.vision_timeout)
                                      .occupancy)
                        if check.outcome is Outcome.SUCCESS:
                            break
                        print(f"  the camera does not see it yet: {check.describe()}")
                    log({"event": "human", "time": started, "uci": turn.move.uci(), "san": san,
                         "outcome": None if check is None else check.outcome.value})
                    if check is None or check.outcome is not Outcome.SUCCESS:
                        print("stopped: the human move was not confirmed", file=sys.stderr)
                        code = 2
                        break
                    print(f"{turn.move.uci()} ({san}): seen on the board")
                    board.push(turn.move)
                    continue
                program = build_move_program(board, turn.move, rmap, config)
                grasps = []
                for leg in program:  # a handoff is two legs, each by its own arm, one after the other
                    trajectories = executors[leg.arm_id].plan([leg])
                    executors[leg.arm_id].run(trajectories)
                    grasps += [f"{t.arm_id} {t.transfer.reason}: {t.grasp}" for t in trajectories]
                via_pad = any(PAD in (s.transfer.src.kind, s.transfer.dst.kind) for s in program)
                seen = reader.read(fresh=True, timeout=args.vision_timeout).occupancy
                check = grade(board, turn.move, seen, also={pad} if via_pad and pad else set())
                log({"event": "move", "time": started, "uci": turn.move.uci(), "san": san, "grasp": grasps,
                     "outcome": check.outcome.value, "detail": check.describe(),
                     "duration_s": round(time.time() - started, 1)})
                print(f"{turn.move.uci()} ({san}): {check.describe()}")
                if check.outcome is not Outcome.SUCCESS:
                    print("stopped: fix the board, then rerun from the position on the board (--fen)",
                          file=sys.stderr)
                    code = 2
                    break
                board.push(turn.move)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            code = 130
        except (ExecutionFault, VisionUnavailable, Unreachable, JointLimitError) as e:
            print(f"stopped, arms holding: {e}", file=sys.stderr)
            code = 3
        finally:
            shut_down(arms, live=True, prompt=prompt)
    print(f"position now: {board.fen()}\nlog: {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Play a real game against Stockfish: the engine plays the robots' side with the arms, you play the
other side by hand on the board and the camera sees your move.

  sim:   <core-venv>/python scripts/play_stockfish.py                 (you type your moves)
  live:  <lerobot-venv>/python scripts/play_stockfish.py --live --elo 1500

Each robot move: Stockfish picks one, it is planned and run in sim (safety rule 4), then played
live and graded by the camera like play_moves.py. If the arms cannot play the engine's pick (no
reach, an untuned piece such as the king, a promotion live, a handoff needing an arm not in
--arms), the next best of its top --candidates moves that they can play is used instead.

Your move: the camera watches until the board shows one legal move, the same on two looks in a row
(a hand still over the board does not count; steady camera misreads cancel out). Promotions are read
as a queen. A board no legal move explains is a warning; --grace s uncorrected stops the game.

Live needs board.yaml `measured: true`, RF-DETR/vision_server.py running, the real board matching
--fen, and a typed "yes". After a robot move a wrong board (a piece missing, or one where none
should be) is only a warning; it stops the game, arms holding, once uncorrected for --grace s (60).
Stockfish: STOCKFISH_PATH, `stockfish` on PATH, or `winget install Stockfish.Stockfish`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import chess

from magnus_bot.config import load_config
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import load_servo_map, servo_stops, stops_in_urdf
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
from magnus_bot.sources.stockfish_source import StockfishSource
from magnus_bot.vision.observation import DEFAULT_VISION_URL, HttpBoardReader, VisionUnavailable
from magnus_bot.vision.verify import expected_occupancy
from play_moves import GRACE, HOVER, grade, pad_square, reach_map, wait_for_any_move, wait_for_robot_move

ROOT = Path(__file__).resolve().parents[1]
HUMAN_TIMEOUT = 0.0  # s to wait for the human's move; 0: as long as they think
CANDIDATES = 5  # engine moves to try, best first, when the arms cannot play its pick
UNPLAYABLE = (Unreachable, JointLimitError, ExecutionFault, Unassignable, ValueError)


class Unplayable(Exception):
    """The arms cannot play this move (here, or live with these arms)."""


def choose_move(board: chess.Board, engine, plan, n: int = CANDIDATES):
    """Stockfish's move and its plan, or the best of its top ``n`` the arms can play; None if none.
    ``plan(move)`` returns the move's program or raises :class:`Unplayable`."""
    def attempt(move):
        try:
            return move, plan(move)
        except Unplayable as e:
            print(f"  skip {move.uci()} ({board.san(move)}): {e}", file=sys.stderr)
            return None

    first = engine.next_move(board)
    if chosen := attempt(first):
        return chosen
    for move in engine.candidates(board, n):
        if move != first and (chosen := attempt(move)):
            return chosen
    return None


def typed_move(board: chess.Board, prompt) -> chess.Move | None:
    """Sim: the human types their move (UCI or SAN); blank stops."""
    while True:
        side = "White" if board.turn else "Black"
        text = prompt(f"your move ({side}, UCI or SAN, blank to stop): ").strip()
        if not text:
            return None
        try:
            return board.parse_uci(text)
        except ValueError:
            pass
        try:
            return board.parse_san(text)
        except ValueError:
            print(f"  not legal here: {text}", file=sys.stderr)


def main(argv: list[str] | None = None, prompt=input) -> int:
    parser = argparse.ArgumentParser(description="Play a real game against Stockfish with the arms.")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="the real board before the first move")
    parser.add_argument("--live", action="store_true", help="drive the real arms (gated)")
    parser.add_argument("--arms", default=None,
                        help="arms allowed to move live, comma-separated (default: both)")
    parser.add_argument("--elo", type=int, default=None, help="cap Stockfish's strength (1320-3190)")
    parser.add_argument("--think", type=float, default=0.5, help="engine seconds per move")
    parser.add_argument("--candidates", type=int, default=CANDIDATES,
                        help="engine moves to try when the arms cannot play its first pick")
    parser.add_argument("--engine", default=None, help="Stockfish binary (default: found automatically)")
    parser.add_argument("--human-timeout", type=float, default=HUMAN_TIMEOUT,
                        help="seconds to wait for each of your moves (0: no limit)")
    parser.add_argument("--grace", type=float, default=GRACE,
                        help="seconds the board may stay wrong after a robot move before the game stops")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--vision-timeout", type=float, default=10.0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    board = chess.Board(args.fen)
    config = load_config(args.config_dir)
    config = dataclasses.replace(config, arms=dataclasses.replace(config.arms, z_clear=HOVER))
    robot = chess.WHITE if config.board.robot_color == "white" else chess.BLACK
    kin = SO101Kinematics()
    rmap = reach_map(config, kin)
    pad = pad_square(config)
    live_arms = {a.strip() for a in (args.arms or ",".join(config.arms.arms)).split(",") if a.strip()}
    if live_arms - set(config.arms.arms):
        parser.error(f"--arms: unknown {sorted(live_arms - set(config.arms.arms))}")
    smaps = {a: load_servo_map(ROOT / "hardware" / "servo_maps" / f"{a}.json") for a in config.arms.arms}
    stops = {a: st for a in config.arms.arms
             if (st := servo_stops(config.arms.arms[a].calibration_id)) is not None}

    sim_arms = {a: SimArm(a) for a in config.arms.arms}
    for arm in sim_arms.values():
        arm.connect()
    sim = Executor(config, sim_arms, rmap, simulate=True, kinematics=kin, tuned_grasps=True,
                   stops={a: stops_in_urdf(smaps[a], st) for a, st in stops.items()})

    def plan(move: chess.Move):
        """Sim pass of one robot move (safety rule 4), plus the live gates."""
        try:
            program = build_move_program(board, move, rmap, config)
            trajectories = sim.plan(program)
            sim.run(trajectories)
        except UNPLAYABLE as e:
            piece = board.piece_at(move.from_square)
            name = chess.piece_name(piece.piece_type) if piece else "piece"
            raise Unplayable(f"no tuned {name} grasp" if name not in config.grasp_tuning else str(e)) from e
        if args.live:
            if needs := sorted({s.arm_id for s in program} - live_arms):
                raise Unplayable(f"needs {', '.join(needs)}, not in --arms")
            if any(STASH in (s.transfer.src.kind, s.transfer.dst.kind) for s in program):
                raise Unplayable("a promotion (the queen stash is not set up)")
        legs = ", ".join(f"{t.arm_id} {t.transfer.reason} "
                         f"{'-'.join(transfer_squares(t.transfer)) or ''}: {t.grasp}" for t in trajectories)
        print(f"SIM OK {move.uci()} ({board.san(move)}): {legs}")
        return program

    try:
        engine = StockfishSource(args.engine, think_time=args.think, elo=args.elo)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    with engine:
        if not args.live:
            return sim_game(board, robot, engine, plan, args, prompt)
        return live_game(board, robot, engine, plan, config, kin, rmap, smaps, stops, pad, live_arms,
                         args, prompt)


def sim_game(board, robot, engine, plan, args, prompt) -> int:
    print(f"SIM: Stockfish plays {'White' if robot else 'Black'}; type your moves.")
    while not board.is_game_over():
        if board.turn == robot:
            chosen = choose_move(board, engine, plan, args.candidates)
            if chosen is None:
                print("stopped: the arms cannot play any of the engine's moves", file=sys.stderr)
                return 2
            board.push(chosen[0])
        else:
            move = typed_move(board, prompt)
            if move is None:
                break
            board.push(move)
    print(f"position now: {board.fen()}")
    if board.is_game_over():
        print(f"game over: {board.result()}")
    return 0


def live_game(board, robot, engine, plan, config, kin, rmap, smaps, stops, pad, live_arms,
              args, prompt) -> int:
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
    off = sorted(s for s in set(expected) | set(obs.occupancy) if expected.get(s) != obs.occupancy.get(s))
    if off:
        print(f"WARNING: the camera's board differs from --fen at {', '.join(off)}; a move near "
              "those squares will fail its grade", file=sys.stderr)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.log or ROOT / "hardware" / "tuning_logs" / f"stockfish_{stamp}.jsonl"
    strength = f"Elo {args.elo}" if args.elo else "full strength"
    print(f"\nLIVE vs Stockfish ({strength}, {args.think:g} s/move) with {', '.join(sorted(live_arms))}; "
          f"you play {'Black' if robot else 'White'}. After each robot move a wrong board is a warning; "
          f"{args.grace:g} s uncorrected stops with the arms holding.\nlog: {log_path}\n"
          "Ctrl+C holds the arms. Keep a hand near the power.")
    if prompt("Type 'yes' to start: ").strip().lower() != "yes":
        print("aborted")
        return 1

    from magnus_bot.hardware.lerobot_arm import LeRobotArm

    arms = {a: LeRobotArm(arm_id=a, port=config.arms.arms[a].port,
                          calibration_id=config.arms.arms[a].calibration_id) for a in sorted(live_arms)}
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
            baseline = None  # what the camera saw after the last robot move (misreads included)
            while not board.is_game_over():
                started = time.time()
                if board.turn != robot:
                    move = wait_for_any_move(reader, board, args.human_timeout, args.vision_timeout,
                                             baseline=baseline, grace=args.grace)
                    baseline = None
                    if move is None:
                        print("stopped: no legal move on the board", file=sys.stderr)
                        log({"event": "human_timeout", "time": started,
                             "waited_s": round(time.time() - started, 1)})
                        code = 2
                        break
                    san = board.san(move)
                    print(f"you: {move.uci()} ({san})")
                    log({"event": "human", "time": started, "uci": move.uci(), "san": san,
                         "waited_s": round(time.time() - started, 1)})
                    board.push(move)
                    continue
                chosen = choose_move(board, engine, plan, args.candidates)
                if chosen is None:
                    print("stopped: the arms cannot play any of the engine's moves", file=sys.stderr)
                    log({"event": "no_playable_move", "time": started, "fen": board.fen()})
                    code = 2
                    break
                move, program = chosen
                san = board.san(move)
                grasps = []
                for leg in program:  # a handoff is two legs, each by its own arm, one after the other
                    trajectories = executors[leg.arm_id].plan([leg])
                    executors[leg.arm_id].run(trajectories)
                    grasps += [f"{t.arm_id} {t.transfer.reason}: {t.grasp}" for t in trajectories]
                via_pad = any(PAD in (s.transfer.src.kind, s.transfer.dst.kind) for s in program)
                also = {pad} if via_pad and pad else set()
                ok, seen = wait_for_robot_move(reader, board, move, also, args.vision_timeout, args.grace)
                baseline = seen
                check = grade(board, move, seen, also=also) if seen is not None else None
                log({"event": "move", "time": started, "uci": move.uci(), "san": san, "grasp": grasps,
                     "ok": ok, "outcome": check and check.outcome.value, "detail": check and check.describe(),
                     "duration_s": round(time.time() - started, 1)})
                print(f"stockfish: {move.uci()} ({san}): {check.describe() if check else 'no stable board'}")
                if not ok:
                    print(f"stopped: the board did not show {move.uci()} within {args.grace:g} s; fix it, "
                          "then rerun from the position on the board (--fen)", file=sys.stderr)
                    code = 2
                    break
                board.push(move)
            if board.is_game_over():
                print(f"game over: {board.result()}")
                log({"event": "game_over", "result": board.result(), "fen": board.fen()})
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

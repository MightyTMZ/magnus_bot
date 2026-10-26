"""Live game loop: chess.com opponent <-> robot arms <-> human on the physical board.

The chess.com Next.js app (chess_com/magnus_bot) runs a Steel browser session.  The opponent on
chess.com plays White (its moves are detected by the monitor endpoint).  The robot arms replicate
those moves on the physical board so the human can see them.  The human responds on the physical
board; the camera detects the move, and this script sends it back to chess.com via the play-move
endpoint.

  sim:   python scripts/play_chess_com.py --session-id <steel-session-id>
  live:  python scripts/play_chess_com.py --session-id <steel-session-id> --live --arms follower_1,follower_2

Flow (each full turn):
  1. Chess.com (White) makes a move -> monitor detects it -> robot arms execute it on the board.
  2. Human (Black) plays their reply on the board -> camera detects it -> sent to chess.com.
  3. Repeat until game over.

Uses the same sim-first safety (rule 4) as play_moves.py: every robot move is planned in sim
before live execution.
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
from magnus_bot.sources.chess_com_source import ChessComSource
from magnus_bot.vision.observation import DEFAULT_VISION_URL, HttpBoardReader, VisionUnavailable
from magnus_bot.vision.verify import (
    check_transfer,
    expected_occupancy,
    neighborhood,
    restrict,
)
from play_moves import GRACE, wait_for_any_move, wait_for_robot_move

ROOT = Path(__file__).resolve().parents[1]
HOVER = 0.10  # lift / travel height (m)
DEFAULT_LIVE_ARMS = "follower_1"
HUMAN_TIMEOUT = 0.0  # s to wait for the human's move; 0: as long as they think
CHESSCOM_POLL = 1.0  # s between chess.com monitor polls
CHESSCOM_TIMEOUT = 600.0  # s to wait for chess.com opponent


def reach_map(config, kin):
    steepest = max([math.radians(t.pitch_deg) for t in config.grasp_tuning.values()] or [0.0])
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    pad = config.handoff.pad
    return compute_reachability(placements, kin, config.arms.z_grasp, config.board.square_size,
                                z_clear=config.arms.z_clear, max_pitch=steepest,
                                pad_xyz=np.array([pad.x, pad.y, config.arms.z_grasp]),
                                corrections=config.arms.corrections())


def pad_square(config) -> str | None:
    s = config.board.square_size
    f, r = int(config.handoff.pad.x // s), int(config.handoff.pad.y // s)
    if 0 <= f < board_geom.BOARD_SIZE and 0 <= r < board_geom.BOARD_SIZE:
        return chess.square_name(chess.square(f, r))
    return None


def grade(before: chess.Board, move: chess.Move, observed, also: set[str] = frozenset()) -> object:
    after = before.copy(stack=False)
    after.push(move)
    was, want = expected_occupancy(before), expected_occupancy(after)
    zone = neighborhood({s for s in set(was) | set(want) if was.get(s) != want.get(s)}) | set(also)
    return check_transfer(restrict(was, zone), restrict(want, zone), restrict(observed, zone))


def sim_pass_move(board, move, sim_executor, rmap, config):
    """Sim-plan a single move.  Returns ``(program, trajectories)`` or raises."""
    program = build_move_program(board, move, rmap, config)
    trajectories = sim_executor.plan(program)
    sim_executor.run(trajectories)
    return program, trajectories


def main(argv: list[str] | None = None, prompt=input) -> int:
    parser = argparse.ArgumentParser(description="Live chess.com <-> robot arm game loop.")
    parser.add_argument("--session-id", required=True, help="Steel browser session ID")
    parser.add_argument("--base-url", default="http://localhost:3000", help="Next.js app URL")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="starting position on the board")
    parser.add_argument("--live", action="store_true", help="drive real arms (gated)")
    parser.add_argument("--arms", default=None,
                        help="arms allowed to move live, comma-separated (default: both)")
    parser.add_argument("--human-timeout", type=float, default=HUMAN_TIMEOUT,
                        help="seconds to wait for each of your moves (0: no limit)")
    parser.add_argument("--grace", type=float, default=GRACE,
                        help="seconds an illegal / unexplained board may stay uncorrected")
    parser.add_argument("--chesscom-timeout", type=float, default=CHESSCOM_TIMEOUT)
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
    arms_text = args.arms or ",".join(config.arms.arms)
    live_arms = {a.strip() for a in arms_text.split(",") if a.strip()}

    smaps = {a: load_servo_map(ROOT / "hardware" / "servo_maps" / f"{a}.json") for a in config.arms.arms}
    stops = {a: st for a in config.arms.arms
             if (st := servo_stops(config.arms.arms[a].calibration_id)) is not None}

    # Chess.com source: polls the monitor for opponent (White) moves
    chesscom = ChessComSource(
        session_id=args.session_id,
        base_url=args.base_url,
        poll_interval=CHESSCOM_POLL,
        timeout=args.chesscom_timeout,
    )

    # Sim executor for safety rule 4
    sim_arms = {a: SimArm(a) for a in config.arms.arms}
    for arm in sim_arms.values():
        arm.connect()
    sim = Executor(config, sim_arms, rmap, simulate=True, kinematics=kin, tuned_grasps=True,
                   stops={a: stops_in_urdf(smaps[a], st) for a, st in stops.items()})

    # Seed the monitor's stored screenshot so the first real diff is meaningful.
    # Without this, the first poll returns "first_screenshot" with no move.
    print("Priming chess.com monitor (seeding initial screenshot)...")
    chesscom._poll_once()
    time.sleep(1.0)

    if not args.live:
        print("SIM MODE: will plan robot moves but not drive hardware. Human moves are skipped.")
        print(f"Polling chess.com at {args.base_url} (session {args.session_id})\n")
        move_num = 0
        try:
            while not board.is_game_over():
                if board.turn == robot:
                    # Robot's turn: get move from chess.com
                    print("Waiting for chess.com move (White)...")
                    move = chesscom.next_move(board)
                    san = board.san(move)
                    move_num += 1
                    try:
                        program, trajectories = sim_pass_move(board, move, sim, rmap, config)
                        legs = ", ".join(
                            f"{t.arm_id} {t.transfer.reason} "
                            f"{'-'.join(transfer_squares(t.transfer)) or ''}: {t.grasp}"
                            for t in trajectories
                        )
                        print(f"SIM OK {move.uci()} ({san}): {legs}")
                    except (Unreachable, JointLimitError, Unassignable, ValueError) as e:
                        print(f"SIM FAIL {move.uci()} ({san}): {e}", file=sys.stderr)
                    board.push(move)
                else:
                    # Human's turn: skip in sim, just wait for chess.com to reflect it
                    print("Human's turn (Black): skipping in sim, waiting for next White move...")
                    # The monitor only fires on White moves, so we need to infer Black's move
                    # from the next monitor cycle.  For sim we just wait.
                    print("  (in sim mode, enter the human's UCI move to continue)")
                    uci = input("  Black move UCI: ").strip()
                    if not uci:
                        continue
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        print(f"  illegal: {uci}", file=sys.stderr)
                        continue
                    board.push(move)
        except (KeyboardInterrupt, TimeoutError) as e:
            print(f"\n{e}" if isinstance(e, TimeoutError) else "\ninterrupted")
        print(f"\nFinal position: {board.fen()}")
        if board.is_game_over():
            print(f"Game over: {board.result()}")
        return 0

    # --- live mode ---
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

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.log or ROOT / "hardware" / "tuning_logs" / f"chess_com_{stamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLIVE chess.com game loop.")
    print(f"  Chess.com session: {args.session_id}")
    print(f"  Arms: {', '.join(sorted(live_arms))}")
    print(f"  Robot plays: {'White' if robot == chess.WHITE else 'Black'}")
    print(f"  Log: {log_path}")
    print("Ctrl+C holds the arms. Keep a hand near the power.")
    if prompt("Type 'yes' to start: ").strip().lower() != "yes":
        print("aborted")
        return 1

    from magnus_bot.hardware.lerobot_arm import LeRobotArm

    arms = {a: LeRobotArm(arm_id=a, port=config.arms.arms[a].port,
                           calibration_id=config.arms.arms[a].calibration_id) for a in sorted(live_arms)}
    code = 0
    with log_path.open("a", encoding="utf-8") as fh:
        def log(record):
            fh.write(json.dumps(record) + "\n")
            fh.flush()

        try:
            executors = {}
            for a, arm in arms.items():
                arm.connect()
                executors[a] = Executor(config, {a: arm}, rmap, live=True, to_command=smaps[a].model_to_servo,
                                        limits=smaps[a].servo_limits(kin.model.limits), kinematics=kin,
                                        tuned_grasps=True, stops={a: stops[a]} if a in stops else None)

            baseline = None  # what the camera saw after the last robot move (misreads included)
            while not board.is_game_over():
                if board.turn == robot:
                    # ---- Robot's turn: get move from chess.com, execute on physical board ----
                    print("\nWaiting for chess.com move (White)...")
                    started = time.time()
                    try:
                        move = chesscom.next_move(board)
                    except TimeoutError as e:
                        print(f"chess.com timeout: {e}", file=sys.stderr)
                        code = 2
                        break
                    san = board.san(move)
                    print(f"Chess.com played: {move.uci()} ({san})")

                    # Sim pass first (safety rule 4)
                    try:
                        program, _ = sim_pass_move(board, move, sim, rmap, config)
                    except (Unreachable, JointLimitError, Unassignable, ValueError) as e:
                        print(f"cannot execute {move.uci()}: {e}", file=sys.stderr)
                        log({"event": "sim_fail", "time": started, "uci": move.uci(), "error": str(e)})
                        code = 3
                        break

                    arms_used = sorted({s.arm_id for s in program})
                    if set(arms_used) - live_arms:
                        print(f"live refused: {move.uci()} needs {', '.join(arms_used)}", file=sys.stderr)
                        code = 1
                        break
                    if any(STASH in (s.transfer.src.kind, s.transfer.dst.kind) for s in program):
                        print(f"live refused: {move.uci()} is a promotion", file=sys.stderr)
                        code = 1
                        break

                    # Execute on real arms
                    grasps = []
                    for leg in program:
                        trajectories = executors[leg.arm_id].plan([leg])
                        executors[leg.arm_id].run(trajectories)
                        grasps += [f"{t.arm_id} {t.transfer.reason}: {t.grasp}" for t in trajectories]

                    # Grade with camera
                    via_pad = any(PAD in (s.transfer.src.kind, s.transfer.dst.kind) for s in program)
                    ok, baseline = wait_for_robot_move(reader, board, move,
                                                       {pad} if via_pad and pad else set(),
                                                       args.vision_timeout, args.grace)
                    log({"event": "robot_move", "time": started, "uci": move.uci(), "san": san,
                         "grasp": grasps, "ok": ok, "duration_s": round(time.time() - started, 1)})
                    if not ok:
                        print(f"stopped: the board did not show {move.uci()} within {args.grace:g} s",
                              file=sys.stderr)
                        code = 2
                        break
                    print(f"{move.uci()} ({san}): on the board")
                    board.push(move)

                else:
                    # ---- Human's turn: watch physical board, send to chess.com ----
                    print("\nYour turn (Black). Play on the board...")
                    started = time.time()
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
                    print(f"Detected: {move.uci()} ({san}): sending to chess.com...")

                    # Send human's move to chess.com
                    try:
                        result = chesscom.play_move(move)
                        if not result.get("success"):
                            print(f"warning: chess.com play-move returned {result}", file=sys.stderr)
                    except Exception as e:
                        print(f"failed to send move to chess.com: {e}", file=sys.stderr)
                        log({"event": "chesscom_send_fail", "time": started, "uci": move.uci(),
                             "error": str(e)})
                        code = 3
                        break

                    # Prime the monitor so it doesn't see our own click as the next
                    # White move (the screenshot diff from our click would confuse it).
                    chesscom.prime()

                    log({"event": "human_move", "time": started, "uci": move.uci(), "san": san,
                         "waited_s": round(time.time() - started, 1)})
                    board.push(move)

            if board.is_game_over():
                result = board.result()
                print(f"\nGame over: {result}")
                log({"event": "game_over", "result": result, "fen": board.fen()})

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

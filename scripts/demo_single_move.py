"""Demo one piece move end to end.

Dry-run (default): build and print the motion program in sim (no hardware). This works today.
Live (--live): execute on hardware; gated by config.measured + explicit approval, and M1.
"""

from __future__ import annotations

import argparse
import sys

import chess

from magnus_bot.cli import _full_reach_map
from magnus_bot.config import load_config
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.orchestration.executor import Executor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo a single move (dry-run by default).")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--uci", default="e2e4")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="position to move from")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--i-have-approval", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    board = chess.Board(args.fen)
    move = chess.Move.from_uci(args.uci)
    if move not in board.legal_moves:
        print(f"{args.uci} is not legal in {args.fen}", file=sys.stderr)
        return 2

    if args.live and not args.i_have_approval:
        print("Refusing: --live needs explicit operator approval (--i-have-approval).", file=sys.stderr)
        return 1

    rmap: ReachabilityMap = _full_reach_map(config)  # PLACEHOLDER until reach_report (M2)
    arms = {arm_id: SimArm(arm_id) for arm_id in config.arms.arms}
    executor = Executor(config, arms, rmap, live=args.live)
    program = executor.execute_move(board, move)
    for i, arm_steps in enumerate(program):
        print(f"[{i}] {arm_steps.arm_id}: {arm_steps.transfer.reason} ({len(arm_steps.steps)} phases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

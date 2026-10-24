"""``magnus-bot`` command-line entry point.

Subcommands:
  * ``info``       - print the loaded config, the measured flag, and the arm placements.
  * ``demo-move``  - build the sim motion program for one move from the start position and print
                     it. Dry-run only; never moves hardware.

Live hardware is intentionally not wired into the CLI yet (M1); when it is, it will require
``--live`` plus ``config.measured`` (see CLAUDE.md safety rules).
"""

from __future__ import annotations

import argparse

import chess

from magnus_bot import __version__
from magnus_bot.config import load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.orchestration.executor import build_move_program


def _full_reach_map(config) -> ReachabilityMap:
    """A stand-in reachability map (every arm reaches every square and the pad) for sim demos.

    PLACEHOLDER until scripts/reach_report.py computes the real map from IK (M2).
    """
    all_squares = {(f, r) for f in range(8) for r in range(8)}
    arms = set(config.arms.arms)
    return ReachabilityMap(reachable={a: set(all_squares) for a in arms}, pad_reach=set(arms))


def cmd_info(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    print(f"magnus-bot {__version__}")
    print(f"measured: {config.measured}  (live mode {'enabled' if config.measured else 'REFUSED'})")
    print(f"human plays: {config.board.human_color}  robots play: {config.board.robot_color} (first)")
    print(f"square_size: {config.board.square_size} m   edge_gap d: {config.arms.edge_gap} m")
    for arm_id, p in arm_placements(config.board.square_size, config.arms.edge_gap).items():
        spec = config.arms.arms[arm_id]
        print(f"  {arm_id}: base={p.base_xy.round(4).tolist()} facing={p.facing.round(1).tolist()} "
              f"port={spec.port} calib={spec.calibration_id}")
    return 0


def cmd_demo_move(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    config.warn_if_unmeasured()
    board = chess.Board()
    move = chess.Move.from_uci(args.uci)
    if move not in board.legal_moves:
        print(f"{args.uci} is not legal from the start position")
        return 2
    program = build_move_program(board, move, _full_reach_map(config), config)
    print(f"Move {args.uci}: {len(program)} arm step-sequence(s) (dry-run, placeholder reach map)")
    for i, arm_steps in enumerate(program):
        print(f"  [{i}] arm={arm_steps.arm_id} reason={arm_steps.transfer.reason} "
              f"piece={arm_steps.transfer.piece_symbol} phases={len(arm_steps.steps)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magnus-bot", description="Two-arm chess robot (dry-run).")
    parser.add_argument("--config-dir", default=None, help="path to configs/ (default: packaged)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show config and arm placements").set_defaults(func=cmd_info)

    demo = sub.add_parser("demo-move", help="build the sim motion program for one move")
    demo.add_argument("--uci", default="e2e4", help="move in UCI (default: e2e4)")
    demo.set_defaults(func=cmd_demo_move)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

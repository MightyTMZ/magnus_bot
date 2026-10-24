"""Compute and report each arm's reachable squares, and recommend a handoff pad position.

STUB (M2): depends on analytic IK (kinematics/so101_ik.py), which is M1. Once IK exists this
script samples every square at grasp height per arm, evaluates candidate off-board pad positions,
writes a reachability map (for the assigner) and prints a recommendation.

Read-only: never moves the arms.
"""

from __future__ import annotations

import argparse
import sys

from magnus_bot.config import load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.kinematics.model import SO101Model
from magnus_bot.kinematics.reach import compute_reachability
from magnus_bot.kinematics.so101_ik import SO101Kinematics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachability report (read-only).")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    kin = SO101Kinematics(SO101Model())
    try:
        rmap = compute_reachability(placements, kin, config.arms.z_grasp, config.board.square_size)
    except NotImplementedError as exc:
        print(f"reach_report is not available yet: {exc}", file=sys.stderr)
        print("Implement IK (M1) and reachability (M2) first; see kinematics/so101_ik.py.")
        return 1
    for arm_id, squares in rmap.reachable.items():  # pragma: no cover - unreachable until M2
        print(f"{arm_id}: {len(squares)} reachable squares; pad reachable: {rmap.can_reach_pad(arm_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

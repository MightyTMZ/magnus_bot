"""Compute and report each arm's reachable squares, and recommend a handoff pad position.

Samples every square per arm at grasp height and at grasp + traverse height (what the
pick/place primitive needs), checks the configured pad / bins / stash, searches off-board pad
candidates, and optionally writes the map (``--out``) for the assigner. ``--z-clear`` and
``--riser`` evaluate alternatives (a lower traverse, arm bases raised on a riser).

Read-only: never moves the arms.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from magnus_bot.config import load_config
from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.kinematics.model import SO101Model
from magnus_bot.kinematics.reach import ReachabilityMap, can_reach_point, compute_reachability
from magnus_bot.kinematics.so101_ik import SO101Kinematics

N = board_geom.BOARD_SIZE
LABEL = {"follower_1": "1", "follower_2": "2"}


def render(rmap: ReachabilityMap) -> str:
    """Board grid, rank 8 on top: 1 / 2 = that arm only, B = both, . = neither."""
    lines = ["      " + " ".join("abcdefgh")]
    for r in range(N - 1, -1, -1):
        cells = []
        for f in range(N):
            arms = rmap.arms_reaching(f, r)
            cells.append("B" if len(arms) > 1 else LABEL.get(next(iter(arms)), "?") if arms else ".")
        lines.append(f"  {r + 1}   " + " ".join(cells))
    return "\n".join(lines)


def summary(rmap: ReachabilityMap) -> str:
    union = set().union(*rmap.reachable.values())
    both = set.intersection(*rmap.reachable.values()) if rmap.reachable else set()
    per_arm = ", ".join(f"{a} {len(s)}" for a, s in sorted(rmap.reachable.items()))
    return f"  {len(union)}/64 reachable by some arm, {len(both)} by both ({per_arm})"


def pad_candidates(placements, kin, limits, z_pad, z_clear, square_size, step):
    """Off-board points (outside the board, clear of the bases) reachable by BOTH arms."""
    edge = N * square_size
    found = []
    for x in np.arange(-0.15, edge + 0.15 + 1e-9, step):
        for y in np.arange(-0.15, edge + 0.15 + 1e-9, step):
            if -0.01 <= x <= edge + 0.01 and -0.01 <= y <= edge + 0.01:
                continue  # on (or touching) the board
            xy = np.array([x, y])
            if any(p.distance_to(xy) < 0.06 for p in placements.values()):
                continue  # inside a base footprint
            if all(can_reach_point(p, kin, np.array([x, y, z_pad]), limits, z_clear)
                   for p in placements.values()):
                found.append(xy)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachability report (read-only).")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--z-clear", type=float, default=None, help="override traverse height (m)")
    parser.add_argument("--riser", type=float, default=0.0, help="raise both arm bases by this (m)")
    parser.add_argument("--pad-step", type=float, default=0.01, help="pad search grid step (m)")
    parser.add_argument("--out", type=Path, default=None, help="write the z_clear map as JSON")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    s, z_grasp = config.board.square_size, config.arms.z_grasp
    z_clear = config.arms.z_clear if args.z_clear is None else args.z_clear
    model = SO101Model()
    model = dataclasses.replace(model, base_height=model.base_height + args.riser)
    kin, limits = SO101Kinematics(model), model.limits
    placements = arm_placements(s, config.arms.edge_gap)
    pad = config.handoff.pad
    pad_xyz = np.array([pad.x, pad.y, pad.z])

    print(f"square {s} m, edge gap {config.arms.edge_gap} m, z_grasp {z_grasp} m, "
          f"z_clear {z_clear} m, riser {args.riser} m; top-down grasp, URDF limits")
    if not config.measured:
        print("NOTE: config.measured is false; some inputs are placeholders.")

    grasp_only = compute_reachability(placements, kin, z_grasp, s)
    print(f"\nGrasp height only ({z_grasp} m):\n{render(grasp_only)}\n{summary(grasp_only)}")
    rmap = compute_reachability(placements, kin, z_grasp, s, z_clear=z_clear, pad_xyz=pad_xyz)
    print(f"\nGrasp + hover at z_clear ({z_clear} m), what pick/place needs:\n{render(rmap)}\n"
          f"{summary(rmap)}")

    print(f"\nConfigured pad ({pad.x}, {pad.y}, {pad.z}): reachable by {sorted(rmap.pad_reach) or 'none'}")
    for arm_id, b in sorted(config.handoff.bins.items()):
        ok = can_reach_point(placements[arm_id], kin, np.array([b.x, b.y, b.z]), limits, z_clear)
        print(f"Bin for {arm_id} ({b.x}, {b.y}, {b.z}): {'reachable' if ok else 'NOT reachable'}")
    st = config.handoff.promotion_stash
    stash = [a for a, p in placements.items()
             if can_reach_point(p, kin, np.array([st.x, st.y, st.z]), limits, z_clear)]
    print(f"Promotion stash ({st.x}, {st.y}, {st.z}): reachable by {sorted(stash) or 'none'}")

    found = pad_candidates(placements, kin, limits, pad.z, z_clear, s, args.pad_step)
    if found:
        # Most central: the candidate whose farther base is nearest (both arms work least hard).
        best = min(found, key=lambda xy: max(p.distance_to(xy) for p in placements.values()))
        print(f"\nPad search: {len(found)} off-board points reachable by both arms. "
              f"Recommended: x={best[0]:.3f}, y={best[1]:.3f} (configs/board.yaml: pad)")
    else:
        print("\nPad search: NO off-board point is reachable by both arms "
              f"(grid {args.pad_step} m, pad z {pad.z} m). A pad handoff is impossible here.")

    if args.out:
        args.out.write_text(json.dumps(rmap.to_dict(), indent=1) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

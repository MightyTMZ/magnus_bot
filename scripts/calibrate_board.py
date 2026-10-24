"""Interactively record the board-frame measurements that fill configs/ (square size, edge gap,
pad/bin/stash poses, piece heights).

STUB (M2/M4): read-only prompts that would write measured values back into configs and flip
``measured: true``. Not implemented yet; measure by hand and edit configs/*.yaml for now.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Board calibration helper (not yet implemented).").parse_args(argv)
    print(
        "calibrate_board is not implemented yet.\n"
        "For now: measure the rig by hand and edit configs/board.yaml, arms.yaml, handoff.yaml,\n"
        "then set measured: true in board.yaml. See docs/frames.md for what each value means."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

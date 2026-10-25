"""Helpers shared by the camera-graded bench scripts (scripts/tune_grasp.py, scripts/reach_probe.py).

Both drive ONE arm, in sim (SimArm) or live (LeRobotArm), under the same stop policy: on any stop
every arm holds position, and live torque is released only once the operator says so (safety
rule 3: hold, do not cut torque unless told).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping

from magnus_bot.vision.observation import Occupancy


def board_diagram(occupancy: Occupancy) -> str:
    """The camera's occupancy as an 8x8 text board, rank 8 at the top."""
    rows = [
        f"{rank}  " + " ".join(occupancy.get(f"{f}{rank}", ".") for f in "abcdefgh")
        for rank in range(8, 0, -1)
    ]
    return "\n".join([*rows, "   a b c d e f g h"])


def shut_down(arms: Mapping, live: bool, prompt: Callable[[str], str]) -> None:
    """Hold every arm, then (live) release torque only once the operator says so."""
    for arm in arms.values():
        try:
            if arm.is_connected:
                arm.hold()
        except Exception as e:  # keep going: the other arms still need holding
            print(f"{arm.arm_id}: hold failed: {e}", file=sys.stderr)
    if live:
        try:
            prompt("Arm holding. Support it, then press Enter to release torque and disconnect: ")
        except (EOFError, KeyboardInterrupt):
            pass
    for arm in arms.values():
        if arm.is_connected:
            arm.disconnect()

"""The first real single-arm move: follower_1 picks a rook off g3 and places it on g4.

Scope (deliberately minimal): one arm, one piece, one adjacent square — no handoff, no tilt, no
other pieces in the way. A LOW hover (6 cm) is used, not the rig's 14 cm z_clear: top-down cannot
hover at 14 cm (the REACH issue, pending the tilt work), and an adjacent move over an empty square
only needs to clear the 4.5 cm rook. Grasp height is the rook's measured value (fingertip 5 mm, so
the pad centre lands on the 28 mm neck).

Modes:
  --sim  (default) : run on SimArm, no hardware. The safety-rule-4 sim pass. Safe anywhere.
  --live           : drive follower_1 for real. GATED — needs board.yaml `measured: true`, the
                     --live flag, follower_1's servo map, and an operator present. Start the arm
                     from a safe pose near the board first (NOT the folded rest), and watch it.

    <core-venv>/python scripts/first_move.py            # sim pass (expect OK)
    <lerobot-venv>/python scripts/first_move.py --live  # real move (operator approval)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import chess

from magnus_bot.config import load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import load_servo_map
from magnus_bot.kinematics.reach import compute_reachability
from magnus_bot.kinematics.so101_ik import SO101Kinematics

ARM = "follower_1"
SRC, DST = (6, 2), (6, 3)  # g3 -> g4 (file g = 6; ranks 3, 4)
PIECE = "rook"
HOVER = 0.06  # low first-move hover (m); full z_clear needs the tilt work
SERVO_MAP = Path(__file__).resolve().parents[1] / "hardware" / "servo_maps" / f"{ARM}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="First single-arm move (g3->g4, follower_1).")
    parser.add_argument("--live", action="store_true", help="drive the real arm (gated)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    # Import the executor lazily so --sim never pulls anything hardware-adjacent unexpectedly.
    from magnus_bot.orchestration.executor import Executor, build_move_program

    config = load_config(args.config_dir)
    # Rook grasp height (fingertip target) for every square, and the low first-move hover.
    z_grasp = config.board.grasp_height.get(PIECE, config.arms.z_grasp)
    config = dataclasses.replace(
        config, arms=dataclasses.replace(config.arms, z_grasp=z_grasp, z_clear=HOVER)
    )

    kin = SO101Kinematics()
    placements = arm_placements(config.board.square_size, config.arms.edge_gap)
    rmap = compute_reachability(
        placements, kin, config.arms.z_grasp, config.board.square_size, z_clear=config.arms.z_clear
    )
    for sq, name in ((SRC, "g3"), (DST, "g4")):
        if not rmap.can_reach(ARM, *sq):
            print(f"{ARM} cannot reach {name} at grasp {z_grasp} m / hover {HOVER} m", file=sys.stderr)
            return 1

    board = chess.Board(None)
    board.set_piece_at(chess.square(*SRC), chess.Piece(chess.ROOK, chess.WHITE))
    move = chess.Move(chess.square(*SRC), chess.square(*DST))
    program = build_move_program(board, move, rmap, config)

    if not args.live:
        arms = {a: SimArm(a) for a in config.arms.arms}
        for arm in arms.values():
            arm.connect()
        ex = Executor(config, arms, rmap, simulate=True)
        ex.run(ex.plan(program))  # raises on any unreachable / limit / verify fault
        print(f"SIM OK: {ARM} g3->g4 planned + executed ({len(program[0].steps)} phases), arms at rest.")
        return 0

    # --- live (gated) ---
    from magnus_bot.hardware.lerobot_arm import LeRobotArm

    smap = load_servo_map(SERVO_MAP)  # follower_1 URDF<->servo map
    spec = config.arms.arms[ARM]
    arm = LeRobotArm(arm_id=ARM, port=spec.port, calibration_id=spec.calibration_id)
    arm.connect()
    try:
        ex = Executor(
            config,
            {ARM: arm},
            rmap,
            live=True,  # Executor enforces measured:true + a model->servo map
            to_command=smap.model_to_servo,
            limits=smap.servo_limits(kin.model.limits),
            kinematics=kin,
        )
        print(f"LIVE: driving {ARM} g3->g4. Watch the arm; hand near the power.")
        ex.run(ex.plan(program))
        print(f"LIVE OK: {ARM} completed g3->g4.")
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

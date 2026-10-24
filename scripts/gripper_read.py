"""Live, read-only gripper position monitor — for reading gripper_open / gripper_closed by hand.

Prints ``gripper.pos`` (0..100 percent) continuously. Move the jaws by hand and read the value:
fully open -> gripper_open; closed firmly on a pawn -> gripper_closed. Ctrl-C to stop.

READ-ONLY: it only reads positions; it never commands motion and never changes torque (connects
the bus directly and disconnects with disable_torque=False). To move the jaws by hand the gripper
torque must already be off (power-cycle, or pose the arm by hand first). Needs the lerobot env:

    C:/projects/lerobot/.venv/Scripts/python.exe scripts/gripper_read.py
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from magnus_bot.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live read-only gripper position monitor.")
    parser.add_argument("--arm", default="follower_1", help="arm id in configs/arms.yaml")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--all", action="store_true",
                        help="show every joint (wiggle one physical joint, see which NAME moves)")
    parser.add_argument("--degrees", action="store_true",
                        help="arm joints in degrees (use_degrees=True, as LeRobotArm runs) instead of "
                             "LeRobot's default normalized range; the gripper stays percent")
    args = parser.parse_args(argv)

    spec = load_config(args.config_dir).arms.arms[args.arm]
    units = {"use_degrees": True} if args.degrees else {}  # unset = LeRobot's default, as before
    robot = SO101Follower(SO101FollowerConfig(port=spec.port, id=spec.calibration_id, **units))
    robot.bus.connect()  # bus only: no configure(), torque left as-is
    watching = "every joint" if args.all else "gripper.pos"
    print(f"{args.arm} on {spec.port}: move a joint by hand, watch {watching}. Ctrl-C to stop.")
    last: float | None = None
    try:
        while True:
            obs = {k.removesuffix(".pos"): v for k, v in robot.get_observation().items()
                   if k.endswith(".pos")}
            if args.all:
                print("  " + "  ".join(f"{k}={v:7.2f}" for k, v in obs.items()), end="\r", flush=True)
            else:
                last = obs["gripper"]
                print(f"  gripper.pos = {last:6.2f}    ", end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.bus.disconnect(disable_torque=False)
    if last is not None:
        print(f"\nfinal gripper.pos = {last:.2f}")  # persists after Ctrl-C
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

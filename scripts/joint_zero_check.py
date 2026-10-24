"""Guided, READ-ONLY fit of one arm's URDF-zero <-> servo map (HANDOVER step 1b).

You pose the arm by hand; the script only reads positions. Steps:

  1. Pose the URDF zero "L" (see POSE below) and press Enter.
  2. For each joint: nudge ONLY that joint ~10-20 deg in the direction shown, hold, press Enter,
     then return it to the L. The reading's direction gives the sign; step 1 gives the offset.
     If another joint's reading changes more, the fit stops (scrambled servo ID or a bump).
  3. Check: rest the fingertip on the table, press Enter. The fitted map + FK should put the
     tip at ~0 cm height; it prints the predicted height and reach so you can sanity-check.
  4. Writes hardware/servo_maps/<arm>.json (commit it).

READ-ONLY: connects the bus only (no configure(), no torque change) and disconnects with
disable_torque=False, like gripper_read.py. To pose by hand, torque must already be off
(power-cycle the arm). Needs the lerobot env:

    C:/projects/lerobot/.venv/Scripts/python.exe scripts/joint_zero_check.py --arm follower_1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from magnus_bot.kinematics.conventions import ARM_JOINTS, NudgeError, ServoMap, fit_servo_map, urdf_to_model
from magnus_bot.kinematics.so101_ik import SO101Kinematics

POSE = """URDF zero is an "L", NOT a straight arm:
  - base: arm pointing straight ahead (across the board, toward the other arm)
  - upper arm: nearly vertical, leaning ~14 deg forward
  - forearm: horizontal, pointing forward
  - gripper: in line with the forearm; seen from behind the arm, the moving jaw is up-left"""

# Physical direction of each joint's URDF-positive rotation from the L (pinned by
# tests/test_conventions.py::test_nudge_directions_match_urdf).
NUDGE = {
    "shoulder_pan": "swing the arm to ITS right (clockwise seen from above)",
    "shoulder_lift": "tilt the upper arm forward (elbow moves forward and down)",
    "elbow_flex": "lower the forearm (wrist moves down)",
    "wrist_flex": "tip the gripper down",
    "wrist_roll": "roll the gripper counter-clockwise, seen from behind the arm",
}

OUT_DIR = Path(__file__).resolve().parents[1] / "hardware" / "servo_maps"
SAMPLES = 10  # readings averaged per step (the bus is noisy at the ~0.1 deg level)

Reader = Callable[[], dict[str, float]]  # arm joints in radians (servo convention)


def averaged(read: Reader, n: int = SAMPLES, pause: float = 0.02) -> dict[str, float]:
    readings = []
    for _ in range(n):
        readings.append(read())
        if pause:
            time.sleep(pause)
    return {j: float(np.mean([r[j] for r in readings])) for j in ARM_JOINTS}


def run_check(arm_id: str, read: Reader, ask: Callable[[str], str] = input,
              say: Callable[[str], None] = print, pause: float = 0.02) -> tuple[ServoMap, float, float]:
    """The guided steps. Returns (servo map, predicted tip height m, predicted reach m)."""
    say(POSE)
    ask("\nPose the L, hold it still, press Enter... ")
    zero = averaged(read, pause=pause)
    nudged = {}
    for joint in ARM_JOINTS:
        ask(f"\n{joint}: {NUDGE[joint]} by ~10-20 deg, hold, press Enter... ")
        nudged[joint] = averaged(read, pause=pause)
        moved = math.degrees(nudged[joint][joint] - zero[joint])
        say(f"  {joint} reading moved {moved:+.1f} deg. Return it to the L.")
    smap = fit_servo_map(arm_id, zero, nudged)

    ask("\nCheck: rest the fingertip on the table (any comfortable spot), press Enter... ")
    tip = SO101Kinematics().forward(urdf_to_model(smap.servo_to_urdf(averaged(read, pause=pause))))
    height, reach = float(tip[2]), float(math.hypot(tip[0], tip[1]))
    say(f"  predicted tip height {height * 100:+.1f} cm (expect ~0), "
        f"{reach * 100:.1f} cm from the base yaw axis (measure it with a ruler)")
    if abs(height) > 0.015:
        say("  WARNING: more than 1.5 cm off. Re-pose the L more carefully and re-run.")
    return smap, height, reach


def report(smap: ServoMap) -> str:
    rows = [f"  {n:14s} sign {s:+d}  offset {math.degrees(o):+7.2f} deg" for n, (s, o) in smap.joints.items()]
    return "\n".join([f"servo = sign * urdf + offset for {smap.arm_id} "
                      "(expected ~ +1 / small offsets for so101_new_calib):", *rows])


def lerobot_reader(arm_id: str, config_dir: str | None):  # pragma: no cover - needs lerobot + arm
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from magnus_bot.config import load_config

    spec = load_config(config_dir).arms.arms[arm_id]
    robot = SO101Follower(SO101FollowerConfig(port=spec.port, id=spec.calibration_id, use_degrees=True))
    robot.bus.connect()  # bus only: no configure(), torque left as-is

    def read() -> dict[str, float]:
        obs = robot.get_observation()
        return {j: math.radians(float(obs[f"{j}.pos"])) for j in ARM_JOINTS}

    return read, lambda: robot.bus.disconnect(disable_torque=False)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - interactive, needs the rig
    parser = argparse.ArgumentParser(description="Read-only URDF-zero <-> servo fit for one arm.")
    parser.add_argument("--arm", default="follower_1", help="arm id in configs/arms.yaml")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    read, close = lerobot_reader(args.arm, args.config_dir)
    try:
        smap, _, _ = run_check(args.arm, read)
    except NudgeError as exc:
        print(f"\nStopped: {exc}")
        return 1
    finally:
        close()
    print("\n" + report(smap))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.arm}.json"
    out.write_text(json.dumps(smap.to_dict(), indent=2) + "\n")
    print(f"\nWrote {out}. Commit it, then repeat for the other arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

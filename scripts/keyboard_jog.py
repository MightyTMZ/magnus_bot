"""Drive ONE SO-101 follower joint-by-joint from the keyboard (live bring-up helper).

Talks to LeRobot directly (like scripts/hardware_selftest.py, NOT through
magnus_bot.hardware.LeRobotArm), so it runs in the lerobot env and uses the same proven
unit handling. SAFETY: this MOVES an arm. Every command is clamped to the arm's calibrated
range and to a small per-command step (max_relative_target); torque is released on quit.

    python scripts/keyboard_jog.py                 # follower_1 on COM3
    python scripts/keyboard_jog.py --id follower_2 --port COM4

Controls (each keypress nudges the SELECTED joint by the current step):
    1..6          select joint (see the on-screen list)
    up / down     jog selected joint + / -        (also: w / s)
    [ / ]         decrease / increase step size
    r             re-read and hold current pose (clears accumulated target)
    space         print current positions
    q / Esc       quit (releases torque)

Positions are degrees for the revolute joints and percent (0-100) for the gripper, matching
LeRobot's observation units on this rig.
"""

from __future__ import annotations

import argparse
import msvcrt
import sys

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# Base (shoulder_pan) -> tip ordering, so the numbered list reads top-of-arm down.
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

STEP_DEFAULT = 2.0   # degrees (gripper: percent) per keypress
STEP_MIN = 0.5
STEP_MAX = 10.0
MARGIN = 2.0         # keep this far inside each calibrated limit
MAX_STEP = 10.0      # per-command safety clamp handed to LeRobot (max_relative_target)


def read_positions(robot: SO101Follower) -> dict[str, float]:
    obs = robot.get_observation()
    return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}


def limits_deg(robot: SO101Follower) -> dict[str, tuple[float, float]]:
    """Calibrated (low, high) of each joint, in the units get_observation reports."""
    out = {}
    for joint, cal in robot.bus.calibration.items():
        if joint == "gripper":
            out[joint] = (0.0, 100.0)
        else:
            mid = (cal.range_min + cal.range_max) / 2
            out[joint] = ((cal.range_min - mid) * 360 / 4095, (cal.range_max - mid) * 360 / 4095)
    return out


def getkey() -> str:
    """Block for one keypress; return a normalized token ('up','down','q','1',' ', ...)."""
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):  # arrow / function keys send a two-byte sequence
        code = msvcrt.getch()
        return {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}.get(code, "")
    if ch == b"\x1b":
        return "q"  # Esc
    try:
        return ch.decode("ascii").lower()
    except UnicodeDecodeError:
        return ""


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo + MARGIN, min(hi - MARGIN, value))


def print_state(selected: str, step: float, target: dict, lims: dict) -> None:
    print("\n  joint          target     calibrated range")
    for i, joint in enumerate(JOINT_ORDER, start=1):
        lo, hi = lims[joint]
        marker = ">" if joint == selected else " "
        print(f"  {marker}{i} {joint:<13}{target[joint]:8.2f}   [{lo:7.1f}, {hi:7.1f}]")
    print(f"  step = {step:.1f}   (selected: {selected})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard jog one SO-101 arm (LIVE).")
    parser.add_argument("--id", default="follower_1", help="calibration id (default: follower_1)")
    parser.add_argument("--port", default="COM3", help="serial port (default: COM3)")
    args = parser.parse_args(argv)

    cfg = SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=MAX_STEP)
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.disconnect()
        sys.exit(f"{args.id} ({args.port}): calibration does not match the servos. Recalibrate first.")

    lims = limits_deg(robot)
    target = read_positions(robot)
    robot.send_action({f"{j}.pos": v for j, v in target.items()})  # hold where it is
    selected = JOINT_ORDER[0]
    step = STEP_DEFAULT

    print(f"Connected to {args.id} on {args.port}. Keys: 1-6 select, up/down jog, [ ] step, "
          "r re-read, space show, q quit.")
    print_state(selected, step, target, lims)
    try:
        while True:
            key = getkey()
            if key == "q":
                break
            elif key in "123456":
                selected = JOINT_ORDER[int(key) - 1]
                print_state(selected, step, target, lims)
            elif key in ("up", "w", "down", "s"):
                direction = 1.0 if key in ("up", "w") else -1.0
                lo, hi = lims[selected]
                target[selected] = clamp(target[selected] + direction * step, lo, hi)
                sent = robot.send_action({f"{j}.pos": v for j, v in target.items()})
                target = {k.removesuffix(".pos"): float(v) for k, v in sent.items()}
                print(f"  {selected:<13}-> {target[selected]:8.2f}")
            elif key == "[":
                step = max(STEP_MIN, round(step - 0.5, 1))
                print(f"  step = {step:.1f}")
            elif key == "]":
                step = min(STEP_MAX, round(step + 0.5, 1))
                print(f"  step = {step:.1f}")
            elif key == "r":
                target = read_positions(robot)
                robot.send_action({f"{j}.pos": v for j, v in target.items()})
                print("  re-read current pose and holding")
                print_state(selected, step, target, lims)
            elif key == " ":
                print_state(selected, step, read_positions(robot), lims)
    finally:
        robot.disconnect()  # releases torque per disable_torque_on_disconnect
        print("\nDisconnected (torque released).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

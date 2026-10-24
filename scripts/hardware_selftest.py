"""SO-101 hardware bring-up self-test (folded in from the standalone arm_test project).

This is the proven, standalone tool used to verify the two arms move and return to a baseline.
It talks to LeRobot directly (not through magnus_bot.hardware.LeRobotArm) and is kept as-is for
bring-up. SAFETY: the `test` mode MOVES the arms (gently, always toward the interior of each
calibrated range). Run `baseline` (read-only) first. Ports are hardcoded here (COM3/COM4);
magnus_bot itself reads them from configs/arms.yaml. TODO: source ports/ids from config.

Record a baseline pose for two SO-101 followers, then run a gentle movement demo.

    python test_arms.py baseline   # read-only: saves current joint positions to baseline.json
    python test_arms.py test       # moves each joint a few degrees from baseline, then returns to it

Positions are in degrees (gripper: 0-100). "Baseline" is the pose the arms are in when you run
`baseline`; nothing is written to the servos or calibration files by that step.

The demo never pushes into a calibrated limit: each joint only moves toward the interior of its
range and stays MARGIN clear of both ends, so it is safe to start from a folded rest pose.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

HERE = Path(__file__).parent
BASELINE_FILE = HERE / "baseline.json"
REPORT_FILE = HERE / "report.json"

ARMS = {"follower_1": "COM3", "follower_2": "COM4"}  # id -> port

JOINT_ORDER = ["gripper", "wrist_roll", "shoulder_pan", "wrist_flex", "elbow_flex", "shoulder_lift"]
DELTA = 8.0  # degrees (gripper: percent) a joint swings from baseline
MARGIN = 5.0  # stay at least this far from either calibrated limit
MIN_MOVE = 2.0  # skip a direction with less room than this
MOVE_SECONDS = 1.5  # time for each sweep segment
HZ = 30
MAX_STEP = 4.0  # per-command safety clamp (max_relative_target)
DRIFT_LIMIT = 15.0  # refuse to run `test` if the arm is further than this from baseline
PASS_MOVE_FRACTION = 0.5  # joint must reach at least this fraction of the planned swing
PASS_RETURN_ERR = 3.0  # and end within this many units of baseline


def make_robot(arm_id: str, port: str, max_step: float | None = None) -> SO101Follower:
    cfg = SO101FollowerConfig(port=port, id=arm_id, max_relative_target=max_step)
    return SO101Follower(cfg)


def read_positions(robot: SO101Follower) -> dict[str, float]:
    obs = robot.get_observation()
    return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}


def limits_deg(robot: SO101Follower) -> dict[str, tuple[float, float]]:
    """Calibrated (low, high) of each joint, in the same units get_observation reports."""
    out = {}
    for joint, cal in robot.bus.calibration.items():
        if joint == "gripper":
            out[joint] = (0.0, 100.0)
        else:
            mid = (cal.range_min + cal.range_max) / 2
            out[joint] = ((cal.range_min - mid) * 360 / 4095, (cal.range_max - mid) * 360 / 4095)
    return out


def do_baseline() -> None:
    data = {}
    for arm_id, port in ARMS.items():
        robot = make_robot(arm_id, port)
        robot.bus.connect()  # bus only: no configure(), so torque state is left as it is
        try:
            if not robot.is_calibrated:
                sys.exit(f"{arm_id} ({port}): calibration file does not match the servos. Recalibrate first.")
            pos = read_positions(robot)
            lims = limits_deg(robot)
        finally:
            robot.bus.disconnect(disable_torque=False)
        data[arm_id] = {"port": port, "time": datetime.now().isoformat(timespec="seconds"), "positions": pos}
        print(f"{arm_id} ({port}) baseline:   [position (% of calibrated range)]")
        for joint in JOINT_ORDER:
            lo, hi = lims[joint]
            print(f"    {joint:<14}{pos[joint]:8.2f}  ({100 * (pos[joint] - lo) / (hi - lo):5.1f}%)")
    BASELINE_FILE.write_text(json.dumps(data, indent=2))
    print(f"\nSaved to {BASELINE_FILE}")


def plan_moves(base: float, lo: float, hi: float) -> list[float]:
    """Signed offsets to try from `base`, each staying MARGIN inside [lo, hi]."""
    moves = []
    for sign, room in ((+1, hi - MARGIN - base), (-1, base - (lo + MARGIN))):
        d = min(DELTA, room)
        if d >= MIN_MOVE:
            moves.append(sign * d)
    return moves


def move(robot: SO101Follower, start: dict, goal: dict, joint: str, base: dict, peak: list) -> None:
    """Linearly interpolate every joint from `start` to `goal`, tracking `joint`'s deviation from `base`."""
    steps = max(1, int(MOVE_SECONDS * HZ))
    for i in range(1, steps + 1):
        t = i / steps
        robot.send_action({f"{j}.pos": start[j] + (goal[j] - start[j]) * t for j in start})
        time.sleep(1 / HZ)
        peak[0] = max(peak[0], abs(read_positions(robot)[joint] - base[joint]))


def test_arm(arm_id: str, port: str, base: dict[str, float]) -> list[dict]:
    robot = make_robot(arm_id, port, MAX_STEP)
    robot.connect(calibrate=False)
    results = []
    try:
        if not robot.is_calibrated:
            sys.exit(f"{arm_id} ({port}): calibration does not match the servos. Recalibrate first.")
        lims = limits_deg(robot)
        now = read_positions(robot)
        robot.send_action({f"{j}.pos": v for j, v in now.items()})  # hold where it is
        drift = max(abs(now[j] - base[j]) for j in base)
        if drift > DRIFT_LIMIT:
            sys.exit(
                f"{arm_id}: arm is {drift:.1f} from baseline (limit {DRIFT_LIMIT}). "
                "Put it back in the baseline pose or re-run `baseline`."
            )

        for joint in JOINT_ORDER:
            offsets = plan_moves(base[joint], *lims[joint])
            if not offsets:
                print(f"  {joint:<14} skipped (no room inside calibrated range)")
                continue
            peak = [0.0]
            for off in offsets:
                goal = dict(base)
                goal[joint] = base[joint] + off
                move(robot, dict(base), goal, joint, base, peak)
                move(robot, goal, dict(base), joint, base, peak)
            time.sleep(0.3)
            final_err = abs(read_positions(robot)[joint] - base[joint])
            planned = max(abs(o) for o in offsets)
            ok = peak[0] >= PASS_MOVE_FRACTION * planned and final_err <= PASS_RETURN_ERR
            results.append(
                {"arm": arm_id, "joint": joint, "planned": round(planned, 2),
                 "peak_move": round(peak[0], 2), "return_error": round(final_err, 2), "pass": ok}
            )
            print(f"  {joint:<14} planned {planned:4.1f}  moved {peak[0]:5.2f}  return err {final_err:5.2f}  "
                  f"{'PASS' if ok else 'FAIL'}")
    finally:
        # Torque is released on disconnect; the arm is back at baseline (its rest pose) by then.
        robot.disconnect()
    return results


def do_test() -> None:
    if not BASELINE_FILE.exists():
        sys.exit("No baseline.json. Run `python test_arms.py baseline` first.")
    data = json.loads(BASELINE_FILE.read_text())
    report = []
    for arm_id, port in ARMS.items():
        print(f"\n{arm_id} ({port}):")
        report += test_arm(arm_id, port, data[arm_id]["positions"])
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    failed = [r for r in report if not r["pass"]]
    print(f"\n{len(report) - len(failed)}/{len(report)} joints passed. Report: {REPORT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "test"])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS), help="arms to include")
    args = parser.parse_args()
    ARMS = {k: v for k, v in ARMS.items() if k in args.arms}
    do_baseline() if args.mode == "baseline" else do_test()

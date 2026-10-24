"""ONE-OFF: fix follower_1's scrambled servo bus IDs by rewriting them directly.

Discovered 2026-10-24 (read-only wiggle test): the six servos were ID'd out of order. The bus
ID actually sitting at each physical joint, base->tip, is [5, 6, 1, 2, 4, 3] instead of the
LeRobot-expected [1, 2, 3, 4, 5, 6]:

    physical base yaw      -> currently id 5  (should be 1, shoulder_pan)
    physical shoulder lift -> currently id 6  (should be 2, shoulder_lift)
    physical elbow         -> currently id 1  (should be 3, elbow_flex)
    physical wrist flex    -> currently id 2  (should be 4, wrist_flex)
    physical wrist roll    -> currently id 4  (should be 5, wrist_roll)
    physical gripper       -> currently id 3  (should be 6, gripper)

As current->target the remap is the single 6-cycle (1 3 6 2 4 5). An in-place rewrite would
create a duplicate ID (an unaddressable bus), so we park one servo at a temporary ID (7) and
cascade, so no two servos ever share an ID. This writes servo EEPROM IDs only; it commands NO
motion. After running, RECALIBRATE (the old calibration is bound to the wrong IDs) and re-run
`gripper_read.py --all` to confirm every joint now reports its matching name.

SAFETY: dry-run by default (prints the plan + a ping scan). Pass --run to actually write. If it
fails part-way, a servo may be left at id 7 — re-scan with --dry-run (default) to see the state;
recovery is to finish the remaining moves by hand or fall back to `lerobot-setup-motors`.

Run from the lerobot env:
    C:/projects/lerobot/.venv/Scripts/python.exe scripts/remap_servo_ids.py          # dry-run
    C:/projects/lerobot/.venv/Scripts/python.exe scripts/remap_servo_ids.py --run     # write IDs
"""

from __future__ import annotations

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from magnus_bot.config import load_config

TEMP_ID = 7
# (from_id, to_id) in an order that never collides: park id 1 at the temp, then cascade.
CASCADE = [(1, TEMP_ID), (5, 1), (4, 5), (2, 4), (6, 2), (3, 6), (TEMP_ID, 3)]
EXPECTED_START = {1, 2, 3, 4, 5, 6}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fix follower_1 scrambled servo IDs (one-off).")
    parser.add_argument("--arm", default="follower_1")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--run", action="store_true", help="actually write IDs (else dry-run)")
    args = parser.parse_args(argv)

    spec = load_config(args.config_dir).arms.arms[args.arm]
    bus = SO101Follower(SO101FollowerConfig(port=spec.port, id=spec.calibration_id)).bus
    bus.connect()
    try:
        present = sorted(i for i in range(1, 7) if bus.ping(i) is not None)
        print(f"{args.arm} on {spec.port}: servo IDs currently answering: {present}")

        if set(present) != EXPECTED_START:
            print(f"ABORT: expected exactly {sorted(EXPECTED_START)} before remapping, found {present}.")
            print("The bus is not in the known scrambled start state; not touching it.")
            return 1

        print("Planned remap (no motion; EEPROM ID writes only):")
        for frm, to in CASCADE:
            print(f"  id {frm} -> id {to}")
        if not args.run:
            print("\nDRY-RUN. Re-run with --run to write these IDs.")
            return 0

        definition = bus._definition("shoulder_pan")  # sts3215 definition (same for all joints)
        for frm, to in CASCADE:
            print(f"  writing id {frm} -> {to} ...", end=" ", flush=True)
            bus._connected_bus.change_id(definition, frm, to)
            print("ok")

        final = sorted(i for i in range(1, 7) if bus.ping(i) is not None)
        ok = final == sorted(EXPECTED_START)
        print(f"\nIDs now answering: {final}  ({'OK' if ok else 'UNEXPECTED — investigate'})")
        print("Next: recalibrate (lerobot-calibrate ... --robot.id=follower_1), then gripper_read.py --all.")
        return 0 if ok else 1
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    raise SystemExit(main())

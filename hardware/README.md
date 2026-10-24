# hardware/

Reference hardware artifacts, folded in from the original standalone `arm_test` project. These
are **reference data**, not something the test suite or runtime loads.

## calibration/
- **OBSOLETE (pre servo-ID remap, 2026-10-24):** follower_1's IDs were scrambled and have since
  been re-ID'd + recalibrated (see HANDOVER.md "Known issues"). This backup is from the old,
  wrong mapping — kept only for history; do not derive limits from it. Re-export if needed.
- `follower_1.calibration.bak.json` — a backup of follower_1's LeRobot calibration (Feetech
  per-joint `id`, `drive_mode`, `homing_offset`, `range_min`, `range_max`). The live calibration
  LeRobot actually uses lives in the user cache:
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`. Use this to derive the
  real `JointLimit`s and the URDF-zero↔servo mapping (see `docs/joint_conventions.md`).

## samples/
Example outputs of `scripts/hardware_selftest.py` from a known-good run (follower_1):
- `baseline.example.json` — a recorded baseline pose (joint positions in degrees; gripper %).
- `report.example.json` — a passing per-joint movement report (planned vs. achieved swing and
  return error). All six joints passed.

## Running the self-test
`scripts/hardware_selftest.py baseline` is read-only; `... test` MOVES the arms gently. See that
file's header for safety notes. It requires LeRobot installed and the arms on COM3/COM4.

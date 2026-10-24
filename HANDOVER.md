# HANDOVER.md — magnus_bot

**Read this at the start of every session. Update it before ending one and after each milestone.
Keep it under ~200 lines by summarizing old entries. Never delete decisions — only supersede.**

## Status
Milestone **M0 (scaffold + tests green)**: COMPLETE. 47 tests pass, ruff clean. No hardware has
been driven by magnus_bot code yet.
**M1 Phase 0 (2026-10-24):** 0a env built (lerobot + serial-motors + magnus_bot[hardware] in
`C:\projects\lerobot\.venv`). 0b done (URDF link lengths in `SO101Model`; nominal limits in
docs/joint_conventions.md). 0c partial: recorded square 0.057, gap 0.067, z_grasp 0.028 (pawn),
king 0.099, z_clear 0.140. 0d done. Found + fixed a scrambled servo-ID assignment on follower_1
(see Known issues) and recalibrated. Remaining before `measured: true`: `gripper_open` +
per-piece `gripper_closed` (see Open questions) + the non-king piece dims.

## Last session (2026-10-23)
- Scaffolded the repo: geometry, config, kinematics (IK stub), motion, hardware (`SimArm` full,
  `LeRobotArm` wired but unvalidated), planning (decompose/assign/handoff), orchestration
  (pure state machine + executor), sources, vision stubs, CLI, scripts, docs, tests.
- Pinned the frames with tests (left/right facts, a1 dark, queen placement).
- Verified LeRobot's API against the installed source (`lerobot.robots.so_follower`):
  `SO101FollowerConfig(port, id, max_relative_target, use_degrees=True, ...)`; observations/actions
  use `"<joint>.pos"` keys; joints shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper; gripper is percent 0..100, the rest degrees.
- Folded the sibling `arm_test/` in: `scripts/hardware_selftest.py` (the proven baseline +
  gentle-movement bring-up tool) and `hardware/` (reference calibration + example outputs).

## Next steps (ordered)
1. **M1 — IK**: implement analytic FK/IK in `kinematics/so101_ik.py` (link lengths from the
   SO-101 URDF; document the URDF-zero <-> servo mapping in docs/joint_conventions.md). Add a
   numeric cross-check test (placo/ikpy, the `ik` extra).
2. **M1 — trajectory/executor**: implement `motion/trajectory.plan_joint_trajectory` and the live
   path in `orchestration/executor.Executor.execute_move` (one arm at a time, pad interlock).
3. **M1 — validate `LeRobotArm`** on one arm (slow), read-only first, then a single pick/place.
4. **M2 — reachability**: implement `kinematics/reach.compute_reachability` and
   `scripts/reach_report.py`; recommend a pad position; wire the real map into assignment.
5. Replace the BIN/STASH "any arm" assumption in `planning/assigner.py` with measured reach.

## Open questions (for the operator)
- Done 2026-10-24: square 0.057, gap 0.067, z_grasp 0.028 (pawn), king 0.099, z_clear 0.140;
  servo-ID issue resolved (see Known issues).
- STILL NEEDED to flip `measured: true` (record with `gripper_read.py` on each piece):
  - `gripper_open` (single %)  ->  configs/arms.yaml: `gripper_open`
  - `gripper_closed` PER PIECE (they differ by base diameter)  ->  configs/board.yaml:
    `gripper_closed_pct` (pawn/knight/bishop/rook/queen/king)
  - remaining piece heights + base diameters (only king measured so far) -> configs/board.yaml
- Pad location preference (M2)? Per-arm reachable region (rough) to sanity-check reach (M2)?

## Decisions log
- 2026-10-23: UCI canonical, python-chess owns legality; FEN for logging, SAN for display.
- 2026-10-23: SI (m, rad) in code; degrees/percent only in `hardware/lerobot_arm.py`.
- 2026-10-23: magnus_bot core does NOT depend on lerobot; it is an optional `hardware` extra,
  installed in LeRobot's own environment. Reason: keep sim/planning/tests importable anywhere.
- 2026-10-23: Analytic IK on the control path; numeric solver is a test cross-check only.
- 2026-10-23: Assignment is reachability-driven (not a half-split); pad handoff when needed.

## Known issues (servo)
- RESOLVED 2026-10-24 — follower_1 servo IDs were SCRAMBLED: bus IDs by physical joint (base->tip)
  were `[5, 6, 1, 2, 4, 3]` instead of `[1..6]`, so every joint reported the wrong name (e.g.
  moving the jaws showed up as `elbow_flex`). Found with the read-only `gripper_read.py --all`
  wiggle test. Fixed by directly rewriting the EEPROM IDs — `scripts/remap_servo_ids.py --run`
  (temp-ID-7 cascade of the single 6-cycle) — then `lerobot-calibrate --robot.id=follower_1`,
  re-verified with the wiggle test. If a servo is ever swapped/replaced, re-run that check.
- Feetech buses occasionally return a corrupted status packet under multi-joint motion;
  `num_read_retries` (default 2) covers it (observed in the LeRobot config comments).

## Hardware config
- Ports (Windows): `follower_1` = COM3, `follower_2` = COM4 (also in configs/arms.yaml).
- Calibration ids: `follower_1`, `follower_2` -> LeRobot files at
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`.
- follower_1 RECALIBRATED 2026-10-24 (after the servo-ID remap); the live calibration is in the
  HF cache. The repo copy `hardware/calibration/follower_1.calibration.bak.json` is PRE-REMAP and
  OBSOLETE — do not derive limits from it; re-export from the new calibration if needed.
- Supply voltages: TODO (operator).

## Command cheatsheet
- Tests: `.venv/Scripts/python -m pytest` · Lint: `.venv/Scripts/python -m ruff check .`
- Sim: `magnus-bot info` · `magnus-bot demo-move --uci e2e4`
- Dry-run a move program: `python scripts/demo_single_move.py --uci e2e4`
- LIVE self-test (moves an arm; needs approval): see `scripts/hardware_selftest.py` header.

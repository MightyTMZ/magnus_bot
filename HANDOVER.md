# HANDOVER.md — magnus_bot

**Read this at the start of every session. Update it before ending one and after each milestone.
Keep it under ~200 lines by summarizing old entries. Never delete decisions — only supersede.**

## Status
Milestone **M0 (scaffold + tests green)**: COMPLETE. 47 tests pass, ruff clean. No hardware has
been driven by magnus_bot code yet.

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
- Pad location preference? Square size `s`? Edge gap `d`?
- Piece heights and base diameters (per type)? Gripper open/closed percent?
- Per-arm reachable region (rough), to sanity-check the reach report?
- Which servos have issues (see Known issues)?

## Decisions log
- 2026-10-23: UCI canonical, python-chess owns legality; FEN for logging, SAN for display.
- 2026-10-23: SI (m, rad) in code; degrees/percent only in `hardware/lerobot_arm.py`.
- 2026-10-23: magnus_bot core does NOT depend on lerobot; it is an optional `hardware` extra,
  installed in LeRobot's own environment. Reason: keep sim/planning/tests importable anywhere.
- 2026-10-23: Analytic IK on the control path; numeric solver is a test cross-check only.
- 2026-10-23: Assignment is reachability-driven (not a half-split); pad handoff when needed.

## Known issues (servo) — PLACEHOLDER, operator to fill in
- (TODO: list any servos with play, calibration drift, torque/overload quirks, or flaky reads.)
- Feetech buses occasionally return a corrupted status packet under multi-joint motion;
  `num_read_retries` (default 2) covers it (observed in the LeRobot config comments).

## Hardware config
- Ports (Windows): `follower_1` = COM3, `follower_2` = COM4 (also in configs/arms.yaml).
- Calibration ids: `follower_1`, `follower_2` -> LeRobot files at
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`.
- Reference calibration for follower_1 is in `hardware/calibration/` (servo ranges/homing).
- Supply voltages: TODO (operator).

## Command cheatsheet
- Tests: `.venv/Scripts/python -m pytest` · Lint: `.venv/Scripts/python -m ruff check .`
- Sim: `magnus-bot info` · `magnus-bot demo-move --uci e2e4`
- Dry-run a move program: `python scripts/demo_single_move.py --uci e2e4`
- LIVE self-test (moves an arm; needs approval): see `scripts/hardware_selftest.py` header.

# HANDOVER.md — magnus_bot

**Read this at the start of every session. Update it before ending one and after each milestone.
Keep it under ~200 lines by summarizing old entries. Never delete decisions — only supersede.**

## Status
Milestone **M0 (scaffold + tests green)**: COMPLETE. No hardware has been driven by magnus_bot
code yet. **81 tests pass, 1 skip** (placo cross-check, needs the `ik` env); ruff clean on
`src/ tests/ scripts/` (`RF-DETR/` has 7 lint findings).
**M1 Phase 0 (2026-10-24):** 0a env built (lerobot + serial-motors + magnus_bot[hardware] in
`C:\projects\lerobot\.venv`). 0b done (URDF link lengths in `SO101Model`; nominal limits in
docs/joint_conventions.md). 0c partial: recorded square 0.057, gap 0.067, z_grasp 0.028 (pawn),
king 0.099, z_clear 0.140. 0d done. Found + fixed a scrambled servo-ID assignment on follower_1
(see Known issues) and recalibrated. Remaining before `measured: true`: `gripper_open` +
per-piece `gripper_closed` (see Open questions) + the non-king piece dims.
**M1 Phase 2 (2026-10-24):** analytic FK/IK done (top-down grasp, model convention).
**M1 step 2 (2026-10-24):** trajectory planner + executor joint path done, **sim only**.
**M1 step 1a (2026-10-24):** IK model <-> URDF-zero mapping done and URDF limits in place.
Live still refuses until 1b (URDF <-> servo, needs the rig).

## Last session (2026-10-24, Lucas, part 2) — step 1a
- `kinematics/conventions.py`: `model_to_urdf` / `urdf_to_model` (sign + offset per joint;
  table in docs/joint_conventions.md). The IK keeps its own convention (Alex's design).
- `SO101Model`: added `shoulder_forward = 0.0304` (shoulder_lift axis is 3.04 cm ahead of the
  yaw axis; ignoring it was a ~3.1 cm mean tip error) and corrected `wrist_to_tip` to the
  in-plane 0.1594 (was the 3-D 0.1605). IK uses `shoulder_forward` (2 lines). Limits are now
  the URDF limits (URDF-zero radians); `SimArm` clamps to them.
- `tests/test_conventions.py`: numpy FK of the URDF chain (no placo). Model FK == URDF FK to
  < 0.1 mm at `wrist_roll = 0`; roll != 0 is approximate (tip 7.9 mm off the roll axis).
- Executor sim now commands URDF-zero angles by default (`to_command` = `model_to_urdf`) and
  checks the URDF limits. Wide test limits removed. Sim tests use a 6 cm `z_clear` because
  the rig's 14 cm is infeasible top-down (see REACH).

## Session before (2026-10-24, Lucas, part 1)
- `motion/trajectory.py`: `plan_joint_trajectory` + `segment_duration`. Segment time =
  max over joints of `1.875 · |Δq| / (v_max · speed_scale)` (1.875 = min-jerk peak/average
  speed), so joints start/stop together and none exceeds its limit. PLACEHOLDER per-joint
  `MAX_JOINT_VELOCITY` in `kinematics/model.py` (~1/3 of STS3215 no-load speed).
- `orchestration/executor.py`: modes dry-run (default) / `simulate=True` (SimArm only) /
  `live=True` (needs `measured: true` AND a `to_command` IK -> servo map, else refuses). Plans
  the WHOLE move (IK + limit check + trajectories) before any motion; Unreachable /
  `JointLimitError` abort with zero commands. One arm at a time; each arm returns to its rest
  pose (its pose at move start) before the next moves (clears the pad). Verifies joints after
  each trajectory (0.05 rad, gripper excluded: it stalls on the piece). Any fault -> every arm
  `hold()` -> `ExecutionFault`.
- FIXED an IK bug: `elbow="up"` (the default) returned the elbow-DOWN branch — at h4 the elbow
  was 5.4 cm above the table, over the board, below piece tops. Sign fixed in `so101_ik.py`;
  pinned by `test_elbow_up_is_above_down`.
- Ran a reach check with the real measurements (see Known issues: REACH). Big finding.
- Reviewed teammates' new work (not yet integrated, see "Other workstreams").

## Other workstreams (not integrated with src/magnus_bot yet)
- **`RF-DETR/` (Aurick): working standalone vision loop.** Webcam -> Roboflow hosted model
  `chess-full-kntuv/1` (cloud, `ROBOFLOW_API_KEY`, every 0.5 s) -> homography from 4 clicked
  corners (`Calibrate.py` -> `corners.json`, click order puts a8 top-left) -> per-square W/B
  occupancy -> 3 stable frames -> `find_move` picks the legal move best matching occupancy.
  Gaps: color-only so promotions are ambiguous (e7e8q vs e7e8n); keeps its OWN `chess.Board`;
  Windows-only (`CAP_DSHOW`); must run from `RF-DETR/`; deps not in `pyproject.toml`; does not
  use the `vision/` / `BoardSource` interfaces; arms will occlude it mid-move.
- **`chess_com/magnus_bot/` (Tom): Next.js 16 + Steel cloud browser on chess.com.** Polls the
  page URL every 3 s (a fresh CDP connect per poll) to detect a game/bot page; "Abandon" and
  "Go to live game" buttons; board view is a placeholder. No move read/write, no link to
  Python yet. Sessions time out at 15 min. This is outside CLAUDE.md's stated scope ("no
  frontend/backend") — team to decide (Open questions).

## Next steps (ordered)
1. **M1 1b — URDF zero -> servo (Alex, rig, read-only).** Put the arm in known poses (URDF
   zero = arm straight out horizontally, gripper in line), read degrees with
   `gripper_read.py`, record sign + offset per joint vs. the URDF (expected ~identity for
   `so101_new_calib`, unverified). Use the NEW calibration, not the obsolete repo backup.
   Tighten `SO101Model.limits` from the calibration range. Then live `to_command` =
   `urdf_to_servo(model_to_urdf(q))`.
2. **M1 — validate `LeRobotArm`** on one arm (slow), read-only first, then a single pick/place.
   Needs `--live` + operator approval + `measured: true`.
3. **M2 — reachability** (software): implement `kinematics/reach.compute_reachability` and
   `scripts/reach_report.py` using IK + `model_to_urdf` + URDF limits at BOTH z_grasp and
   z_clear; wire into assignment. Evidence for the REACH decision below.
4. **Decide the REACH fix** (team; see Known issues). Top-down cannot hover high enough
   anywhere, so this now blocks real games. Then re-plan the handoff pad (no off-board point
   is reachable by both).
5. Replace the BIN/STASH "any arm" assumption in `planning/assigner.py` with measured reach.
6. M4: wrap `RF-DETR/` behind `BoardSource` so the orchestrator owns game state and reads the
   camera only when both arms are at rest.

## Open questions (for the operator / team)
- Done 2026-10-24: square 0.057, gap 0.067, z_grasp 0.028 (pawn), king 0.099, z_clear 0.140;
  servo-ID issue resolved (see Known issues).
- STILL NEEDED to flip `measured: true` (record with `gripper_read.py` on each piece):
  - `gripper_open` (single %)  ->  configs/arms.yaml: `gripper_open`
  - `gripper_closed` PER PIECE (they differ by base diameter)  ->  configs/board.yaml:
    `gripper_closed_pct` (pawn/knight/bishop/rook/queen/king)
  - remaining piece heights + base diameters (only king measured so far) -> configs/board.yaml
- REACH: tilted-tool IK, lower/adaptive traverse height, base risers, or a smaller board?
- Product goal: robots vs. human with Stockfish (current CLAUDE.md), or a human playing
  chess.com over the physical board (Tom's app)? Changes the opponent `MoveSource` and needs a
  Python <-> Next.js bridge. Note chess.com fair-play rules forbid engine moves vs. humans.
- Real joint velocity limits (bench-tune `MAX_JOINT_VELOCITY`)?

## Decisions log
- 2026-10-23: UCI canonical, python-chess owns legality; FEN for logging, SAN for display.
- 2026-10-23: SI (m, rad) in code; degrees/percent only in `hardware/lerobot_arm.py`.
- 2026-10-23: magnus_bot core does NOT depend on lerobot; it is an optional `hardware` extra,
  installed in LeRobot's own environment. Reason: keep sim/planning/tests importable anywhere.
- 2026-10-23: Analytic IK on the control path; numeric solver is a test cross-check only.
- 2026-10-23: Assignment is reachability-driven (not a half-split); pad handoff when needed.
- 2026-10-24: Executor plans the whole move before moving; never clamps a plan (out-of-limit ->
  `JointLimitError`, clamping stays a last-resort guard inside the arms).
- 2026-10-24: Arms return to their rest pose (pose at move start) after each transfer.
- 2026-10-24: Live requires an explicit `to_command` joint map; no identity default for hardware.
- 2026-10-24: Three joint conventions (IK model / URDF zero / servo). IK stays in its model
  convention; `kinematics/conventions.py` maps to URDF zero, where limits are checked. Sim
  commands URDF-zero angles.

## Known issues
- **REACH (2026-10-24, open, blocking).** Updated after step 1a (shoulder offset + URDF
  limits, top-down IK, elbow up):
  - At grasp height (2.8 cm): **48/64** squares commandable (a–c / f–h files, fewer near the corners, plus d/e on
    ranks 4–5); none by both arms.
  - At z_clear (14 cm): **0/64**. The wrist would need > 95° (URDF `wrist_flex` limit ±1.658)
    to point the tool down. The highest top-down hover anywhere is **9.6 cm** (b/g files),
    below the king (9.9 cm) and far below the 13.7 cm needed to carry a pawn over a king.
  - So a top-down grasp cannot traverse over tall pieces at all. Options: tilted-tool IK
    (rough model: ~60/64 at 14 cm with ≤45° tilt), routing/lift heights per move, base
    risers, smaller board. Limits are the URDF nominals; the real servo range (1b) may differ.
  - No off-board pad point is reachable by both arms. The farthest square on each arm's half
    (e1/e8) is 0.333 m from the base.
  - Consequence today: sim tests use h4h5 with a 6 cm z_clear; the rig config raises
    `JointLimitError` at hover (pinned by `test_rig_z_clear_breaks_the_wrist_limit_top_down`).
- RESOLVED 2026-10-24 — sim used wide test limits (IK angles vs. servo placeholders); now
  `model_to_urdf` + URDF limits.
- RESOLVED 2026-10-24 — IK `elbow="up"` returned the elbow-down branch (see Last session).
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
- Tests: `.venv/Scripts/python -m pytest` (Windows) · `.venv/bin/python -m pytest` (macOS/Linux)
- Lint: `<venv python> -m ruff check src tests scripts`
- Sim: `magnus-bot info` · `magnus-bot demo-move --uci e2e4`
- Dry-run a move program: `python scripts/demo_single_move.py --uci e2e4`
- Sim-execute a move: `Executor(..., simulate=True)` with a feasible `z_clear` (see tests/test_executor.py `_low_clear`)
- LIVE self-test (moves an arm; needs approval): see `scripts/hardware_selftest.py` header.

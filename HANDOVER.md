# HANDOVER.md — magnus_bot

**Read this at the start of every session. Update it before ending one and after each milestone.
Keep it under ~200 lines by summarizing old entries. Never delete decisions — only supersede.**

## Status
Milestone **M0 (scaffold + tests green)**: COMPLETE. No hardware has been driven by magnus_bot
code yet. **116 tests pass, 1 skip** (placo cross-check, needs the `ik` env); ruff clean on
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
**M2 step 3 (2026-10-24):** reachability map + `reach_report.py` done (top-down IK).
**M1 1b tooling (2026-10-24):** `scripts/joint_zero_check.py` + `ServoMap` ready; needs the rig.
**First real motion + REACH fix (2026-10-24):** both arms driven read-only + gentle self-test
(PASS 12/12). Implemented the tilted-tool IK (the REACH fix) and the approach-collision guard.
**Measured geometry integrated (2026-10-24):** real square (56 mm), PER-ARM gaps (follower_1
85 mm, follower_2 67 mm — asymmetric!), piece heights/base/stem diameters, and per-piece
`grasp_height` (fingertip target = neck − 23 mm gripper offset, floored). Top-down reach is now
**f1=22 (files f-h) / f2=24 (files a-d), union 46**; with ≤50° tilt the FULL board (64) is
reachable. King/queen/rook grasp cleanly; pawn/bishop/knight neck is low → near-tip grasp (to test).
**Servo map / step 1b DONE (2026-10-24):** `joint_zero_check.py` run on both arms →
`hardware/servo_maps/{follower_1,follower_2}.json`. follower_1 validated (tip-on-table check
−0.2 cm). follower_2 vertical joints good, BUT `wrist_roll` offset ≈ −179° (NOT a posing error —
re-posing didn't change it; its wrist_roll servo is calibrated ~180° rotated vs follower_1).
Harmless for round-piece grasps at roll=0; clean up later with `lerobot-calibrate` on follower_2.
First live move will use **follower_1** (clean map, reaches files f–h).
**First move — SIM PASS (2026-10-24):** gripper measured (follower_1: open 77, rook closed 4.15).
`scripts/first_move.py` runs follower_1 g3→g4 (rook, per-piece grasp height, low 6 cm hover since
top-down can't hover at 14 cm): `--sim` planned+executed clean (8 phases, back to rest). `--live`
is written + gated (needs `measured: true` + follower_1 servo map + operator). REMAINING for live:
flip `measured: true`, start follower_1 from a safe near-board pose, run `--live` with approval.

## Last session (2026-10-24, Alex + Claude) — real motion + tilted-tool IK + collision
- **First real motion:** both arms via `scripts/hardware_selftest.py` — read-only `baseline` OK,
  gentle `test` PASS 12/12 (each joint nudged ~8° and returned <0.4°). Physical chain confirmed
  (power/USB/LeRobot/calibration). LeRobot's `max_relative_target` clamp fires on `wrist_flex` at
  the near-limit rest pose (benign). Rest pose sits near elbow/shoulder_lift limits — fine for the
  interior-only self-test, but IK-driven poses must stay off those extremes.
- **Tilted-tool IK (implements the leading REACH fix, Known issues / old next-step 4):**
  `so101_ik.inverse(pitch=…)` + `link_points`; `reach.reach_pose` / `compute_reachability(max_pitch=…)`
  pick the SMALLEST tilt per square ("lean as little as possible") and record a per-square
  `GraspPose`. Default `max_pitch=0` keeps the top-down map (all existing reach tests unchanged);
  tilt is opt-in and not yet the default map/executor path.
- **Approach-collision guard (the knock-over risk the REACH issue flagged):** `motion/collision.py`
  — arm link polyline vs. standing-piece cylinders; `approach_clears` samples the descent;
  `min_clear_pitch` = smallest tilt that reaches AND stays clear. Runtime check (live board), NOT
  baked into the static reach map.
- Branch `feat/reach-pitch-collision` off origin/main. 116 pass / 1 skip, ruff clean.
- NOTE: an earlier local attempt built the same ideas on a stale `main` (pre-pull) and also added
  `scripts/fk_probe.py` — superseded by `joint_zero_check.py`, so dropped in this integration.

## Earlier session (2026-10-24, Lucas, part 4) — tooling for step 1b
- `kinematics/conventions.py`: `ServoMap` (per-arm `servo = sign·urdf + offset`, radians),
  `urdf_to_servo` / `servo_to_urdf` / `model_to_servo` (the live `to_command`), `servo_limits`
  (URDF limits in servo angles), JSON load/save, and `fit_servo_map` (sign from each nudge's
  direction, offset from the L reading; raises `NudgeError` if a nudge is < 3° or another joint
  moved more, which also catches scrambled servo IDs).
- `scripts/joint_zero_check.py`: guided, read-only fit for one arm (procedure in
  docs/joint_conventions.md). Ends with a fingertip-on-table check (FK should say ~0 cm) and
  writes `hardware/servo_maps/<arm>.json`. Not run on hardware yet (no lerobot on the Mac).
- `scripts/gripper_read.py --degrees`: passes `use_degrees=True`; without the flag, unchanged.
- Tests: nudge directions pinned against the URDF; fit/scramble/limits/JSON; the guided script
  end to end with a fake arm (`tests/test_servo_map.py`).

## Earlier sessions (2026-10-24, Lucas) — summary
- **Trajectory + executor (M1 step 2):** `plan_joint_trajectory` times each segment by the
  slowest joint (`1.875·|Δq| / (v_max·speed_scale)`; PLACEHOLDER `MAX_JOINT_VELOCITY`). The
  executor runs dry-run / `simulate` / `live` (live needs `measured: true` + `to_command`). It
  plans the whole move before moving, one arm at a time, back to rest after each transfer,
  verifies joints (gripper excluded) and holds every arm on any fault (`ExecutionFault`).
- **IK fixes (1a):** `elbow="up"` returned the elbow-down branch (fixed, pinned).
  `SO101Model.shoulder_forward = 0.0304` (was a ~3.1 cm tip error) and in-plane
  `wrist_to_tip = 0.1594`. `model_to_urdf` / `urdf_to_model`; URDF limits; numpy URDF FK
  cross-check (`tests/test_conventions.py`, < 0.1 mm at roll 0). Sim commands URDF angles.
- **Reachability (step 3):** `compute_reachability` (grasp AND hover heights, URDF limits),
  `reach_report.py` (`--z-clear`, `--riser`, pad search, `--out`). Not wired into the CLI/demo:
  at the rig's z_clear the map is empty. Findings in Known issues: REACH.

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
1. **M1 1b — URDF zero -> servo (Alex, rig, read-only, both arms).** Power-cycle (torque
   off), then run `scripts/joint_zero_check.py --arm follower_1` (and `follower_2`) in the
   lerobot env. It prints every pose and nudge direction. URDF zero is an **"L"**, NOT a
   straight arm: upper arm nearly vertical (leaning 14° forward), forearm horizontal pointing
   forward, gripper in line. Expect sign +1 / small offsets for `so101_new_calib` (unverified).
   Commit `hardware/servo_maps/*.json`. A `NudgeError` naming another joint = scrambled IDs
   (this covers the follower_2 open question). Then tighten `SO101Model.limits` from the NEW
   calibration's ranges if smaller (not the obsolete repo backup). Live wiring:
   `to_command=smap.model_to_servo, limits=smap.servo_limits(SO101Model().limits)`.
2. **M1 — validate `LeRobotArm`** on one arm (slow), read-only first, then a single pick/place.
   Needs `--live` + operator approval + `measured: true`.
3. ~~M2 — reachability~~ DONE 2026-10-24 (`reach_report.py`), except wiring the cached map
   into the CLI/demo, which waits for step 4.
4. **REACH fix — tilted-tool IK: IMPLEMENTED (opt-in), now INTEGRATE.** `inverse(pitch=…)` +
   `compute_reachability(max_pitch=…)` + `motion/collision.py` exist (see Last session). Remaining:
   (a) choose the default `max_pitch` and flip the cached map / `reach_report.py` / assigner to use
   it (today everything still defaults to top-down); (b) wire `rmap.pose()` pitch + the collision
   check into the executor's descend/lift (and add Cartesian DESCEND/LIFT sub-waypoints, since the
   min-jerk descent strays ~3.5 mm); (c) re-plan the handoff pad (still no point reachable by both
   arms; risers help — `reach_report.py --riser 0.06`).
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
- REACH: tilted-tool IK, base risers, lower/per-move traverse height, or a smaller board? (Data
  in Known issues: REACH.) Also: how to avoid knocking neighbours over with a tilted approach.
- Product goal: robots vs. human with Stockfish (current CLAUDE.md), or a human playing
  chess.com over the physical board (Tom's app)? Changes the opponent `MoveSource` and needs a
  Python <-> Next.js bridge. Note chess.com fair-play rules forbid engine moves vs. humans.
- Real joint velocity limits (bench-tune `MAX_JOINT_VELOCITY`)?
- follower_2: has it had the `gripper_read.py --all` servo-ID wiggle check, and is its
  calibration current? Nothing records it (only follower_1's remap is logged).

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
- 2026-10-24: REACH fix = **tilted-tool IK with a "lean as little as possible" policy** (smallest
  pitch per square), NOT a top-down-only system. Approach pitch is a property of the SQUARE
  (geometry, derived by reach.py); grip firmness is a property of the PIECE (6 measured %s). The
  executor composes them (~per-square pitch + 6 grips), never a per-(piece×square) table.
- 2026-10-24: Piece-collision is checked at EXECUTION time against the live board
  (`motion/collision.py`), not baked into the static reach map (occupancy changes every move).

## Known issues
- **REACH (2026-10-24, open, blocking).** Updated after step 1a (shoulder offset + URDF
  limits, top-down IK, elbow up):
  - At grasp height (2.8 cm): **48/64** squares commandable (a–c / f–h files, fewer near the corners, plus d/e on
    ranks 4–5); none by both arms.
  - At z_clear (14 cm): **0/64**. The wrist would need > 95° (URDF `wrist_flex` limit ±1.658)
    to point the tool down. The highest top-down hover anywhere is **9.6 cm** (b/g files),
    below the king (9.9 cm) and far below the 13.7 cm needed to carry a pawn over a king.
  - So a top-down grasp cannot traverse over tall pieces at all. Limits are the URDF nominals;
    the real servo range (1b) may differ.
  - Minimum tool tilt from vertical (planar model + URDF limits, tilt toward the base, the only
    direction a 5-DoF SO-101 can tilt): grasp height 0° on 48 squares, up to 22° (d/e files
    on ranks 1/8). Hover at 14 cm: 17–39° everywhere except a4/a5/h4/h5 (too close to the
    base; impossible at any tilt). So **straight-down grasping still works on most squares;
    only the lifted part of the move needs tilt**. No side grip needed anywhere.
  - 6 cm riser under both bases: `reach_report.py --riser 0.06` gives 28/64 top-down at 14 cm
    (52/64 at grasp height); with tilt, hover needs ≤ 28° and a4/a5/h4/h5 become possible.
  - Knock-over risks to handle in the tilt work: the tilt is always toward the arm's base, so
    the neighbour on the base side is the one at risk; descend along the tool axis; the
    joint-space min-jerk descent strays ~3.5 mm sideways from a straight line (add Cartesian
    sub-waypoints for DESCEND/LIFT); check gripper vs. neighbour pieces (board state + piece
    dims) before moving, like the limit check; vision check after each robot move.
  - No off-board pad point is reachable by both arms. The farthest square on each arm's half
    (e1/e8) is 0.333 m from the base.
  - Consequence today: sim tests use h4h5 with a 6 cm z_clear; the rig config raises
    `JointLimitError` at hover (pinned by `test_rig_z_clear_breaks_the_wrist_limit_top_down`).
  - UPDATE 2026-10-24: the tilted-tool IK + collision guard are now implemented (opt-in; see Last
    session + next-step 4). Still OPEN until the default map/executor use tilt and the pad is
    re-planned; a 6 cm riser likely still wanted to keep tilt angles modest.
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

## Environment setup (new device)
TWO separate environments — the core never imports lerobot (keeps sim/tests portable).
- **Core / dev / sim — any device, no hardware.** Prereqs: git, `uv`, Python 3.13.
  `git clone https://github.com/MightyTMZ/magnus_bot.git && cd magnus_bot`;
  `uv venv --python 3.13 .venv`; `uv pip install --python .venv -e ".[dev]"`. Then
  `.venv/Scripts/python -m pytest` (expect 116 pass / 1 skip) and `-m ruff check src tests`.
  Runs tests, sim (`magnus-bot info` / `demo-move`), reach + collision. macOS/Linux: `.venv/bin/`.
- **Hardware — ONLY the machine wired to the arms.** LeRobot lives in its OWN venv (this rig:
  `C:\projects\lerobot\.venv`; path is rig-specific). Install LeRobot per its upstream docs
  (Python 3.10+, Feetech/serial backend) [confirm exact steps w/ repo owner], then install
  magnus_bot into that same venv (`pip install -e <path>`, lerobot already present). Ports in
  `configs/arms.yaml` (COM3/COM4). Per-arm calibration is device-specific and NOT in the repo —
  it lives in the HF cache (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`);
  on a new rig calibrate each arm (`lerobot-calibrate --robot.id=follower_1`, then `follower_2`).
  The repo's `hardware/calibration/*.bak.json` is a PRE-remap reference — OBSOLETE for limits.
  Read-only bring-up (needs neither `measured:true` nor `--live`):
  `scripts/hardware_selftest.py baseline` and `scripts/joint_zero_check.py --arm follower_1`.

## Command cheatsheet
- Tests: `.venv/Scripts/python -m pytest` (Windows) · `.venv/bin/python -m pytest` (macOS/Linux)
- Lint: `<venv python> -m ruff check src tests scripts`
- Sim: `magnus-bot info` · `magnus-bot demo-move --uci e2e4`
- Dry-run a move program: `python scripts/demo_single_move.py --uci e2e4`
- Sim-execute a move: `Executor(..., simulate=True)` with a feasible `z_clear` (see tests/test_executor.py `_low_clear`)
- LIVE self-test (moves an arm; needs approval): see `scripts/hardware_selftest.py` header.

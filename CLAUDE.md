# CLAUDE.md — magnus_bot

**Read HANDOVER.md first. Update it last (before ending a session, and after each milestone).**

## Mission
Two SO-101 follower arms (LeRobot, Feetech STS3215) automate the opponent in a physical chess
game against a human. The arms cooperate through a handoff pad, orchestrated by a state machine.
Current scope: controller, IK, planning, orchestration. No frontend/backend. Computer vision runs
as a separate local service (`RF-DETR/vision_server.py`, own venv) that the core reads over
localhost; it grades robot transfers today. The robots play **White** and move first; the human plays **Black** and
sits at the rank-8 edge (`human_color: black`, flippable).

## Frames and orientation facts (source of truth: docs/frames.md)
- **Board frame B**: origin at the a1 corner, +x toward the h-file, +y toward rank 8, +z up.
  Square (file f=0..7 for a..h, rank r=0..7 for 1..8) has center `((f+0.5)s, (r+0.5)s)`;
  `s` = square size. **a1 is DARK**, h1 is light.
- **Arms sit OFF the board**, beyond opposite short edges, each facing along a rank:
  - `follower_1`: beyond the h-file edge (near h4/h5), base `(8s + d, 4s)`, facing **-x**.
    From its view **LEFT = h1, RIGHT = h8**.
  - `follower_2`: beyond the a-file edge (near a4/a5), base `(-d, 4s)`, facing **+x**.
    From its view **LEFT = a8, RIGHT = a1**.
  - `d` = base-to-board-edge gap (measured; placeholder for now).
- **Arm frame A_i**: origin on the base yaw axis, +x forward, +y left, +z up. "Left" of a
  heading is that heading rotated **+90 degrees** (`transforms.perp_left`).
- Pinned by tests (fix the frames, never the tests): the left/right facts above; a1 dark /
  h1 light; the standard start puts the black queen on d8 and the white queen on d1 at the
  right coordinates.

## Units & move format
- **SI everywhere in code** (meters, radians). Degrees/percent only at the hardware edge
  (`hardware/lerobot_arm.py`).
- UCI strings are canonical (e2e4, e7e8q); python-chess `Board` owns state and legality; FEN
  for logging, SAN for display only. Every `MoveSource.next_move(board)` returns a move
  validated against `board.legal_moves`.

## Repo map
- `configs/` — board.yaml, arms.yaml, handoff.yaml (all PLACEHOLDER; `measured: false`).
- `src/magnus_bot/geometry/` — board squares, arm frames, transforms.
- `src/magnus_bot/kinematics/` — joint model, analytic IK (stub, M1), reachability.
- `src/magnus_bot/motion/` — safety clamps, pick/place primitives, min-jerk trajectory.
- `src/magnus_bot/hardware/` — `Arm` protocol, `SimArm` (full), `LeRobotArm` (stub).
- `src/magnus_bot/planning/` — move decomposition, arm assignment, handoff.
- `src/magnus_bot/orchestration/` — state machine (pure), executor, closed-loop grasp tuning.
- `src/magnus_bot/sources/` — Stockfish / CLI / RandomLegal / BoardSource (vision stub).
- `src/magnus_bot/vision/` — stdlib client for the vision server, transfer grading
  (`verify.py`), `SimulatedBoard`; plus the old in-process Camera/homography stubs.
- `RF-DETR/` — vision service + calibration (its own venv `RF-DETR/.venv`; torch+CUDA).
- `scripts/` — reach_report, jog_arm, first_move, tune_grasp, reach_probe, fit_arm_geometry,
  calibrate_board, hardware_selftest.
- `hardware/` — reference calibration + example self-test outputs (folded from arm_test).
- `tests/`, `docs/`.

## Commands
- Setup: `uv venv --python 3.13 .venv && uv pip install --python .venv -e ".[dev]"`
  (lerobot is NOT installed here; it lives in its own environment — see HANDOVER.md).
- Test: `.venv/Scripts/python -m pytest`
- Lint: `.venv/Scripts/python -m ruff check .`
- Info / sim demo: `magnus-bot info` · `magnus-bot demo-move --uci e2e4`
- Reach report (M2): `python scripts/reach_report.py`

## Conventions
- Analytic IK is the control path; a numeric solver (placo/ikpy) is only ever a test
  cross-check. Document the URDF-zero <-> servo-calibration mapping in docs/joint_conventions.md.
- Arm assignment uses the reachability map (not a hard-coded half-split). One arm reaches both
  src and dst -> single-arm; otherwise a pad handoff (A: src->pad->retract; B: pad->dst->retract).
- One arm moves at a time; the other holds a rest pose; interlock around the pad.
- Trajectories: min-jerk between waypoints; speed scale default 0.3 (from config).

## Safety rules (verbatim)
1. Default mode is dry-run/sim. Live hardware requires `--live` AND operator approval in-session
   AND `measured: true`.
2. Clamp every joint target to configured limits; speed scale comes from config.
3. On any fault: stop commanding and hold position; do not cut torque unless told so.
4. Never run two-arm motion without a sim pass of the same plan first.
5. Known servo issues exist; see HANDOVER.md.

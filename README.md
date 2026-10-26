# magnus_bot

Two SO-101 follower arms that play **White** in a physical chess game against a human, cooperating
through a handoff pad under a state-machine orchestrator. The core covers control, IK, planning and
orchestration; a separate local vision service (RF-DETR) reads the board from an overhead camera.
There is no frontend/backend (the `chess_com/` Next.js app is a separate bridge to a chess.com game).

> **Status:** the arms play real moves on the real board (single-arm moves, pad handoffs and
> captures), each one graded by the camera. The rig is measured (`configs/board.yaml`
> `measured: true`). **No command moves hardware without `--live` and an in-session "yes" from
> the operator.** See `CLAUDE.md` and `HANDOVER.md`.

## Quick start

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv -e ".[dev]"      # core + dev tools (NOT lerobot; see below)
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check .
magnus-bot info                                # config + arm placements
magnus-bot demo-move --uci e2e4                # build a sim motion program for one move
```

`lerobot` is **not** a core dependency: the sim, planner and tests import without it. For live
hardware, install the `hardware` extra inside LeRobot's own environment (see `HANDOVER.md`).

## Playing

Every script runs in sim by default; `--live` (from the lerobot venv, with
`RF-DETR/vision_server.py` running) drives the arms after the whole plan passes in sim.

- `scripts/play_moves.py --game scholars-mate`: a saved game; the camera watches for each of
  the human's moves (N passes for the human).
- `scripts/play_stockfish.py [--elo 1500]`: Stockfish plays White with the arms; the camera reads
  any legal reply. If the arms cannot play the engine's pick, the next best playable move is used.
- `scripts/play_chess_com.py --session-id <id>`: mirrors a chess.com game through the
  `chess_com/` app (a Steel browser session).

After a robot move, a board that looks wrong is a warning; the game stops (arms holding) only if
it stays wrong for `--grace` seconds (60).

## How it works

A chess move (UCI) is decomposed into physical **piece transfers** (captures remove the victim
first; castling moves king then rook; en passant removes the pawn behind the destination;
promotion swaps a stashed queen). Each transfer is **assigned** to an arm using a reachability
map: one arm that reaches both ends does it alone, otherwise the two arms **hand off** over a
shared pad. Each transfer expands into the pick/place primitive (hover -> descend -> grip -> lift ->
traverse -> descend -> release -> retract). A pure **state machine** sequences the game loop.

## Layout

`geometry/` frames & squares · `kinematics/` IK & reach · `motion/` safety, primitives,
trajectory · `hardware/` `Arm` + `SimArm` + `LeRobotArm` · `planning/` decompose/assign/handoff ·
`orchestration/` state machine + executor + grasp tuning · `sources/` engines & input ·
`vision/` vision-server client + transfer grading. `RF-DETR/` is the vision service (own venv).
Docs: `docs/frames.md`, `docs/state_machine.md`, `docs/joint_conventions.md`.

# magnus_bot

Two SO-101 follower arms that play **White** in a physical chess game against a human, cooperating
through a handoff pad under a state-machine orchestrator. Scope today: controller, IK, planning,
orchestration. Computer vision is an interface stub only; there is no frontend/backend.

> **Status: M0 (scaffold).** Sim and planning work end to end; IK, live motion, reachability and
> vision are stubs for later milestones. **No command moves hardware without `--live`, in-session
> operator approval, and `measured: true`.** See `CLAUDE.md` and `HANDOVER.md`.

## Quick start

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv -e ".[dev]"      # core + dev tools (NOT lerobot; see below)
.venv/Scripts/python -m pytest                 # 47 tests
.venv/Scripts/python -m ruff check .
magnus-bot info                                # config + arm placements
magnus-bot demo-move --uci e2e4                # build a sim motion program for one move
```

`lerobot` is **not** a core dependency — the sim, planner and tests import without it. For live
hardware, install the `hardware` extra inside LeRobot's own environment (see `HANDOVER.md`).

## How it works

A chess move (UCI) is decomposed into physical **piece transfers** (captures remove the victim
first; castling moves king then rook; en passant removes the pawn behind the destination;
promotion swaps a stashed queen). Each transfer is **assigned** to an arm using a reachability
map: one arm that reaches both ends does it alone, otherwise the two arms **hand off** over a
shared pad. Each transfer expands into the pick/place primitive (hover → descend → grip → lift →
traverse → descend → release → retract). A pure **state machine** sequences the game loop.

## Layout

`geometry/` frames & squares · `kinematics/` IK & reach · `motion/` safety, primitives,
trajectory · `hardware/` `Arm` + `SimArm` + `LeRobotArm` · `planning/` decompose/assign/handoff ·
`orchestration/` state machine + executor · `sources/` engines & input · `vision/` camera stubs.
Docs: `docs/frames.md`, `docs/state_machine.md`, `docs/joint_conventions.md`.

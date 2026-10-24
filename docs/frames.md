# Frames and orientation

All coordinates are SI (meters, radians). This file is the source of truth for the geometry the
unit tests pin; **fix the code to match this, never the tests.**

## Board frame B

- Origin at the **a1 corner** of the board.
- **+x** toward the h-file, **+y** toward rank 8, **+z** up from the table.
- A square is indexed by `(file f, rank r)` with `f = 0..7` for files a..h and `r = 0..7` for
  ranks 1..8 (identical to python-chess `square_file` / `square_rank`).
- The **center** of square `(f, r)` is at `((f + 0.5)·s, (r + 0.5)·s)`, where `s` is the square
  size (center-to-center spacing).
- **a1 is a dark square.** Equivalently, `(f + r)` even ⇒ dark; so h1 (7, 0) is light.

```
 y (rank 8)
 ^
 |  a8 b8 ... h8
 |  ...
 |  a1 b1 ... h1
 +----------------> x (h-file)
 (a1 corner = origin)
```

## Players

The human plays **Black** and sits at the rank-8 edge (`+y` side). The robots play **White** and
move first. `configs/board.yaml: human_color` flips this; the code derives `robot_color` from it.

## Arm placement

Both arms sit **off** the board, beyond opposite short edges, each facing along a rank. `d` is the
gap from an arm's base yaw axis to the near board edge.

| arm        | near edge | base `(x, y)` | facing | LEFT | RIGHT |
|------------|-----------|---------------|--------|------|-------|
| follower_1 | h-file    | `(8s + d, 4s)`| `-x`   | h1   | h8    |
| follower_2 | a-file    | `(-d, 4s)`    | `+x`   | a8   | a1    |

"Left" of a heading `f` is `f` rotated **+90°** (counter-clockwise): `(x, y) → (-y, x)`
(`transforms.perp_left`). Check: follower_1 faces `(-1, 0)`, so its left is `(0, -1)` — toward
rank 1, i.e. **h1**; follower_2 faces `(1, 0)`, left `(0, 1)` — toward rank 8, i.e. **a8**.

## Arm frame A_i

Origin on the arm's base yaw axis; **+x** forward (along `facing`), **+y** left
(`perp_left(facing)`), **+z** up (shared with B). `frames.ArmPlacement.board_to_arm` /
`arm_to_board` convert points between B and A_i in the plane; z is common to both.

## Measurements (all PLACEHOLDER until `measured: true`)

`s` (square size), `d` (edge gap), piece heights & base diameters, gripper open/closed, grasp
height `z_grasp`, traverse height `z_clear`, and the pad/bin/stash poses. These live in
`configs/*.yaml`. Never copy a test's expected value from a measured number, or vice versa.

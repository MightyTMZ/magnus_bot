# Joint conventions (URDF zero vs. servo calibration)

This is the known pitfall for SO-101 IK: the angle the **URDF/analytic model** calls "zero" for a
joint is almost never the angle the **servo calibration** calls zero. Getting the per-joint
offset and sign wrong makes IK point the arm somewhere plausible-looking but physically wrong.
Pin every joint here before trusting live motion (M1).

## Joints (LeRobot SO101Follower order)

Verified against the installed source (`lerobot/robots/so_follower/so_follower.py`):

| # | name | motor id | hardware unit | role |
|---|------|----------|---------------|------|
| 1 | shoulder_pan | 1 | degrees | base yaw (IK: `atan2(y, x)`) |
| 2 | shoulder_lift | 2 | degrees | planar 2-link, proximal |
| 3 | elbow_flex | 3 | degrees | planar 2-link, distal |
| 4 | wrist_flex | 4 | degrees | wrist pitch (fixed for top-down grasp) |
| 5 | wrist_roll | 5 | degrees | gripper jaw orientation |
| 6 | gripper | 6 | **percent 0..100** | end effector open/close |

LeRobot runs with `use_degrees=True`; the five arm joints are degrees, the gripper is percent.
magnus_bot works in **radians** and converts only in `hardware/lerobot_arm.py`
(`ARM_JOINTS` get rad↔deg; the gripper passes through).

## IK model <-> URDF zero (DONE 2026-10-24)

Three conventions exist: the **IK model** (angles from +z; `kinematics/so101_ik.py`), the
**URDF zero** (this file's tables), and the **servo** (what LeRobot commands). The first hop is
pure geometry and lives in `kinematics/conventions.py`:

```
model = sign · urdf + offset        (radians)
```

| joint | sign | offset (rad) | why |
|-------|------|--------------|-----|
| shoulder_pan | −1 | 0 | URDF pan axis points down (−z) |
| shoulder_lift | +1 | 0.243783 | upper arm is 13.97° from vertical at URDF zero |
| elbow_flex | +1 | 1.288485 | forearm 87.79° from vertical, minus the upper arm |
| wrist_flex | +1 | 0.088104 | wrist→tip line 92.84° from vertical, minus the forearm |
| wrist_roll | +1 | 0 | passed through; IK uses 0 |

Pinned by `tests/test_conventions.py`, which walks the URDF chain with numpy (no placo):
model FK and URDF FK agree to < 0.1 mm at `wrist_roll = 0`. Non-zero roll is approximate: the
tip sits 7.9 mm off the roll axis. The "tool vertical" the IK enforces is the wrist→tip line;
the gripper frame's own axis is 2.8° off it (negligible for a grasp).

Joint limits (`SO101Model.limits`) are now the URDF limits below, in URDF-zero radians, and are
checked on the URDF-convention commands (after the mapping).

## URDF zero <-> servo (TODO, needs the rig)

For each arm joint, record, from the SO-101 URDF and the per-arm calibration JSON
(`range_min`, `range_max`, `homing_offset`, `drive_mode`; see `hardware/calibration/`):

```
servo_deg = sign · (urdf_rad · 180/π) + offset_deg      # per joint
```

- **sign** (±1): does the servo count the same rotation direction as the URDF? (`drive_mode`.)
- **offset_deg**: servo reading when the URDF model is at its zero pose.
- **limits**: map `range_min`/`range_max` (servo counts) through the same relation into the
  `JointLimit` (radians) in `kinematics/model.py`. Replace the PLACEHOLDER symmetric limits.

Reference calibration (`hardware/calibration/follower_1.calibration.bak.json`) records, per joint,
the Feetech `range_min/range_max` (0..4095 counts) and `homing_offset`. The self-test's
`limits_deg()` converts counts to degrees as `(count − mid)·360/4095` with `mid =
(range_min+range_max)/2`; mirror that when deriving limits.

## Link lengths and nominal limits (from the URDF)

Extracted from `hardware/urdf/so101_new_calib.urdf` by composing the joint transforms at the zero
pose. Link lengths are already filled into `kinematics/model.py: SO101Model`:

| field | value (m) | between |
|-------|-----------|---------|
| base_height | 0.1166 | base/table → shoulder_lift axis |
| upper_arm | 0.1160 | shoulder_lift → elbow_flex |
| forearm | 0.1350 | elbow_flex → wrist_flex |
| shoulder_forward | 0.0304 | yaw axis → shoulder_lift axis (horizontal) |
| wrist_to_tip | 0.1594 | wrist_flex → gripper_frame_link, in-plane (3-D length 0.1605; the 1.83 cm along the pitch axis cancels the shoulder's lateral offset) |

Axis directions at the zero pose confirm the model: shoulder_pan ‖ world z (yaw);
shoulder_lift / elbow_flex / wrist_flex ‖ world y (parallel pitch ⇒ planar 2-link);
wrist_roll ‖ world x (tool roll).

Nominal joint limits straight from the URDF (radians, URDF-zero convention). These are now
`SO101Model.limits` (2026-10-24); tighten per-arm from the calibration JSON if a servo's usable
range is smaller:

| joint | lower (rad) | upper (rad) | degrees |
|-------|-------------|-------------|---------|
| shoulder_pan | -1.9199 | +1.9199 | ±110.0 |
| shoulder_lift | -1.7453 | +1.7453 | ±100.0 |
| elbow_flex | -1.6900 | +1.6900 | ±96.8 |
| wrist_flex | -1.6581 | +1.6581 | ±95.0 |
| wrist_roll | -2.7439 | +2.8412 | -157.2..+162.8 |
| gripper | — | — | percent 0..100 at the edge (URDF models it as -0.1745..+1.7453 rad) |


## Validation (M1)

1. FK of the recorded calibration/rest pose should match the physically observed fingertip.
2. IK then FK of a grid of reachable targets should round-trip to within tolerance.
3. Cross-check IK against a numeric solver (placo/ikpy, the `ik` extra) on the same targets.
4. Only then command a real arm — slowly, read-only first (see `scripts/hardware_selftest.py`).

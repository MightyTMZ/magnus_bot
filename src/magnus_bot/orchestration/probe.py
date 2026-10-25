"""Reach probe: send the arm to a square, hold it there, and measure where it really is.

An arm can land in the wrong place for two different reasons, with different fixes. Three
measurements at the held pose tell them apart:

* **joints**: each joint's commanded angle vs the angle it actually reached. A gap means the
  motors sag or lag (soft gains, gravity, the per-command clamp): a control problem.
* **model**: where the config geometry puts the fingertip for the angles actually reached. Its
  offset from the target is the part of the miss the joint gap explains.
* **camera**: where the green fingertip really is. Camera minus model is the **geometry** error
  (base position, link lengths, joint zeros), which one calibration fixes on every square at once.

The camera maps pixels to the board SURFACE, so a fingertip held above it appears shifted away
from the camera (parallax: ~1 mm per mm of height on this rig). Measuring the tip at two heights
gives that shift per mm; extrapolating back to height 0 removes it.

Offsets are reported in mm along the arm's reach (+ = too far, - = short of the target) and
sideways (+ = to the arm's left). Pure: planning and analysis only, nothing here moves an arm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import chess
import numpy as np

from magnus_bot.config import ArmCorrection
from magnus_bot.geometry import board as board_geom
from magnus_bot.geometry.frames import ArmPlacement, arm_placements
from magnus_bot.geometry.transforms import perp_left, unit
from magnus_bot.kinematics.conventions import ARM_JOINTS
from magnus_bot.kinematics.model import JointLimit
from magnus_bot.kinematics.so101_ik import SO101Kinematics
from magnus_bot.motion.trajectory import plan_joint_trajectory
from magnus_bot.orchestration.executor import check_limits

JointMap = Callable[[Mapping[str, float]], dict[str, float]]

MOTOR_GAP_DEG = 1.5  # a reached-vs-commanded gap above this counts as the motors not getting there
GEOMETRY_GAP_MM = 5.0  # an AVERAGE camera-vs-model gap above this: one offset everywhere (e.g. edge_gap)
SQUARE_GAP_MM = 10.0  # any ONE square off by more than this: the error varies across the board
HEIGHT_GAP_MM = 2.0  # an average ruler-vs-commanded height gap above this is a height error


def square_xy(square: str, square_size: float) -> np.ndarray:
    """Board-frame center of a named square (meters)."""
    index = chess.parse_square(square)
    return board_geom.square_center(chess.square_file(index), chess.square_rank(index), square_size)


@dataclass(frozen=True)
class ProbePlan:
    """How to reach one square: joint commands above it and at the two held heights."""

    square: str
    target_xy: np.ndarray  # board frame, m
    target_z: float  # fingertip height at the (low) held pose, m
    above: dict[str, float]  # joint command at the hover height
    at: dict[str, float]  # joint command at the low held pose
    z2: float | None = None  # second held height (parallax), m; None = single height
    at2: dict[str, float] | None = None  # joint command at z2
    correction: ArmCorrection = ArmCorrection()  # the arm's measured landing error (config arms.*_correction)

    def down(self, rest: Mapping[str, float], speed_scale: float, hz: float) -> list[dict[str, float]]:
        """Samples from ``rest`` to above the square to the low held pose."""
        return plan_joint_trajectory([dict(rest), self.above, self.at], speed_scale, hz)

    def raise_to_z2(self, speed_scale: float, hz: float) -> list[dict[str, float]]:
        """Samples straight up from the low held pose to the second height."""
        return plan_joint_trajectory([self.at, self.at2], speed_scale, hz)

    def up(self, rest: Mapping[str, float], speed_scale: float, hz: float) -> list[dict[str, float]]:
        """Samples from the last held pose back up and home to ``rest``."""
        last = self.at2 if self.at2 is not None else self.at
        return plan_joint_trajectory([last, self.above, dict(rest)], speed_scale, hz)


def plan_probe(
    square: str,
    placement: ArmPlacement,
    kinematics: SO101Kinematics,
    to_command: JointMap,
    limits: Mapping[str, JointLimit],
    *,
    square_size: float,
    z: float,
    hover: float,
    pitch: float,
    gripper: float,
    z2: float | None = None,
    correction: ArmCorrection | None = None,
) -> ProbePlan:
    """Joint commands to hold the fingertip at height ``z`` over ``square`` (then at ``z2``, if
    given, for the parallax measurement; and at ``hover`` on the way in and out). ``correction`` is
    applied to every command, as the executor does, so the probe checks the calibration in use.

    Raises :class:`~magnus_bot.kinematics.so101_ik.Unreachable` or
    :class:`~magnus_bot.orchestration.executor.JointLimitError` before anything moves.
    """
    xy = square_xy(square, square_size)
    x, y = placement.board_to_arm(xy)
    correction = correction or ArmCorrection()

    def command(height: float) -> dict[str, float]:
        joints = to_command(kinematics.inverse(correction.apply(np.array([x, y, height])), pitch=pitch))
        joints["gripper"] = gripper
        check_limits(joints, limits)
        return joints

    return ProbePlan(square, xy, z, above=command(hover), at=command(z),
                     z2=z2, at2=command(z2) if z2 is not None else None, correction=correction)


def remove_parallax(low_xy: np.ndarray, high_xy: np.ndarray, z_low: float, z_high: float) -> np.ndarray:
    """Where the fingertip is in the board plane, from where the camera saw it at two heights.

    The apparent position moves linearly with height (a camera ray), so extrapolate to height 0.
    """
    per_m = (np.asarray(high_xy) - np.asarray(low_xy)) / (z_high - z_low)
    return np.asarray(low_xy) - z_low * per_m


def split_mm(offset_xy: np.ndarray, reach_dir: np.ndarray) -> tuple[float, float]:
    """A board-frame offset (m) as (along the reach, to the arm's left), in mm."""
    return float(1000 * offset_xy @ reach_dir), float(1000 * offset_xy @ perp_left(reach_dir))


@dataclass(frozen=True)
class ProbeReport:
    """What the probe measured at one square."""

    square: str
    joint_error_deg: dict[str, float]  # reached - commanded, per arm joint
    model_mm: tuple[float, float]  # (reach, side) of the model fingertip from the COMMANDED target
    model_z_mm: float  # model fingertip height minus the commanded height
    camera_mm: tuple[float, float] | None  # (reach, side) of the seen fingertip from the target
    geometry_mm: tuple[float, float] | None  # camera minus model: what joints + corrections miss
    parallax_mm: tuple[float, float] | None = None  # apparent shift per 10 mm of height, if measured
    height_error_mm: float | None = None  # ruler gap minus commanded height (- = tip lower than told)

    @property
    def worst_joint(self) -> tuple[str, float]:
        name = max(self.joint_error_deg, key=lambda n: abs(self.joint_error_deg[n]))
        return name, self.joint_error_deg[name]

    def lines(self) -> list[str]:
        joints = "  ".join(f"{n} {e:+.1f}" for n, e in self.joint_error_deg.items())
        out = [f"{self.square}:",
               f"  joints, reached minus commanded (deg): {joints}",
               f"  joint gap alone moves the tip: {self.model_mm[0]:+.0f} mm reach, "
               f"{self.model_mm[1]:+.0f} mm side, {self.model_z_mm:+.0f} mm height"]
        if self.camera_mm is None:
            out.append("  camera: fingertip NOT seen (hidden by the arm or a piece? paint visible?)")
        else:
            if self.parallax_mm is not None:
                out.append(f"  parallax removed: the tip appears to shift {self.parallax_mm[0]:+.1f} mm "
                           f"reach, {self.parallax_mm[1]:+.1f} mm side per 10 mm of height")
            out.append(f"  camera: tip is {self.camera_mm[0]:+.0f} mm along the reach (- = short), "
                       f"{self.camera_mm[1]:+.0f} mm sideways from the square's center")
            out.append(f"  geometry (camera minus model): {self.geometry_mm[0]:+.0f} mm reach, "
                       f"{self.geometry_mm[1]:+.0f} mm side")
        if self.height_error_mm is not None:
            out.append(f"  height (ruler): tip is {self.height_error_mm:+.1f} mm from the commanded height "
                       "(- = lower than told)")
        return out

    def to_record(self) -> dict:
        return {
            "square": self.square,
            "joint_error_deg": {n: round(e, 3) for n, e in self.joint_error_deg.items()},
            "model_mm": [round(v, 1) for v in self.model_mm],
            "model_z_mm": round(self.model_z_mm, 1),
            "camera_mm": None if self.camera_mm is None else [round(v, 1) for v in self.camera_mm],
            "geometry_mm": None if self.geometry_mm is None else [round(v, 1) for v in self.geometry_mm],
            "parallax_mm": None if self.parallax_mm is None else [round(v, 2) for v in self.parallax_mm],
            "height_error_mm": None if self.height_error_mm is None else round(self.height_error_mm, 1),
        }


def analyze(
    plan: ProbePlan,
    reached: Mapping[str, float],
    *,
    placement: ArmPlacement,
    kinematics: SO101Kinematics,
    to_model: JointMap,
    camera_xy: np.ndarray | None,
    camera_xy2: np.ndarray | None = None,
    measured_gap_mm: float | None = None,
) -> ProbeReport:
    """Grade one held pose: ``reached`` are the joint angles read back at the low hold (command
    convention), ``camera_xy`` / ``camera_xy2`` where the camera saw the fingertip at the low hold
    and at ``plan.z2`` (board frame, m; None if not seen). With both, parallax is removed.
    ``measured_gap_mm`` is the tip-to-board gap the operator measured at the low hold, if any."""
    errors = {n: math.degrees(reached[n] - plan.at[n]) for n in ARM_JOINTS}
    tip = kinematics.forward(to_model(reached))
    model_xy = placement.arm_to_board(tip[:2])
    reach_dir = unit(plan.target_xy - placement.base_xy)
    # Where the commands aimed (the square plus the configured correction): the model tip is
    # graded against this, the camera against the square itself.
    target_arm = np.append(placement.board_to_arm(plan.target_xy), plan.target_z)
    commanded = plan.correction.apply(target_arm)
    commanded_xy = placement.arm_to_board(commanded[:2])
    model_mm = split_mm(model_xy - commanded_xy, reach_dir)
    camera_mm = geometry_mm = parallax_mm = None
    if camera_xy is not None:
        seen = np.asarray(camera_xy)
        if camera_xy2 is not None and plan.z2 is not None:
            seen = remove_parallax(camera_xy, camera_xy2, plan.target_z, plan.z2)
            per_10mm = (np.asarray(camera_xy2) - np.asarray(camera_xy)) * 0.010 / (plan.z2 - plan.target_z)
            parallax_mm = split_mm(per_10mm, reach_dir)
        camera_mm = split_mm(seen - plan.target_xy, reach_dir)
        geometry_mm = (camera_mm[0] - model_mm[0], camera_mm[1] - model_mm[1])
    return ProbeReport(
        square=plan.square,
        joint_error_deg=errors,
        model_mm=model_mm,
        model_z_mm=1000 * (tip[2] - commanded[2]),
        camera_mm=camera_mm,
        geometry_mm=geometry_mm,
        parallax_mm=parallax_mm,
        height_error_mm=None if measured_gap_mm is None else measured_gap_mm - 1000 * plan.target_z,
    )


@dataclass(frozen=True)
class ProbeSample:
    """One probed hold for :func:`fit_geometry`: where the model put the tip (arm frame, from the
    joints read back) and where the camera saw it on the board (parallax removed), both in m."""

    square: str
    model_tip_xy: np.ndarray
    seen_xy: np.ndarray


@dataclass(frozen=True)
class GeometryFit:
    """The arm geometry that best explains a set of probes."""

    edge_gap: float  # base yaw axis -> near board edge, m
    correction: ArmCorrection  # reach / side / lateral landing error (z is left to the ruler)
    rms_mm: float
    residuals_mm: list[tuple[float, float]]  # per sample, in order: (reach, side) the fit still misses


def predict_seen(sample: ProbeSample, arm_id: str, square_size: float, edge_gap: float,
                 correction: ArmCorrection) -> np.ndarray:
    """Where the tip would really be (board frame) if the base sits ``edge_gap`` off the board and
    the arm lands ``correction`` short / to its right of its model."""
    placement = arm_placements(square_size, edge_gap)[arm_id]
    return placement.arm_to_board(correction.undo(np.append(sample.model_tip_xy, 0.0))[:2])


def fit_geometry(samples: list[ProbeSample], arm_id: str, square_size: float,
                 start_gap: float) -> GeometryFit:
    """Least-squares edge_gap + reach + side + lateral corrections from camera probes.

    The base gap and the reach shortfall look alike on the arm's centre line but not off it: a gap
    error rotates the line to every square (sideways misses that grow away from the centre line),
    a reach shortfall does not. A base shifted sideways makes squares on one side of the arm land
    long and on the other short (2026-10-25 follower_2: rank 3 -4 mm, rank 6 +14 mm). So the probes
    must spread across ranks: 4+ squares on 2+ ranks."""
    if len(samples) < 4 or len({chess.square_rank(chess.parse_square(s.square)) for s in samples}) < 2:
        raise ValueError("need 4+ probed squares on 2+ ranks to separate the base gap from the reach")
    params = np.array([start_gap, 0.0, 0.0, 0.0])

    def residuals(p: np.ndarray) -> np.ndarray:
        c = ArmCorrection(reach=p[1], side=p[2], lateral=p[3])
        return np.concatenate([predict_seen(s, arm_id, square_size, p[0], c) - s.seen_xy for s in samples])

    for _ in range(20):  # Gauss-Newton; the problem is nearly linear, so this converges in a few steps
        res = residuals(params)
        jac = np.column_stack([(residuals(params + d) - res) / 1e-6 for d in np.eye(4) * 1e-6])
        step = np.linalg.lstsq(jac, -res, rcond=None)[0]
        params = params + step
        if np.max(np.abs(step)) < 1e-7:
            break
    correction = ArmCorrection(reach=float(params[1]), side=float(params[2]), lateral=float(params[3]))
    placement = arm_placements(square_size, float(params[0]))[arm_id]
    per_sample = [split_mm(s.seen_xy - predict_seen(s, arm_id, square_size, params[0], correction),
                           unit(s.seen_xy - placement.base_xy)) for s in samples]
    rms = 1000 * float(np.sqrt(np.mean(residuals(params) ** 2) * 2))  # per-point planar rms
    return GeometryFit(float(params[0]), correction, rms, per_sample)


def verdict(reports: list[ProbeReport]) -> list[str]:
    """What the reports say, across all the squares probed."""
    if not reports:
        return ["nothing was probed"]
    out = []
    worst = max((r.worst_joint for r in reports), key=lambda w: abs(w[1]))
    if abs(worst[1]) >= MOTOR_GAP_DEG:
        out.append(f"MOTORS: {worst[0]} ended {worst[1]:+.1f} deg off its command. The servos are not "
                   "reaching the angles sent (soft gains / gravity / the 4 deg per-command clamp): "
                   "fix control first.")
    else:
        out.append(f"motors OK: every joint within {MOTOR_GAP_DEG} deg of its command.")
    heights = [r.height_error_mm for r in reports if r.height_error_mm is not None]
    if heights:
        mean = float(np.mean(heights))
        if abs(mean) >= HEIGHT_GAP_MM:
            out.append(f"HEIGHT: by ruler the tip sits {mean:+.1f} mm from the commanded height on average "
                       f"({'lower' if mean < 0 else 'higher'} than told): grasps aim that far "
                       f"{'too low' if mean < 0 else 'too high'}.")
        else:
            out.append(f"height OK: by ruler within {HEIGHT_GAP_MM:.0f} mm of the commanded height.")
    seen = [r for r in reports if r.geometry_mm is not None]
    if not seen:
        out.append("camera: the fingertip was never seen, so geometry could not be measured.")
        return out
    reach = float(np.mean([r.geometry_mm[0] for r in seen]))
    side = float(np.mean([r.geometry_mm[1] for r in seen]))
    worst = max(seen, key=lambda r: max(abs(r.geometry_mm[0]), abs(r.geometry_mm[1])))
    worst_mm = max(abs(worst.geometry_mm[0]), abs(worst.geometry_mm[1]))
    systematic = max(abs(reach), abs(side)) >= GEOMETRY_GAP_MM
    if systematic:
        out.append(f"GEOMETRY: on average the tip lands {reach:+.0f} mm along the reach and {side:+.0f} mm "
                   "sideways from where the config says it should. The config geometry is off (arm base "
                   "position / edge_gap, link lengths, joint zeros): calibrate.")
    if worst_mm >= SQUARE_GAP_MM:
        out.append(f"GEOMETRY VARIES: {worst.square} is off by {worst.geometry_mm[0]:+.0f} mm reach, "
                   f"{worst.geometry_mm[1]:+.0f} mm side. The error changes from square to square (a joint "
                   "scale or zero, a base rotation), so no single offset fixes it.")
    if not systematic and worst_mm < SQUARE_GAP_MM:
        out.append(f"geometry OK: on average within {GEOMETRY_GAP_MM:.0f} mm, and every square within "
                   f"{SQUARE_GAP_MM:.0f} mm.")
    return out

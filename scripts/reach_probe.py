"""Reach probe: send the arm to squares, hold it there, and measure where it really is.

The first live tuning run (2026-10-24) saw follower_1 stop SHORT of g3 at every tilt, so tuning the
grip is pointless until the arm reaches. This finds out why. For each square in --squares:

  1. move slowly to --hover above the square with the gripper closed, then lower the fingertip
     to --z;
  2. HOLD, settle, and measure: the joint angles read back vs those commanded; where the config
     geometry puts the fingertip for the angles reached; where the camera sees the green tip;
  3. rise to --z2 and look again: the apparent shift with height is camera parallax, which is
     then removed (extrapolated to the board surface);
  4. lift and return to where the arm started.

Per square it prints the joint gaps (deg) and offsets (mm, along the reach / sideways), then a
verdict: MOTORS (the servos do not reach their commands) and/or GEOMETRY (the config is off:
calibrate). One JSON line per square goes to hardware/tuning_logs/probe_*.jsonl: the input for a
calibration fit.

  sim:   python scripts/reach_probe.py --squares g3,f4
         (SimArm. The "camera" sees the model fingertip shifted by --sim-camera-offset-mm, and
         --sim-sag-deg makes shoulder_lift fall short, to dry-run both verdicts.)
  live:  <lerobot-venv>/python scripts/reach_probe.py --live --squares g3

Live needs board.yaml `measured: true`, the arm's servo map, RF-DETR/vision_server.py running, the
target squares EMPTY (the closed jaws come down to --z there), and a typed "yes". Nothing is
connected or moved before the "yes". Start the arm from a safe pose near the board; Ctrl+C holds it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import chess
import numpy as np

from magnus_bot.config import ArmCorrection, load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.geometry.transforms import perp_left, unit
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import load_servo_map, model_to_urdf, urdf_to_model
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable
from magnus_bot.orchestration.bench import board_diagram, shut_down
from magnus_bot.orchestration.executor import JointLimitError
from magnus_bot.orchestration.probe import analyze, plan_probe, verdict
from magnus_bot.vision.observation import (
    DEFAULT_VISION_URL,
    BoardObservation,
    HttpBoardReader,
    VisionUnavailable,
)

ROOT = Path(__file__).resolve().parents[1]
MIN_Z = 0.003  # m: the jaw tips must never touch the board
HZ = 30.0


class SaggingSimArm(SimArm):
    """A SimArm whose shoulder_lift settles ``sag`` radians off every command (sim only)."""

    def __init__(self, arm_id: str, sag: float) -> None:
        super().__init__(arm_id)
        self.sag = sag

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        targets = dict(targets)
        if "shoulder_lift" in targets:
            targets["shoulder_lift"] += self.sag
        return super().send_joints(targets)

    def hold(self) -> dict[str, float]:
        return SimArm.send_joints(self, dict(self._state))


class SimCamera:
    """The vision server's stand-in in sim: sees the fingertip where the model puts it for the arm's
    current joints, shifted by ``offset_mm`` = (along the reach, to the arm's left), plus a parallax
    of ``parallax_mm`` (board x, y) per 10 mm of fingertip height, like the real oblique camera.
    The simulated arm lands off its model by the config's ``correction`` (short, to its right, low),
    as the real one does (that is what the correction compensates), so commands land where intended."""

    def __init__(self, arm, placement, kinematics, to_model, square_size: float, offset_mm,
                 parallax_mm=(0.0, 0.0), correction: ArmCorrection | None = None) -> None:
        self.arm, self.placement, self.kin, self.to_model = arm, placement, kinematics, to_model
        self.square_size, self.offset_mm, self.parallax_mm = square_size, offset_mm, parallax_mm
        self.correction = correction or ArmCorrection()

    def read(self, fresh: bool = True, timeout: float = 8.0) -> BoardObservation:
        tip = self.kin.forward(self.to_model(self.arm.read_joints()))
        tip = self.correction.undo(tip)  # the physical arm lands this far short / right / low
        xy = self.placement.arm_to_board(tip[:2])
        u = unit(xy - self.placement.base_xy)
        xy = xy + (self.offset_mm[0] * u + self.offset_mm[1] * perp_left(u)) / 1000.0
        xy = xy + np.array(self.parallax_mm) / 1000.0 * (tip[2] / 0.010)
        return BoardObservation(occupancy={}, marker=tuple(xy / self.square_size), marker_seen=5)


def stream(arm, samples: list[dict[str, float]], live: bool) -> None:
    """Send a trajectory sample by sample, paced at HZ live (as fast as possible in sim)."""
    period = 1.0 / HZ if live else 0.0
    deadline = time.monotonic()
    for command in samples:
        arm.send_joints(command)
        if period:
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))


def median_joints(arm, reads: int = 5) -> dict[str, float]:
    """Joint angles read back several times, per-joint median (servo reads are a little noisy)."""
    samples = [arm.read_joints() for _ in range(reads)]
    return {name: float(np.median([s[name] for s in samples])) for name in samples[0]}


def _pair(text: str) -> tuple[float, float]:
    values = [float(v) for v in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"expected two numbers 'a,b', got {text!r}")
    return values[0], values[1]


def main(argv: list[str] | None = None, prompt=input) -> int:
    parser = argparse.ArgumentParser(description="Hold the arm over squares and measure where it really is.")
    parser.add_argument("--squares", default="g3", help="comma-separated, e.g. g3,f4,h2 (must be empty live)")
    parser.add_argument("--arm", default="follower_1")
    parser.add_argument("--z", type=float, default=0.010, help="fingertip height at the hold, m")
    parser.add_argument("--z2", type=float, default=0.040,
                        help="second hold height for the parallax correction, m (0 = skip it)")
    parser.add_argument("--hover", type=float, default=0.06, help="height above the square first, m")
    parser.add_argument("--pitch", type=float, default=0.0, help="tool tilt from vertical, degrees")
    parser.add_argument("--speed", type=float, default=0.2, help="trajectory speed scale (0..1)")
    parser.add_argument("--settle", type=float, default=1.5,
                        help="seconds to let the servos settle at the hold")
    parser.add_argument("--gripper", type=float, default=None,
                        help="gripper percent during the probe (default: arms.yaml gripper_closed)")
    parser.add_argument("--ask-height", action="store_true",
                        help="at each low hold, wait while you measure the tip-to-board gap (mm)")
    parser.add_argument("--live", action="store_true", help="drive the real arm (gated)")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--vision-timeout", type=float, default=8.0)
    parser.add_argument("--sim-camera-offset-mm", default="0,0",
                        help="sim only: geometry error the camera sees, 'reach,side' mm (use = if negative)")
    parser.add_argument("--sim-sag-deg", type=float, default=0.0, help="sim only: shoulder_lift falls short")
    parser.add_argument("--sim-parallax-mm", default="0,0",
                        help="sim only: apparent tip shift per 10 mm of height, board 'x,y' mm")
    parser.add_argument("--log", type=Path, default=None,
                        help="JSONL path (default hardware/tuning_logs/...)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    try:
        squares = [s.strip() for s in args.squares.split(",") if s.strip()]
        for s in squares:
            chess.parse_square(s)
        camera_offset = _pair(args.sim_camera_offset_mm)
        sim_parallax = _pair(args.sim_parallax_mm)
    except ValueError as e:
        parser.error(str(e))
    if args.z < MIN_Z:
        parser.error(f"--z {args.z} m would put the jaw tips within {MIN_Z * 1000:.0f} mm of the board")
    z2 = args.z2 or None
    if z2 is not None and not args.z < z2 < args.hover:
        parser.error(f"--z2 must lie between --z ({args.z}) and --hover ({args.hover}), or be 0")

    config = load_config(args.config_dir)
    if args.arm not in config.arms.arms:
        parser.error(f"unknown arm {args.arm!r}; configured: {sorted(config.arms.arms)}")
    size = config.board.square_size
    kin = SO101Kinematics()
    placement = arm_placements(size, config.arms.edge_gap)[args.arm]
    correction = config.arms.correction(args.arm)
    gripper = args.gripper if args.gripper is not None else config.arms.gripper_closed
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("" if args.live else "_sim")
    log_path = args.log or ROOT / "hardware" / "tuning_logs" / f"probe_{args.arm}_{stamp}.jsonl"

    if args.live:
        try:
            config.require_measured()  # also gate before touching the arm
            smap = load_servo_map(ROOT / "hardware" / "servo_maps" / f"{args.arm}.json")
        except (RuntimeError, OSError) as e:
            print(f"live refused: {e}", file=sys.stderr)
            return 1
        to_command = smap.model_to_servo

        def to_model(joints):
            return urdf_to_model(smap.servo_to_urdf(joints))

        limits = smap.servo_limits(kin.model.limits)
    else:
        to_command, to_model, limits = model_to_urdf, urdf_to_model, kin.model.limits

    plans = []
    for square in squares:  # plan everything before anything moves
        try:
            plans.append(plan_probe(square, placement, kin, to_command, limits, square_size=size,
                                    z=args.z, hover=args.hover, pitch=math.radians(args.pitch),
                                    gripper=gripper, z2=z2, correction=correction))
        except (Unreachable, JointLimitError) as e:
            print(f"skipping {square}: cannot plan it ({e})", file=sys.stderr)
    if not plans:
        print("nothing to probe", file=sys.stderr)
        return 1

    if args.live:
        reader = HttpBoardReader(args.vision_url)
        try:
            obs = reader.read(fresh=True, timeout=args.vision_timeout)
        except VisionUnavailable as e:
            print(f"vision: {e}", file=sys.stderr)
            return 1
        print(board_diagram(obs.occupancy))
        occupied = [p.square for p in plans if p.square in obs.occupancy]
        if occupied:
            print(f"clear {', '.join(occupied)} first: the closed jaws come down to "
                  f"{args.z * 1000:.0f} mm there", file=sys.stderr)
            return 1
        then = f", rises to {z2 * 1000:.0f} mm and looks again" if z2 else ""
        print(f"\nLIVE: {args.arm} goes to {', '.join(p.square for p in plans)} one at a time: "
              f"{args.hover * 100:.0f} cm above, down to {args.z * 1000:.0f} mm, gripper closed, tilt "
              f"{args.pitch:g} deg, speed {args.speed}; holds {args.settle:g} s and measures{then}; "
              "comes back.")
        print(f"log: {log_path}\nCtrl+C holds the arm. Keep a hand near the power.")
        if prompt("Type 'yes' to start: ").strip().lower() != "yes":
            print("aborted")
            return 1
        from magnus_bot.hardware.lerobot_arm import LeRobotArm

        spec = config.arms.arms[args.arm]
        arm = LeRobotArm(arm_id=args.arm, port=spec.port, calibration_id=spec.calibration_id)
    else:
        if args.sim_sag_deg:
            arm = SaggingSimArm(args.arm, math.radians(args.sim_sag_deg))
        else:
            arm = SimArm(args.arm)
        reader = SimCamera(arm, placement, kin, to_model, size, camera_offset, sim_parallax,
                           correction=correction)

    arms = {args.arm: arm}
    reports = []
    code = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        try:
            arm.connect()
            rest = arm.read_joints()
            def look():
                """Where the camera sees the fingertip now (board squares), or None."""
                try:
                    return reader.read(fresh=True, timeout=args.vision_timeout).marker
                except VisionUnavailable as e:
                    print(f"  vision: {e}", file=sys.stderr)
                    return None

            for plan in plans:
                stream(arm, plan.down(rest, args.speed, HZ), args.live)
                if args.live:
                    time.sleep(args.settle)
                reached = median_joints(arm)
                marker = look()
                gap = None
                if args.ask_height:  # the arm holds here while the operator measures
                    answer = prompt(f"  {plan.square}: gap from the green tip to the board, mm "
                                    f"(commanded {args.z * 1000:.0f}; Enter to skip): ").strip()
                    try:
                        gap = float(answer) if answer else None
                    except ValueError:
                        print(f"  not a number ({answer!r}); skipping the height here", file=sys.stderr)
                marker2 = None
                if plan.at2 is not None:
                    stream(arm, plan.raise_to_z2(args.speed, HZ), args.live)
                    if args.live:
                        time.sleep(args.settle)
                    marker2 = look()
                report = analyze(plan, reached, placement=placement, kinematics=kin, to_model=to_model,
                                 camera_xy=None if marker is None else np.array(marker) * size,
                                 camera_xy2=None if marker2 is None else np.array(marker2) * size,
                                 measured_gap_mm=gap)
                reports.append(report)
                print("\n".join(report.lines()))
                fh.write(json.dumps({
                    "event": "probe", "time": time.time(), "arm": args.arm, "live": args.live,
                    "pitch_deg": args.pitch, "z_m": args.z, "z2_m": z2, "square_size_m": size,
                    "edge_gap_m": config.arms.edge_gap[args.arm],
                    "correction_m": [correction.reach, correction.side, correction.z],
                    "commanded": plan.at, "reached": reached, "camera_squares": marker,
                    "camera_squares_z2": marker2,
                    **report.to_record(),
                }) + "\n")
                fh.flush()
                stream(arm, plan.up(rest, args.speed, HZ), args.live)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            code = 130
        except Exception as e:  # any fault: report it; shut_down below holds the arm
            print(f"stopped, arm holding: {type(e).__name__}: {e}", file=sys.stderr)
            code = 3
        finally:
            shut_down(arms, live=args.live, prompt=prompt)

    print("\n" + "\n".join(verdict(reports)))
    print(f"log: {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

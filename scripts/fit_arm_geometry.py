"""Fit an arm's real geometry to the reach-probe logs: base gap + reach / side landing corrections.

Each live probe (scripts/reach_probe.py) logged the joint angles the arm actually reached and where
the camera saw the green fingertip at two heights. From the joints, the model says where the tip
should be; the camera says where it is. This finds the edge_gap, reach_correction and
side_correction (configs/arms.yaml) that best explain every probe at once, and shows how far each
square is off before (current config) and after (the fit).

Why a fit and not one offset: on 2026-10-24 follower_1 landed 7-20 mm to its right on squares
toward rank 1 and slightly left toward rank 8. One offset cannot do that; a base that sits closer
to the board than the config says (so every pan angle is too small) plus an arm that falls short
along its reach does, to ~3 mm.

    <core-venv>/python scripts/fit_arm_geometry.py                    # every follower_1 probe log
    <core-venv>/python scripts/fit_arm_geometry.py hardware/tuning_logs/probe_follower_1_2026*.jsonl

Probes without the second (parallax) height are skipped: their camera position is off by ~5-9 mm.
Read-only: prints the YAML to paste into configs/arms.yaml; nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from magnus_bot.config import load_config
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.geometry.transforms import unit
from magnus_bot.kinematics.conventions import load_servo_map, urdf_to_model
from magnus_bot.kinematics.so101_ik import SO101Kinematics
from magnus_bot.orchestration.probe import (
    ProbeSample,
    fit_geometry,
    predict_seen,
    remove_parallax,
    split_mm,
)

ROOT = Path(__file__).resolve().parents[1]


def load_samples(paths: list[Path], arm_id: str, square_size: float, to_model) -> list[ProbeSample]:
    """Live probes of ``arm_id`` that saw the tip at both heights, oldest first."""
    kin = SO101Kinematics()
    samples = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("event") != "probe" or not rec.get("live") or rec.get("arm") != arm_id
                    or not rec.get("camera_squares") or not rec.get("camera_squares_z2")):
                continue
            tip = kin.forward(to_model(rec["reached"]))
            low = np.array(rec["camera_squares"]) * square_size
            high = np.array(rec["camera_squares_z2"]) * square_size
            seen = remove_parallax(low, high, rec["z_m"], rec["z2_m"])
            samples.append(ProbeSample(f"{rec['square']}@{path.stem[-6:]}", tip[:2], seen))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit edge_gap + reach/side corrections to probe logs.")
    parser.add_argument("logs", nargs="*", type=Path,
                        help="probe JSONL files (default: every hardware/tuning_logs/probe_<arm>_*.jsonl)")
    parser.add_argument("--arm", default="follower_1")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    size = config.board.square_size
    paths = args.logs or sorted((ROOT / "hardware" / "tuning_logs").glob(f"probe_{args.arm}_*.jsonl"))
    paths = [p for p in paths if not p.stem.endswith("_sim")]
    smap = load_servo_map(ROOT / "hardware" / "servo_maps" / f"{args.arm}.json")
    samples = load_samples(paths, args.arm, size, lambda j: urdf_to_model(smap.servo_to_urdf(j)))
    print(f"{len(samples)} usable probes (live, tip seen at both heights) from {len(paths)} log(s)")
    try:
        fit = fit_geometry([ProbeSample(s.square.split("@")[0], s.model_tip_xy, s.seen_xy) for s in samples],
                           args.arm, size, config.arms.edge_gap[args.arm])
    except ValueError as e:
        print(f"cannot fit: {e}", file=sys.stderr)
        return 1

    now_gap, now = config.arms.edge_gap[args.arm], config.arms.correction(args.arm)
    placement = arm_placements(size, now_gap)[args.arm]
    print(f"\n{'square':14} {'current config':>22}   {'after the fit':>20}   (mm: reach, side; + = too far / "
          "to the arm's left)")
    for s, after in zip(samples, fit.residuals_mm, strict=True):
        miss = s.seen_xy - predict_seen(s, args.arm, size, now_gap, now)
        before = split_mm(miss, unit(s.seen_xy - placement.base_xy))
        print(f"{s.square:14} {before[0]:+9.1f} {before[1]:+9.1f}      {after[0]:+9.1f} {after[1]:+9.1f}")
    before_rms = 1000 * float(np.sqrt(np.mean(
        [np.sum((s.seen_xy - predict_seen(s, args.arm, size, now_gap, now)) ** 2) for s in samples])))
    print(f"\nrms miss: current {before_rms:.1f} mm -> fitted {fit.rms_mm:.1f} mm")
    print("\nPaste into configs/arms.yaml (keep z_correction: it comes from the ruler):")
    print(f"  edge_gap:          {args.arm}: {fit.edge_gap:.3f}   (now {now_gap:.3f})")
    print(f"  reach_correction:  {args.arm}: {fit.correction.reach:.3f}   (now {now.reach:.3f})")
    print(f"  side_correction:   {args.arm}: {fit.correction.side:.3f}   (now {now.side:.3f})")
    print(f"  lateral_correction: {args.arm}: {fit.correction.lateral:.3f}   (now {now.lateral:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""scripts/fit_arm_geometry.py: recover the rig's geometry from (synthetic) live probe logs."""

import importlib.util
import json
from pathlib import Path

from magnus_bot.config import ArmCorrection
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.kinematics.conventions import load_servo_map, model_to_urdf, urdf_to_model
from magnus_bot.kinematics.so101_ik import SO101Kinematics
from magnus_bot.orchestration.probe import plan_probe

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_arm_geometry.py"
SIZE = 0.056


def load():
    spec = importlib.util.spec_from_file_location("fit_arm_geometry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_probes(path, gap, fix, squares=("g3", "g4", "g5", "g6", "h4", "f4")):
    """Live-style probe records of a rig whose base sits ``gap`` out and lands ``fix`` off its model."""
    kin = SO101Kinematics()
    smap = load_servo_map(ROOT / "hardware" / "servo_maps" / "follower_1.json")
    model_base = arm_placements(SIZE, 0.10)["follower_1"]  # whatever the config said at probe time
    truth = arm_placements(SIZE, gap)["follower_1"]
    lines = []
    for sq in squares:
        p = plan_probe(sq, model_base, kin, model_to_urdf, kin.model.limits, square_size=SIZE,
                       z=0.010, hover=0.06, pitch=0.0, gripper=4.0)
        model = urdf_to_model(p.at)
        tip = kin.forward(model)
        seen = truth.arm_to_board(fix.undo(tip)[:2]) / SIZE  # camera: board squares, no parallax
        lines.append(json.dumps({
            "event": "probe", "live": True, "arm": "follower_1", "square": sq, "z_m": 0.010, "z2_m": 0.040,
            "reached": smap.model_to_servo(model), "camera_squares": list(seen),
            "camera_squares_z2": list(seen),
        }))
    path.write_text("\n".join(lines) + "\n")


def test_fit_prints_the_geometry_that_explains_the_probes(tmp_path, capsys):
    log = tmp_path / "probe_follower_1_20261024-000000.jsonl"
    write_probes(log, 0.080, ArmCorrection(reach=0.030, side=0.004))
    assert load().main([str(log)]) == 0
    out = capsys.readouterr().out
    assert "6 usable probes" in out
    assert "edge_gap:          follower_1: 0.080" in out
    assert "reach_correction:  follower_1: 0.030" in out
    assert "side_correction:   follower_1: 0.004" in out
    assert "fitted 0.0 mm" in out


def test_too_few_probes_is_refused(tmp_path, capsys):
    log = tmp_path / "probe_follower_1_20261024-000001.jsonl"
    write_probes(log, 0.080, ArmCorrection(), squares=("g4", "h4"))
    assert load().main([str(log)]) == 1
    assert "cannot fit" in capsys.readouterr().err

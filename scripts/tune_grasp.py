"""Closed-loop grasp tuning: sweep tilt and grasp height, let the camera grade every attempt.

The arm shuttles one piece between two squares (default g3 <-> g4 with follower_1) at each
setting of the sweep. After every attempt RF-DETR/vision_server.py reports the board, the attempt
is graded (success / not_picked / lost / misplaced / disturbed / unclear) and logged as one JSON
line. Misses retry on their own; anything else holds the arm and waits for you to fix the board.
A summary table and the best setting print at the end.

Two ways to choose what to try:
  --pitches 25,30,35 [--z-offsets 0,5]     a fixed grid
  --search --pitch-range 20,40 [--z-range=-5,10]   coarse to fine: 5 spread-out values, then keep
                                           halving the step around the winner (tilt, then height)
The winner is the most robust top scorer (farthest from any failing setting). With --save it is
confirmed with --confirm fresh attempts and, if at least --min-success of those pass, written to
configs/grasp.yaml, which first_move.py then uses by default.

  1. camera side (RF-DETR venv):    RF-DETR/.venv/Scripts/python.exe RF-DETR/vision_server.py --show
  2. check the link (lerobot venv): <lerobot-venv>/python scripts/tune_grasp.py --check-vision
  3. sim pass (any venv, no camera): <core-venv>/python scripts/tune_grasp.py --search
  4. live (operator present):       <lerobot-venv>/python scripts/tune_grasp.py --live --search --save

Live needs board.yaml `measured: true`, the arm's servo map, --live, and a typed "yes". Start the
arm from a safe pose near the board whose links do not hide the two squares from the camera, put
the piece on the first square of --uci and leave the second empty. Only the chosen arm goes in
the reach map, so the planner cannot hand the move to the other arm at steep tilts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import chess

from magnus_bot.config import TunedGrasp, load_config, save_tuned_grasp
from magnus_bot.geometry.frames import arm_placements
from magnus_bot.hardware.sim_arm import SimArm
from magnus_bot.kinematics.conventions import load_servo_map, real_urdf_limits
from magnus_bot.kinematics.reach import compute_reachability
from magnus_bot.kinematics.so101_ik import SO101Kinematics, Unreachable
from magnus_bot.orchestration.bench import board_diagram, shut_down
from magnus_bot.orchestration.executor import (
    ArmSteps,
    ExecutionFault,
    Executor,
    JointLimitError,
    build_move_program,
)
from magnus_bot.orchestration.tuning import (
    Setting,
    SettingInfeasible,
    SettingResult,
    TuningStopped,
    best_setting,
    refine_search,
    run_sweep,
    summary_lines,
)
from magnus_bot.vision.observation import (
    DEFAULT_VISION_URL,
    BoardReader,
    HttpBoardReader,
    VisionUnavailable,
)
from magnus_bot.vision.sim_board import SimulatedBoard

ROOT = Path(__file__).resolve().parents[1]
MIN_FINGERTIP_Z = 0.003  # m; board.yaml floors grasp heights here so the jaw tips never scrape
PIECES = {
    "pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
    "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING,
}


class ArmShuttle:
    """Moves the tuning piece with an Executor (sim or live); ``world`` mirrors it in sim.

    ``sim_miss`` (sim only) makes every grasp at this setting miss, to model a bad setting.
    """

    def __init__(
        self,
        executor: Executor,
        piece: chess.Piece,
        world: SimulatedBoard | None = None,
        sim_miss: bool = False,
    ) -> None:
        self.executor = executor
        self.piece = piece
        self.world = world
        self.sim_miss = sim_miss

    def program(self, src: str, dst: str) -> list[ArmSteps]:
        board = chess.Board(None)
        board.set_piece_at(chess.parse_square(src), self.piece)
        move = chess.Move(chess.parse_square(src), chess.parse_square(dst))
        return build_move_program(board, move, self.executor.rmap, self.executor.config)

    def transfer(self, src: str, dst: str) -> None:
        self.executor.run(self.executor.plan(self.program(src, dst)))
        if self.world is not None and not self.sim_miss:
            self.world.move(src, dst)


def check_vision(reader: HttpBoardReader, timeout: float) -> int:
    """Print one fresh board from the server: proves the link from this venv."""
    try:
        health = reader.health()
        t0 = time.monotonic()
        obs = reader.read(fresh=True, timeout=timeout)
    except VisionUnavailable as e:
        print(f"vision: {e}", file=sys.stderr)
        return 1
    print(f"vision server OK ({health.get('fps', 0):.0f} fps, camera_ok={health.get('camera_ok')}); "
          f"fresh stable board in {time.monotonic() - t0:.2f} s, frame {obs.frame_id}")
    print(board_diagram(obs.occupancy))
    for sq, name in sorted(obs.pieces.items()):
        print(f"  {sq}: {name}")
    return 0


def _floats(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"expected comma-separated numbers, got {text!r}") from None


def _range(text: str) -> tuple[float, float]:
    values = _floats(text)
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"expected a range 'low,high', got {text!r}")
    return values[0], values[1]


def main(argv: list[str] | None = None, prompt=input) -> int:
    parser = argparse.ArgumentParser(description="Closed-loop grasp tuning, graded by the camera.")
    parser.add_argument("--uci", default="g3g4", help="the two squares; the piece starts on the first")
    parser.add_argument("--piece", default="rook", choices=sorted(PIECES))
    parser.add_argument("--arm", default="follower_1")
    parser.add_argument("--pitches", default="25,30,35", help="grid: tilts to try, degrees from vertical")
    parser.add_argument("--z-offsets", default="0",
                        help="grid: grasp-height offsets to try, mm (+ = higher; write --z-offsets=-5,0)")
    parser.add_argument("--search", action="store_true", help="search coarse to fine instead of a grid")
    parser.add_argument("--pitch-range", default="20,40", help="search: tilt range, degrees")
    parser.add_argument("--z-range", default=None,
                        help="search: also search grasp height over this range, mm (e.g. --z-range=-5,10)")
    parser.add_argument("--coarse-points", type=int, default=5, help="search: values in the first pass")
    parser.add_argument("--min-pitch-step", type=float, default=1.0, help="search: finest tilt step, degrees")
    parser.add_argument("--min-z-step", type=float, default=1.0, help="search: finest height step, mm")
    parser.add_argument("--save", action="store_true",
                        help="confirm the winner, then write it to configs/grasp.yaml if it holds up")
    parser.add_argument("--confirm", type=int, default=10, help="fresh attempts to confirm the winner")
    parser.add_argument("--min-success", type=float, default=0.9,
                        help="confirmation success rate needed to save (0..1)")
    parser.add_argument("--grip-depth-mm", type=float, default=None,
                        help="hold the stem this far back from the jaw tips (default: the pads, 23 mm); "
                             "less = nearer the tips, which saves reach on far squares")
    parser.add_argument("--trials", type=int, default=6, help="graded attempts per setting")
    parser.add_argument("--hover", type=float, default=0.06, help="hover height, m (see first_move.py)")
    parser.add_argument("--no-shuttle-back", action="store_true",
                        help="only attempt --uci forward; you put the piece back after each success")
    parser.add_argument("--max-misses", type=int, default=3,
                        help="misses in a row before a setting is given up as bad")
    parser.add_argument("--max-failed-settings", type=int, default=3,
                        help="given-up settings in a row (no success between) before asking you")
    parser.add_argument("--live", action="store_true", help="drive the real arm (gated)")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--vision-timeout", type=float, default=8.0,
                        help="seconds to wait for a stable board")
    parser.add_argument("--check-vision", action="store_true",
                        help="print one board from the server and exit")
    parser.add_argument("--sim-miss-rate", type=float, default=0.0, help="sim only: chance a grasp misses")
    parser.add_argument("--sim-good-pitch", default=None,
                        help="sim only: grasps work only for tilts in this range, e.g. 27,34 (models tuning)")
    parser.add_argument("--log", type=Path, default=None,
                        help="JSONL path (default hardware/tuning_logs/...)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    if args.check_vision:
        return check_vision(HttpBoardReader(args.vision_url), args.vision_timeout)

    try:
        settings = [Setting(p, z) for p in _floats(args.pitches) for z in _floats(args.z_offsets)]
        pitch_range = _range(args.pitch_range)
        z_range = _range(args.z_range) if args.z_range else None
        good_pitch = _range(args.sim_good_pitch) if args.sim_good_pitch else None
        mv = chess.Move.from_uci(args.uci)
    except ValueError as e:
        parser.error(str(e))
    if args.save and not args.live and args.config_dir is None:
        parser.error("refusing to save SIMULATED results into configs/; "
                     "use --live, or --config-dir elsewhere")
    home, away = chess.square_name(mv.from_square), chess.square_name(mv.to_square)

    config = load_config(args.config_dir)
    if args.arm not in config.arms.arms:
        parser.error(f"unknown arm {args.arm!r}; configured: {sorted(config.arms.arms)}")
    base_z = config.board.grasp_height.get(args.piece, config.arms.z_grasp)
    grip_depth = None if args.grip_depth_mm is None else args.grip_depth_mm / 1000.0
    kin = SO101Kinematics()
    placements = {args.arm: arm_placements(config.board.square_size, config.arms.edge_gap)[args.arm]}
    piece = chess.Piece(PIECES[args.piece], chess.WHITE)
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("" if args.live else "_sim")
    log_path = args.log or ROOT / "hardware" / "tuning_logs" / f"{args.uci}_{stamp}.jsonl"

    # Plan inside what the servos can really reach (their calibrated ranges), live and in sim alike.
    spec = config.arms.arms[args.arm]
    servo_map = ROOT / "hardware" / "servo_maps" / f"{args.arm}.json"
    smap = load_servo_map(servo_map) if servo_map.is_file() else None
    real_limits = None if smap is None else real_urdf_limits(smap, kin.model.limits, spec.calibration_id)
    urdf_limits = real_limits or kin.model.limits
    print(f"joint limits: {'calibrated servo ranges' if real_limits else 'URDF (no calibration found)'}")

    world: SimulatedBoard | None = None
    reader: BoardReader
    if args.live:
        try:
            config.require_measured()  # also enforced by Executor; fail before touching hardware
            if smap is None or real_limits is None:
                raise RuntimeError(f"{args.arm} needs {servo_map} and its LeRobot calibration "
                                   f"({spec.calibration_id}.json) to know its real joint ranges")
        except RuntimeError as e:
            print(f"live refused: {e}", file=sys.stderr)
            return 1
        reader = HttpBoardReader(args.vision_url)
        try:
            obs = reader.read(fresh=True, timeout=args.vision_timeout)
        except VisionUnavailable as e:
            print(f"vision: {e}", file=sys.stderr)
            return 1
        print(board_diagram(obs.occupancy))
        if home not in obs.occupancy or away in obs.occupancy:
            print(f"put the {args.piece} on {home} and clear {away} first", file=sys.stderr)
            return 1
        if args.search:
            plan = f"searching tilt {pitch_range[0]:g}..{pitch_range[1]:g} deg"
            if z_range:
                plan += f", then grasp height {z_range[0]:+g}..{z_range[1]:+g} mm"
        else:
            plan = "trying " + "; ".join(s.label() for s in settings)
        print(f"\nLIVE: {args.arm} shuttles the {args.piece} {home}<->{away}, up to {args.trials} graded "
              f"attempts per setting, {plan}.")
        if args.save:
            print(f"Then {args.confirm} fresh attempts at the winner; saved to "
                  f"{config.source_dir / 'grasp.yaml'} only if >= {args.min_success:.0%} pass.")
        print(f"log: {log_path}\nCtrl+C holds the arm. Keep a hand near the power.")
        if prompt("Type 'yes' to start: ").strip().lower() != "yes":
            print("aborted")
            return 1

        from magnus_bot.hardware.lerobot_arm import LeRobotArm

        arms = {args.arm: LeRobotArm(arm_id=args.arm, port=spec.port, calibration_id=spec.calibration_id)}
        exec_kwargs = dict(live=True, to_command=smap.model_to_servo,
                           limits=smap.servo_limits(urdf_limits), kinematics=kin)
    else:
        arms = {a: SimArm(a) for a in config.arms.arms}
        world = SimulatedBoard({home: "W"}, miss_rate=args.sim_miss_rate)
        reader = world
        exec_kwargs = dict(simulate=True, kinematics=kin, limits=urdf_limits)  # sim commands URDF angles

    def make_shuttle(setting: Setting) -> ArmShuttle:
        pitch = math.radians(setting.pitch_deg)
        z_grasp = base_z + setting.z_offset_mm / 1000.0
        # The grip point is tip_to_pad above the top-down tip; the tip sits grip_depth down the tool
        # from it (tilted_tip_target). Level and shallow grips lift the tip, so the pads can go lower.
        depth = kin.model.tip_to_pad if grip_depth is None else grip_depth
        tip_z = z_grasp + kin.model.tip_to_pad - depth * math.cos(pitch)
        if tip_z < MIN_FINGERTIP_Z:
            raise SettingInfeasible(f"fingertip would be {tip_z * 1000:.1f} mm above the board "
                                    f"(< {MIN_FINGERTIP_Z * 1000:.0f} mm: the jaw tips would scrape)")
        cfg = dataclasses.replace(config, arms=dataclasses.replace(
            config.arms, z_grasp=z_grasp, z_clear=args.hover))
        rmap = compute_reachability(placements, kin, cfg.arms.z_grasp, cfg.board.square_size,
                                    z_clear=cfg.arms.z_clear, limits=urdf_limits, max_pitch=pitch,
                                    grip_depth=grip_depth, corrections=cfg.arms.corrections())
        for sq in (home, away):
            sq_index = chess.parse_square(sq)
            if not rmap.can_reach(args.arm, chess.square_file(sq_index), chess.square_rank(sq_index)):
                raise SettingInfeasible(f"{args.arm} cannot reach {sq}")
        sim_miss = good_pitch is not None and not good_pitch[0] <= setting.pitch_deg <= good_pitch[1]
        shuttle = ArmShuttle(Executor(cfg, arms, rmap, pitch=pitch, grip_depth=grip_depth, **exec_kwargs),
                             piece, world, sim_miss)
        try:  # plan both directions now, so a too-steep tilt fails here and not mid-sweep
            for src, dst in ((home, away), (away, home)):
                shuttle.executor.plan(shuttle.program(src, dst))
        except (JointLimitError, Unreachable) as e:
            raise SettingInfeasible(f"cannot plan: {e}") from e
        return shuttle

    results: list[SettingResult] = []
    confirmation: list[SettingResult] = []
    winner: SettingResult | None = None
    scale = (args.min_pitch_step, args.min_z_step)
    code = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:

        def log(record: dict) -> None:
            fh.write(json.dumps({"arm": args.arm, "piece": args.piece, "live": args.live, **record}) + "\n")
            fh.flush()

        common = dict(
            log=log, prompt=prompt if args.live else None, shuttle_back=not args.no_shuttle_back,
            max_misses=args.max_misses, max_failed_settings=args.max_failed_settings,
            vision_timeout=args.vision_timeout,
        )
        try:
            for arm in arms.values():
                arm.connect()
            if args.search:
                winner = refine_search(
                    home, away, args.trials, make_shuttle, reader, pitch_range=pitch_range,
                    z_range=z_range, coarse_points=args.coarse_points, min_step=scale,
                    results=results, **common,
                )
            else:
                run_sweep(settings, home, away, args.trials, make_shuttle, reader, results=results, **common)
                winner = best_setting(results, scale)
            if args.save and winner is not None and winner.successes:
                log({"event": "confirm", "time": time.time(), "pitch_deg": winner.setting.pitch_deg,
                     "z_offset_mm": winner.setting.z_offset_mm, "attempts": args.confirm})
                run_sweep([winner.setting], home, away, args.confirm, make_shuttle, reader,
                          results=confirmation, **common)
        except TuningStopped as e:
            print(f"stopped: {e}", file=sys.stderr)
            code = 2
        except ExecutionFault as e:
            print(f"MOTION FAULT, arm holding: {e}", file=sys.stderr)
            code = 3
        except VisionUnavailable as e:
            print(f"vision: {e} (arm at rest, holding)", file=sys.stderr)
            code = 4
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            code = 130
        finally:
            shut_down(arms, live=args.live, prompt=prompt)

    print("\n" + "\n".join(summary_lines(results, scale, best=winner)))
    if args.save and code == 0:
        code = _save_winner(args, config.source_dir, winner, confirmation[0] if confirmation else None)
    print(f"log: {log_path}")
    return code


def _save_winner(args, config_dir: Path, winner: SettingResult | None, check: SettingResult | None) -> int:
    """Write the winner to grasp.yaml if its fresh confirmation attempts held up; 0 if saved, else 5."""
    if winner is None or not winner.successes:
        print("not saved: no setting ever succeeded", file=sys.stderr)
        return 5
    label = winner.setting.label()
    if check is None or not check.attempts or check.success_rate < args.min_success:
        got = f"{check.successes}/{check.attempts}" if check is not None else "0/0"
        print(f"not saved: {label} passed only {got} confirmation attempts "
              f"(need {args.min_success:.0%}); widen the search or add --trials", file=sys.stderr)
        return 5
    confirmed = f"{check.successes}/{check.attempts}"
    path = save_tuned_grasp(config_dir, args.piece, TunedGrasp(
        pitch_deg=winner.setting.pitch_deg, z_offset_mm=winner.setting.z_offset_mm,
        grip_depth_mm=args.grip_depth_mm,
        confirmed=confirmed, tuned_on=f"{args.arm} {args.uci}", date=time.strftime("%Y-%m-%d"),
    ))
    print(f"saved {args.piece}: {label}, confirmed {confirmed} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Typed configuration loaded from ``configs/*.yaml``.

All distances are meters, angles radians. Every geometric value in the shipped configs is a
PLACEHOLDER until the robot is measured; the top-level ``measured`` flag (in board.yaml) gates
live operation. Sim mode runs with placeholders and warns; live mode refuses while
``measured`` is false. See CLAUDE.md "Safety rules".

Never read a test's expected value from these files: tests construct their own geometry.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

PIECE_TYPES = ("pawn", "knight", "bishop", "rook", "queen", "king")

GRASP_FILE = "grasp.yaml"  # optional, machine-written by scripts/tune_grasp.py --save
GRASP_HEADER = """\
# Grasp settings found by closed-loop tuning. Written by `scripts/tune_grasp.py --save`.
# pitch_deg: tool tilt from vertical. z_offset_mm: added to board.yaml grasp_height[piece].
# confirmed: fresh trials that passed at this setting before it was saved.
# Read by Config.grasp_for (first_move.py uses it). Delete a piece's entry to go back to untuned.
"""


@dataclass(frozen=True)
class BoardConfig:
    square_size: float
    dark_a1: bool
    human_color: str  # "black" (default) or "white"; robots play the other color and move first.
    piece_heights: dict[str, float]
    piece_base_diameters: dict[str, float]  # base = widest point = collision-cylinder diameter
    piece_stem_diameters: dict[str, float] = field(default_factory=dict)  # grippable neck width
    grasp_height: dict[str, float] = field(default_factory=dict)  # per-piece fingertip target (m)

    @property
    def robot_color(self) -> str:
        return "white" if self.human_color == "black" else "black"


@dataclass(frozen=True)
class ArmConfig:
    arm_id: str
    port: str
    calibration_id: str


@dataclass(frozen=True)
class ArmsConfig:
    edge_gap: dict[str, float]  # d per arm id: base yaw axis -> near board edge
    speed_scale: float  # default trajectory speed scale (0..1)
    z_clear: float  # traverse height, > tallest piece + margin
    z_grasp: float  # gripper height when grasping a piece on the board
    gripper_open: float  # gripper percent (0..100) fully open
    gripper_closed: float  # gripper percent when holding a piece
    arms: dict[str, ArmConfig]
    # Per arm, added to every commanded fingertip height: the arm sits this much LOWER than the
    # model says (measured by ruler with reach_probe.py --ask-height). Missing = 0.
    z_correction: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Pose:
    """A grasp pose in frame B (meters). ``yaw`` optional (radians); None = top-down default."""

    x: float
    y: float
    z: float
    yaw: float | None = None


@dataclass(frozen=True)
class HandoffConfig:
    pad: Pose  # off-board handoff pad where one arm places and the other picks
    bins: dict[str, Pose]  # captured-piece bin per arm id
    promotion_stash: Pose  # where spare queens wait for a promotion


@dataclass(frozen=True)
class TunedGrasp:
    """A grasp found by closed-loop tuning for one piece type."""

    pitch_deg: float
    z_offset_mm: float = 0.0
    grip_depth_mm: float | None = None  # stem this far back from the jaw tips; None = at the pads
    confirmed: str = ""  # e.g. "10/10"
    tuned_on: str = ""  # e.g. "follower_1 g3g4"
    date: str = ""


@dataclass(frozen=True)
class Config:
    measured: bool
    board: BoardConfig
    arms: ArmsConfig
    handoff: HandoffConfig
    source_dir: Path = field(default=DEFAULT_CONFIG_DIR)
    grasp_tuning: dict[str, TunedGrasp] = field(default_factory=dict)  # from grasp.yaml

    def grasp_for(self, piece: str, default_pitch: float = 0.0) -> tuple[float, float]:
        """``(pitch rad, fingertip z m)`` for ``piece``: its tuned grasp if one was saved, else the
        measured ``grasp_height`` with ``default_pitch``."""
        z = self.board.grasp_height.get(piece, self.arms.z_grasp)
        tuned = self.grasp_tuning.get(piece)
        if tuned is None:
            return default_pitch, z
        return math.radians(tuned.pitch_deg), z + tuned.z_offset_mm / 1000.0

    def grip_depth_for(self, piece: str) -> float | None:
        """Where ``piece``'s stem sits in the jaws (m back from the tips), or None for the pads."""
        tuned = self.grasp_tuning.get(piece)
        return None if tuned is None or tuned.grip_depth_mm is None else tuned.grip_depth_mm / 1000.0

    def require_measured(self) -> None:
        """Raise unless the rig has been measured. Call before any live motion."""
        if not self.measured:
            raise RuntimeError(
                "config.measured is false: geometry values are placeholders. "
                "Measure the rig and set measured: true in configs/board.yaml before live mode."
            )

    def warn_if_unmeasured(self) -> None:
        """Emit a warning (used by sim mode) when running on placeholder geometry."""
        if not self.measured:
            warnings.warn(
                "Running on PLACEHOLDER geometry (config.measured is false). Sim only.",
                stacklevel=2,
            )


def load_config(config_dir: str | Path | None = None) -> Config:
    """Load and validate board/arms/handoff YAML from ``config_dir``."""
    config_dir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    board_raw = _load_yaml(config_dir / "board.yaml")
    arms_raw = _load_yaml(config_dir / "arms.yaml")
    handoff_raw = _load_yaml(config_dir / "handoff.yaml")

    human_color = str(board_raw.get("human_color", "black")).lower()
    if human_color not in ("black", "white"):
        raise ValueError(f"board.human_color must be 'black' or 'white', got {human_color!r}")

    board = BoardConfig(
        square_size=_float(board_raw, "square_size"),
        dark_a1=bool(board_raw.get("dark_a1", True)),
        human_color=human_color,
        piece_heights=_float_map(board_raw, "piece_heights"),
        piece_base_diameters=_float_map(board_raw, "piece_base_diameters"),
        piece_stem_diameters=_float_map(board_raw, "piece_stem_diameters"),
        grasp_height=_float_map(board_raw, "grasp_height"),
    )

    arms = ArmsConfig(
        edge_gap=_edge_gap(arms_raw, arms_raw["arms"]),
        speed_scale=_float(arms_raw, "speed_scale"),
        z_clear=_float(arms_raw, "z_clear"),
        z_grasp=_float(arms_raw, "z_grasp"),
        gripper_open=_float(arms_raw, "gripper_open"),
        gripper_closed=_float(arms_raw, "gripper_closed"),
        arms={
            arm_id: ArmConfig(
                arm_id=arm_id,
                port=str(spec["port"]),
                calibration_id=str(spec.get("calibration_id", arm_id)),
            )
            for arm_id, spec in arms_raw["arms"].items()
        },
        z_correction=_float_map(arms_raw, "z_correction"),
    )

    handoff = HandoffConfig(
        pad=_pose(handoff_raw["pad"]),
        bins={arm_id: _pose(p) for arm_id, p in handoff_raw.get("bins", {}).items()},
        promotion_stash=_pose(handoff_raw["promotion_stash"]),
    )

    return Config(
        measured=bool(board_raw.get("measured", False)),
        board=board,
        arms=arms,
        handoff=handoff,
        source_dir=config_dir,
        grasp_tuning=_load_grasp_tuning(config_dir / GRASP_FILE),
    )


def save_tuned_grasp(config_dir: str | Path, piece: str, tuned: TunedGrasp) -> Path:
    """Write ``tuned`` as ``piece``'s entry in ``config_dir/grasp.yaml`` (other pieces kept)."""
    if piece not in PIECE_TYPES:
        raise ValueError(f"unknown piece {piece!r}; expected one of {PIECE_TYPES}")
    path = Path(config_dir) / GRASP_FILE
    data = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    data[piece] = asdict(tuned)
    path.write_text(GRASP_HEADER + yaml.safe_dump(data, sort_keys=True))
    return path


# --- helpers ---------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _float(d: dict[str, Any], key: str) -> float:
    if key not in d:
        raise KeyError(f"missing required key: {key}")
    return float(d[key])


def _float_map(d: dict[str, Any], key: str) -> dict[str, float]:
    raw = d.get(key, {}) or {}
    return {str(k): float(v) for k, v in raw.items()}


def _edge_gap(arms_raw: dict[str, Any], arm_specs: dict[str, Any]) -> dict[str, float]:
    """Parse edge_gap as a per-arm map, or a scalar applied to every arm."""
    raw = arms_raw["edge_gap"]
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    gap = float(raw)
    return {arm_id: gap for arm_id in arm_specs}


def _load_grasp_tuning(path: Path) -> dict[str, TunedGrasp]:
    """Tuned grasps from the optional grasp.yaml; missing or empty file -> no tuning."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    tuned = {}
    for piece, entry in data.items():
        if piece not in PIECE_TYPES:
            raise ValueError(f"{path}: unknown piece {piece!r}; expected one of {PIECE_TYPES}")
        tuned[piece] = TunedGrasp(
            pitch_deg=_float(entry, "pitch_deg"),
            z_offset_mm=float(entry.get("z_offset_mm", 0.0)),
            grip_depth_mm=None if entry.get("grip_depth_mm") is None else float(entry["grip_depth_mm"]),
            confirmed=str(entry.get("confirmed", "")),
            tuned_on=str(entry.get("tuned_on", "")),
            date=str(entry.get("date", "")),
        )
    return tuned


def _pose(d: dict[str, Any]) -> Pose:
    return Pose(
        x=float(d["x"]),
        y=float(d["y"]),
        z=float(d["z"]),
        yaw=float(d["yaw"]) if d.get("yaw") is not None else None,
    )

"""Shared fixtures: a stand-in vision server, and config dirs with a pinned ``measured`` flag."""

import json
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

BOARD_JSON = {
    "frame_id": 42,
    "timestamp": 1.5,
    "stable": True,
    "stable_frames": 5,
    "pieces": {"g3": "white-rook"},
    "occupancy": {"g3": "W"},
}


@pytest.fixture
def vision_server():
    """Stand-in for RF-DETR/vision_server.py: answers every GET with ``.reply`` (status, body)
    and records request paths in ``.paths``. The default reply is :data:`BOARD_JSON`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            server.paths.append(self.path)
            status, body = server.reply
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.paths = []
    server.reply = (200, BOARD_JSON)
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def config_dir(tmp_path):
    """Factory: a copy of configs/ with ``measured`` pinned, independent of the local checkout
    (the operator's working copy may carry ``measured: true``; the repo ships ``false``)."""

    def make(measured: bool) -> Path:
        out = tmp_path / f"configs-measured-{measured}"
        shutil.copytree(CONFIGS, out)
        board = out / "board.yaml"
        text, n = re.subn(r"(?m)^measured:\s*\w+", f"measured: {str(measured).lower()}", board.read_text())
        assert n == 1, "board.yaml should have exactly one top-level measured: key"
        board.write_text(text)
        return out

    return make


@pytest.fixture(autouse=True)
def _no_machine_calibration(tmp_path_factory, monkeypatch):
    """Keep tests off this machine's LeRobot calibration files: plan with URDF limits unless a test
    points MAGNUS_CALIBRATION_DIR at its own calibration."""
    monkeypatch.setenv("MAGNUS_CALIBRATION_DIR", str(tmp_path_factory.mktemp("no_calibration")))


# follower_1's LeRobot calibration as of 2026-10-25 (elbow range +/-96.3 deg).
F1_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1992, "range_min": 747, "range_max": 3474},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1148, "range_min": 841, "range_max": 3147},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 488, "range_min": 875, "range_max": 3065},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -138, "range_min": 824, "range_max": 3117},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -1590, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1543, "range_min": 1755, "range_max": 3225},
}


@pytest.fixture
def lerobot_calibration(tmp_path, monkeypatch):
    """A LeRobot calibration directory holding follower_1's (and the same for follower_2)."""
    directory = tmp_path / "calibration"
    directory.mkdir()
    for arm in ("follower_1", "follower_2"):
        (directory / f"{arm}.json").write_text(json.dumps(F1_CALIBRATION))
    monkeypatch.setenv("MAGNUS_CALIBRATION_DIR", str(directory))
    return directory

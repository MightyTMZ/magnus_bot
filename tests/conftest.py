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
    (the operator's working copy may carry ``measured: true``; the repo ships ``false``), and no
    tuned grasps (configs/grasp.yaml is left out)."""

    def make(measured: bool) -> Path:
        out = tmp_path / f"configs-measured-{measured}"
        shutil.copytree(CONFIGS, out)
        board = out / "board.yaml"
        text, n = re.subn(r"(?m)^measured:\s*\w+", f"measured: {str(measured).lower()}", board.read_text())
        assert n == 1, "board.yaml should have exactly one top-level measured: key"
        board.write_text(text)
        (out / "grasp.yaml").unlink(missing_ok=True)  # start untuned; tests save their own grasps
        return out

    return make

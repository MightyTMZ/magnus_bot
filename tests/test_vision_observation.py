"""Board observations: the HTTP client for RF-DETR/vision_server.py, and the simulated board."""

import socket
from urllib.parse import parse_qs, urlparse

import pytest

from conftest import BOARD_JSON
from magnus_bot.vision.observation import (
    BoardObservation,
    BoardReader,
    HttpBoardReader,
    VisionTimeout,
    VisionUnavailable,
)
from magnus_bot.vision.sim_board import SimulatedBoard


def _query(path):
    return {k: v[0] for k, v in parse_qs(urlparse(path).query).items()}


def test_observation_from_json():
    obs = BoardObservation.from_json(BOARD_JSON)
    assert obs.occupancy == {"g3": "W"}
    assert obs.pieces == {"g3": "white-rook"}
    assert (obs.frame_id, obs.timestamp, obs.stable) == (42, 1.5, True)


def test_readers_satisfy_the_protocol():
    assert isinstance(HttpBoardReader(), BoardReader)
    assert isinstance(SimulatedBoard({}), BoardReader)


def test_reader_asks_for_a_fresh_board_and_parses_it(vision_server):
    obs = HttpBoardReader(vision_server.url).read(fresh=True, timeout=3.0)
    assert obs.occupancy == {"g3": "W"} and obs.frame_id == 42
    assert urlparse(vision_server.paths[-1]).path == "/board"
    assert _query(vision_server.paths[-1]) == {"fresh": "1", "timeout": "3.0"}


def test_reader_can_ask_for_the_latest_board(vision_server):
    HttpBoardReader(vision_server.url).read(fresh=False)
    assert _query(vision_server.paths[-1])["fresh"] == "0"


def test_gateway_timeout_raises_vision_timeout(vision_server):
    vision_server.reply = (504, {"error": "no stable board within 8s"})
    with pytest.raises(VisionTimeout, match="no stable board within 8s"):
        HttpBoardReader(vision_server.url).read()


def test_other_errors_raise_vision_unavailable(vision_server):
    vision_server.reply = (503, {"error": "no frames yet"})
    with pytest.raises(VisionUnavailable, match="503: no frames yet") as excinfo:
        HttpBoardReader(vision_server.url).read()
    assert not isinstance(excinfo.value, VisionTimeout)


def test_unreachable_server_raises_vision_unavailable():
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(VisionUnavailable, match="vision_server.py running"):
        HttpBoardReader(f"http://127.0.0.1:{port}", slack=1.0).read(timeout=0.5)


def test_health(vision_server):
    vision_server.reply = (200, {"ok": True, "fps": 25.0, "camera_ok": True})
    assert HttpBoardReader(vision_server.url).health()["fps"] == 25.0
    assert vision_server.paths[-1] == "/health"


def test_simulated_board_applies_transfers():
    world = SimulatedBoard({"g3": "W", "a1": "B"})
    first = world.read()
    assert world.move("g3", "g4") is True
    second = world.read()
    assert second.occupancy == {"g4": "W", "a1": "B"}
    assert second.frame_id > first.frame_id


def test_simulated_board_misses_are_seeded():
    always = SimulatedBoard({"g3": "W"}, miss_rate=1.0)
    assert always.move("g3", "g4") is False
    assert always.read().occupancy == {"g3": "W"}

    def run(seed):
        world = SimulatedBoard({"g3": "W"}, miss_rate=0.5, seed=seed)
        hits = []
        for _ in range(10):
            src = "g3" if "g3" in world.occupancy else "g4"
            hits.append(world.move(src, "g4" if src == "g3" else "g3"))
        return hits

    assert run(7) == run(7)

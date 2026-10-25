"""Board observations from the vision process, and the HTTP client that fetches them.

The camera and RF-DETR run as their OWN process (``RF-DETR/vision_server.py``, in RF-DETR's
venv): its torch 2.14 + CUDA cannot share an environment with lerobot's ``torch<2.12`` pin. The
server publishes the latest per-square reading over localhost; this module is the core side of
that link — plain data plus urllib, so magnus_bot never imports cv2 or torch.

Squares are named as in python-chess (``"g3"``). Colors are ``"W"``, ``"B"``, or ``"?"``
(occupied, color unknown — the detector's generic "pieces" class). The green-painted fingertip
(``marker``) is a board position in SQUARES: x toward the h-file, y toward rank 8, origin at a1's
outer corner (a1's center is (0.5, 0.5)); multiply by the square size for meters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

DEFAULT_VISION_URL = "http://127.0.0.1:8765"

Occupancy = dict[str, str]  # square name -> "W" | "B" | "?"


@dataclass(frozen=True)
class BoardObservation:
    """One reading of the physical board."""

    occupancy: Occupancy
    pieces: dict[str, str] = field(default_factory=dict)  # square -> class name, e.g. "white-rook"
    frame_id: int = 0
    timestamp: float = 0.0
    stable: bool = True
    marker: tuple[float, float] | None = None  # fingertip, board squares (see module docstring)
    marker_seen: int = 0  # readings in the window that saw the fingertip
    board_visible: bool = True

    @classmethod
    def from_json(cls, data: dict) -> BoardObservation:
        marker = data.get("marker")
        return cls(
            occupancy={str(sq): str(c) for sq, c in data["occupancy"].items()},
            pieces={str(sq): str(name) for sq, name in data.get("pieces", {}).items()},
            frame_id=int(data.get("frame_id", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
            stable=bool(data.get("stable", False)),
            marker=(float(marker[0]), float(marker[1])) if marker else None,
            marker_seen=int(data.get("marker_seen", 0)),
            board_visible=bool(data.get("board_visible", True)),
        )


class VisionUnavailable(RuntimeError):
    """The vision server could not be reached, or answered with an error."""


class VisionTimeout(VisionUnavailable):
    """The server is up but saw no stable board in time (arm or hand in view, camera stalled)."""


@runtime_checkable
class BoardReader(Protocol):
    """Anything that can report what is on the physical board."""

    def read(self, fresh: bool = True, timeout: float = 8.0) -> BoardObservation:
        """Return a reading. ``fresh``: a stable board captured entirely after this call."""
        ...


class HttpBoardReader:
    """Reads the board from ``RF-DETR/vision_server.py`` over HTTP."""

    def __init__(self, url: str = DEFAULT_VISION_URL, slack: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.slack = slack  # socket time allowed beyond the server-side wait

    def read(self, fresh: bool = True, timeout: float = 8.0) -> BoardObservation:
        query = urlencode({"fresh": int(fresh), "timeout": timeout})
        return BoardObservation.from_json(self._get(f"/board?{query}", timeout + self.slack))

    def health(self) -> dict:
        return self._get("/health", self.slack)

    def _get(self, path: str, timeout: float) -> dict:
        try:
            with urlopen(self.url + path, timeout=timeout) as resp:
                return json.load(resp)
        except HTTPError as e:  # before URLError: HTTPError subclasses it
            detail = _error_detail(e)
            if e.code == 504:
                raise VisionTimeout(f"no stable board from {self.url}: {detail}") from e
            raise VisionUnavailable(f"vision server {self.url} answered {e.code}: {detail}") from e
        except (URLError, OSError, ValueError) as e:  # refused, socket timeout, bad JSON
            raise VisionUnavailable(
                f"vision server not reachable at {self.url} ({e}); is RF-DETR/vision_server.py running?"
            ) from e


def _error_detail(e: HTTPError) -> str:
    try:
        return str(json.load(e).get("error", e.reason))
    except Exception:
        return str(e.reason)

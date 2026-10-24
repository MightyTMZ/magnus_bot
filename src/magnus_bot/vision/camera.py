"""The :class:`Camera` protocol and an OpenCV-backed stub.

An overhead camera with locked exposure/focus feeds board verification (M4) and, eventually,
physical move input (:mod:`magnus_bot.sources.board_source`). Any backend (USB webcam app, Pi
camera) that yields frames fits behind this one interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Camera(Protocol):
    """A source of overhead frames of the board."""

    def open(self) -> None:
        """Acquire the device (and apply locked exposure/focus settings)."""
        ...

    def read(self) -> np.ndarray:
        """Return the latest frame as an (H, W, 3) BGR uint8 array."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...


class OpenCVCamera:
    """STUB (M4): OpenCV ``VideoCapture`` camera. cv2 is imported lazily so the core package
    never requires OpenCV."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._cap = None  # cv2.VideoCapture once opened

    def open(self) -> None:
        # TODO(M4): import cv2 lazily, open VideoCapture(device_index), lock exposure/focus
        # (CAP_PROP_AUTO_EXPOSURE / CAP_PROP_AUTOFOCUS), verify a frame reads back.
        raise NotImplementedError("OpenCVCamera lands in milestone M4")

    def read(self) -> np.ndarray:
        raise NotImplementedError("OpenCVCamera lands in milestone M4")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

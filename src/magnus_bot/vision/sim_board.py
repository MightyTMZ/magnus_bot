"""A virtual board that stands in for the camera in sim runs and tests (cf. ``SimArm``)."""

from __future__ import annotations

import random

from magnus_bot.vision.observation import BoardObservation, Occupancy
from magnus_bot.vision.verify import relocate


class SimulatedBoard:
    """A :class:`~magnus_bot.vision.observation.BoardReader` over a virtual board.

    Sim callers report each physical transfer with :meth:`move`; ``miss_rate`` is the chance a
    transfer leaves the board untouched (a missed grasp), seeded for repeatable runs.
    """

    def __init__(self, occupancy: Occupancy, miss_rate: float = 0.0, seed: int = 0) -> None:
        self.occupancy = dict(occupancy)
        self.miss_rate = miss_rate
        self._rng = random.Random(seed)
        self._frame = 0

    def move(self, src: str, dst: str) -> bool:
        """Apply a transfer; returns False if this one was a simulated miss."""
        if self._rng.random() < self.miss_rate:
            return False
        self.occupancy = relocate(self.occupancy, src, dst)
        return True

    def read(self, fresh: bool = True, timeout: float = 8.0) -> BoardObservation:
        self._frame += 1
        return BoardObservation(occupancy=dict(self.occupancy), frame_id=self._frame)

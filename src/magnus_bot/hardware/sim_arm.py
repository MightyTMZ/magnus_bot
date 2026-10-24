"""SimArm: an in-memory arm for dry-run execution and tests. Never touches hardware.

Holds joint state, clamps every target to limits, and records a command history so plans can be
asserted against in tests. This is the default arm in dry-run/sim mode.
"""

from __future__ import annotations

from magnus_bot.kinematics.model import JOINT_NAMES, JointLimit, SO101Model
from magnus_bot.motion import safety


class SimArm:
    """A simulated :class:`~magnus_bot.hardware.arm.Arm`.

    ``start`` is the initial joint state (defaults to all-zeros, gripper open at 100). Targets are
    clamped to ``limits`` (from :class:`SO101Model` by default) exactly as a real arm would.
    """

    def __init__(
        self,
        arm_id: str,
        limits: dict[str, JointLimit] | None = None,
        start: dict[str, float] | None = None,
    ) -> None:
        self.arm_id = arm_id
        self._limits = limits or SO101Model().limits
        self._connected = False
        self._state: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self._state["gripper"] = 100.0
        if start:
            self._state.update(start)
        self.command_log: list[dict[str, float]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_joints(self) -> dict[str, float]:
        self._require_connected()
        return dict(self._state)

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        self._require_connected()
        sent = safety.clamp_joints(targets, self._limits)
        self._state.update(sent)
        self.command_log.append(dict(sent))
        return sent

    def hold(self) -> dict[str, float]:
        return self.send_joints(dict(self._state))

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"{self.arm_id}: not connected")

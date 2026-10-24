"""The :class:`Arm` protocol: the single interface the orchestrator uses to drive an arm.

Units are SI: arm joint angles in radians, the gripper in percent (0..100) as the one hardware
exception (the gripper has no meaningful "radians"). Joint dicts are keyed by
:data:`magnus_bot.kinematics.model.JOINT_NAMES`. Implementations must clamp targets to limits
before commanding (see :mod:`magnus_bot.motion.safety`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Arm(Protocol):
    """A single controllable arm. Constructing one must not move or energize hardware."""

    arm_id: str

    @property
    def is_connected(self) -> bool:
        ...

    def connect(self) -> None:
        """Open the connection. Does not move the arm."""
        ...

    def disconnect(self) -> None:
        """Release the connection, following the configured torque policy."""
        ...

    def read_joints(self) -> dict[str, float]:
        """Current joint positions (radians; gripper percent), keyed by joint name."""
        ...

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Command joint targets (radians; gripper percent). Returns the values actually sent
        (post-clamp), mirroring LeRobot's send_action contract."""
        ...

    def hold(self) -> dict[str, float]:
        """Command the current position (fault policy: hold, do not cut torque)."""
        ...

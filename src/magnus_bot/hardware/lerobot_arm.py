"""STUB (M1, unvalidated on hardware): the LeRobot-backed live arm.

Wraps LeRobot's SO101Follower (lerobot.robots.so_follower). This is the ONLY place degrees cross
the boundary: LeRobot runs in degrees (``use_degrees=True``) with the gripper in percent, while
magnus_bot works in radians. The five revolute joints convert rad<->deg here; the gripper passes
through as percent.

``lerobot`` is imported lazily inside :meth:`connect`, so the core package never requires it.
The wiring mirrors the proven arm_test harness (max_relative_target clamp, .pos observation
keys), but it is NOT yet validated end-to-end — hence a stub. Live motion stays gated by the
executor/CLI (``--live`` + approval + ``measured: true``); constructing this class does nothing
to hardware.
"""

from __future__ import annotations

import math

from magnus_bot.kinematics.model import ARM_JOINTS

# Per-command safety clamp handed to SO101FollowerConfig(max_relative_target=...), in the arm's
# native units (degrees for revolute joints / percent for gripper). PLACEHOLDER until tuned.
DEFAULT_MAX_RELATIVE_TARGET = 4.0


class LeRobotArm:
    """Live SO-101 follower arm. See module docstring; unvalidated on hardware (M1)."""

    def __init__(
        self,
        arm_id: str,
        port: str,
        calibration_id: str | None = None,
        max_relative_target: float | None = DEFAULT_MAX_RELATIVE_TARGET,
    ) -> None:
        self.arm_id = arm_id
        self.port = port
        self.calibration_id = calibration_id or arm_id
        self.max_relative_target = max_relative_target
        self._robot = None  # lerobot SO101Follower once connected

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and self._robot.is_connected

    def connect(self) -> None:
        # Lazy import so the core package has no hard dependency on lerobot.
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cfg = SO101FollowerConfig(
            port=self.port,
            id=self.calibration_id,
            max_relative_target=self.max_relative_target,
            use_degrees=True,
        )
        robot = SO101Follower(cfg)
        robot.connect(calibrate=False)  # never auto-calibrate from magnus_bot
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"{self.arm_id} ({self.port}): calibration does not match the servos. "
                "Recalibrate with LeRobot before running."
            )
        self._robot = robot

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()  # follows disable_torque_on_disconnect
            self._robot = None

    def read_joints(self) -> dict[str, float]:
        obs = self._require_robot().get_observation()
        native = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
        return {name: self._to_si(name, val) for name, val in native.items()}

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        action = {f"{name}.pos": self._to_native(name, val) for name, val in targets.items()}
        sent = self._require_robot().send_action(action)  # returns possibly-clamped action
        return {k.removesuffix(".pos"): self._to_si(k.removesuffix(".pos"), v) for k, v in sent.items()}

    def hold(self) -> dict[str, float]:
        return self.send_joints(self.read_joints())

    # --- unit conversion at the hardware edge ---

    @staticmethod
    def _to_native(name: str, value_si: float) -> float:
        """SI (radians / percent) -> LeRobot native (degrees / percent)."""
        return math.degrees(value_si) if name in ARM_JOINTS else value_si

    @staticmethod
    def _to_si(name: str, value_native: float) -> float:
        """LeRobot native (degrees / percent) -> SI (radians / percent)."""
        return math.radians(value_native) if name in ARM_JOINTS else value_native

    def _require_robot(self):
        if self._robot is None:
            raise RuntimeError(f"{self.arm_id}: not connected")
        return self._robot

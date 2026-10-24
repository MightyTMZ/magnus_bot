"""Hardware abstraction: the :class:`Arm` protocol and its sim/live implementations.

The orchestrator talks only to :class:`Arm` (SI: radians, gripper percent). :class:`SimArm` is a
full in-memory arm for dry-run/tests; :class:`LeRobotArm` wraps LeRobot's SO101Follower and is
the only place degrees/percent cross the boundary. Live motion is gated elsewhere (see CLAUDE.md
safety rules): constructing an arm never moves hardware.
"""

from magnus_bot.hardware.arm import Arm
from magnus_bot.hardware.sim_arm import SimArm

__all__ = ["Arm", "SimArm"]

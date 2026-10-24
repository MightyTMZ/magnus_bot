"""Kinematics for the SO-101 arm: joint model, analytic IK (stub), reachability.

Analytic IK is the control path (M1); a numeric solver is only ever a test cross-check.
Joint names/order match LeRobot's SO101Follower bus exactly; see model.py and
docs/joint_conventions.md for the URDF-zero <-> servo-calibration mapping (a known pitfall).
"""

from magnus_bot.kinematics.model import JOINT_NAMES, SO101Model
from magnus_bot.kinematics.reach import ReachabilityMap

__all__ = ["JOINT_NAMES", "SO101Model", "ReachabilityMap"]

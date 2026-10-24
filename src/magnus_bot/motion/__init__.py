"""Motion: safety clamps, pick/place primitives, and trajectory generation.

  * safety: clamp joint targets to limits and the speed scale (always on the command path).
  * primitives: the hover/descend/grip/lift/traverse/descend/release/retract step sequence.
  * trajectory: min-jerk interpolation between waypoints (full planner is M1).
"""

from magnus_bot.motion import primitives, safety, trajectory

__all__ = ["primitives", "safety", "trajectory"]

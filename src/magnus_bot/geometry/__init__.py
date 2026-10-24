"""Board and arm geometry in the board frame B.

Frame B: origin at the a1 corner, +x toward the h-file, +y toward rank 8, +z up.
See docs/frames.md for the full definition and the orientation facts the tests pin.
"""

from magnus_bot.geometry import board, frames, transforms

__all__ = ["board", "frames", "transforms"]

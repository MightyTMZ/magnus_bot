"""Planning: turn a chess move into physical piece transfers and assign them to arms.

  * move_decomposer: chess.Move (+ board) -> ordered list of Transfer (capture, castling,
    en passant, promotion all handled).
  * assigner: decide single-arm vs. handoff for each transfer from a ReachabilityMap.
  * handoff: expand a cross-arm transfer into the pad-mediated A->pad, pad->B sequence.
"""

from magnus_bot.planning.assigner import Assignment, assign_transfer
from magnus_bot.planning.move_decomposer import Location, Transfer, decompose_move

__all__ = ["Location", "Transfer", "decompose_move", "Assignment", "assign_transfer"]

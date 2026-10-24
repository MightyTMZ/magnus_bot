"""Move sources: everything that can produce the next move for a side to play.

All sources implement :class:`MoveSource` and return a move validated against
``board.legal_moves``. UCI strings are canonical; python-chess owns legality.
"""

from magnus_bot.sources.move_source import MoveSource, validate_move
from magnus_bot.sources.random_source import RandomLegalSource

__all__ = ["MoveSource", "validate_move", "RandomLegalSource"]

"""Orchestration: the game state machine and the executor that drives hardware through it.

The state machine (state_machine.py) is pure and fully testable; the executor (executor.py)
injects hardware and planning and realizes transitions. Robots play White and move first.
"""

from magnus_bot.orchestration.state_machine import (
    Event,
    InvalidTransition,
    State,
    StateMachine,
    next_state,
)

__all__ = ["State", "Event", "StateMachine", "next_state", "InvalidTransition"]

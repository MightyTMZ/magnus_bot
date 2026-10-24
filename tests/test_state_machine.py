"""Game state-machine transition table (pure)."""

import pytest

from magnus_bot.orchestration.state_machine import (
    Event,
    InvalidTransition,
    State,
    StateMachine,
    next_state,
)


def test_happy_path_sequence():
    sm = StateMachine()
    assert sm.state == State.START
    assert sm.send(Event.BEGIN) == State.ENGINE_THINK
    assert sm.send(Event.MOVE_READY) == State.PLAN
    assert sm.send(Event.PLAN_READY) == State.EXEC_STEP
    assert sm.send(Event.MORE_STEPS) == State.EXEC_STEP
    assert sm.send(Event.ALL_STEPS_DONE) == State.VERIFY
    assert sm.send(Event.VERIFIED) == State.AWAIT_HUMAN_MOVE
    assert sm.send(Event.HUMAN_MOVED) == State.VALIDATE_MOVE
    assert sm.send(Event.MOVE_VALID) == State.ENGINE_THINK


def test_handoff_detour():
    assert next_state(State.EXEC_STEP, Event.NEED_HANDOFF_PLACE) == State.HANDOFF_PLACE
    assert next_state(State.HANDOFF_PLACE, Event.HANDOFF_PLACED) == State.HANDOFF_PICK
    assert next_state(State.HANDOFF_PICK, Event.HANDOFF_PICKED) == State.EXEC_STEP


def test_invalid_move_returns_to_await():
    assert next_state(State.VALIDATE_MOVE, Event.MOVE_INVALID) == State.AWAIT_HUMAN_MOVE


@pytest.mark.parametrize(
    "state",
    [State.ENGINE_THINK, State.EXEC_STEP, State.HANDOFF_PICK, State.VERIFY, State.AWAIT_HUMAN_MOVE],
)
def test_fault_and_estop_reachable_from_anywhere(state):
    assert next_state(state, Event.FAULT) == State.ERROR_RECOVERY
    assert next_state(state, Event.ESTOP) == State.ESTOP


def test_recovery_and_reset():
    assert next_state(State.ERROR_RECOVERY, Event.RESUME) == State.ENGINE_THINK
    assert next_state(State.ERROR_RECOVERY, Event.RESET) == State.START
    assert next_state(State.ESTOP, Event.RESET) == State.START


def test_invalid_transition_raises():
    with pytest.raises(InvalidTransition):
        next_state(State.START, Event.MOVE_READY)


def test_terminal_state_accepts_nothing():
    assert next_state(State.ENGINE_THINK, Event.GAME_ENDED) == State.GAME_OVER
    with pytest.raises(InvalidTransition):
        next_state(State.GAME_OVER, Event.BEGIN)

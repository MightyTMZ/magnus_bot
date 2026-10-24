"""The game orchestration state machine: states, events, and an explicit transition table.

Pure and hardware-free so it is fully unit-testable; the executor supplies the events from real
or simulated hardware. Robots play White and move first, so the loop starts by thinking.

Happy path:
    START -> ENGINE_THINK -> PLAN -> EXEC_STEP (with HANDOFF_PLACE/HANDOFF_PICK as needed)
          -> VERIFY -> AWAIT_HUMAN_MOVE -> VALIDATE_MOVE -> ENGINE_THINK -> ...

ERROR_RECOVERY and ESTOP are reachable from ANY state (via the FAULT / ESTOP events). See
docs/state_machine.md for the mermaid diagram.
"""

from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    START = auto()
    ENGINE_THINK = auto()  # robots' move source produces a move
    PLAN = auto()  # decompose + assign + build motion program
    EXEC_STEP = auto()  # execute the next piece transfer (single-arm)
    HANDOFF_PLACE = auto()  # arm A places on the pad
    HANDOFF_PICK = auto()  # arm B picks from the pad
    VERIFY = auto()  # confirm the board matches the intended position
    AWAIT_HUMAN_MOVE = auto()  # wait for the human to move
    VALIDATE_MOVE = auto()  # check the human's move is legal
    ERROR_RECOVERY = auto()
    ESTOP = auto()
    GAME_OVER = auto()


class Event(Enum):
    BEGIN = auto()
    MOVE_READY = auto()  # a move was produced
    PLAN_READY = auto()  # motion program built
    MORE_STEPS = auto()  # another single-arm transfer remains
    NEED_HANDOFF_PLACE = auto()  # next transfer needs a pad handoff
    HANDOFF_PLACED = auto()  # arm A done placing on pad
    HANDOFF_PICKED = auto()  # arm B done picking from pad
    ALL_STEPS_DONE = auto()  # the whole move is executed
    VERIFIED = auto()
    VERIFY_FAILED = auto()
    HUMAN_MOVED = auto()
    MOVE_VALID = auto()
    MOVE_INVALID = auto()
    GAME_ENDED = auto()  # checkmate/stalemate/draw
    FAULT = auto()  # any hardware/logic fault -> ERROR_RECOVERY
    ESTOP = auto()  # emergency stop -> ESTOP
    RESUME = auto()  # leave ERROR_RECOVERY, retry the move
    RESET = auto()  # leave ERROR_RECOVERY/ESTOP back to START


class InvalidTransition(RuntimeError):
    """Raised when an event is not valid in the current state."""

    def __init__(self, state: State, event: Event) -> None:
        self.state = state
        self.event = event
        super().__init__(f"no transition from {state.name} on {event.name}")


# Explicit (state, event) -> state table. FAULT/ESTOP are handled globally below and are
# intentionally absent here.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.START, Event.BEGIN): State.ENGINE_THINK,
    (State.ENGINE_THINK, Event.MOVE_READY): State.PLAN,
    (State.ENGINE_THINK, Event.GAME_ENDED): State.GAME_OVER,
    (State.PLAN, Event.PLAN_READY): State.EXEC_STEP,
    (State.EXEC_STEP, Event.MORE_STEPS): State.EXEC_STEP,
    (State.EXEC_STEP, Event.NEED_HANDOFF_PLACE): State.HANDOFF_PLACE,
    (State.EXEC_STEP, Event.ALL_STEPS_DONE): State.VERIFY,
    (State.HANDOFF_PLACE, Event.HANDOFF_PLACED): State.HANDOFF_PICK,
    (State.HANDOFF_PICK, Event.HANDOFF_PICKED): State.EXEC_STEP,
    (State.VERIFY, Event.VERIFIED): State.AWAIT_HUMAN_MOVE,
    (State.VERIFY, Event.VERIFY_FAILED): State.ERROR_RECOVERY,
    (State.AWAIT_HUMAN_MOVE, Event.HUMAN_MOVED): State.VALIDATE_MOVE,
    (State.AWAIT_HUMAN_MOVE, Event.GAME_ENDED): State.GAME_OVER,
    (State.VALIDATE_MOVE, Event.MOVE_VALID): State.ENGINE_THINK,
    (State.VALIDATE_MOVE, Event.MOVE_INVALID): State.AWAIT_HUMAN_MOVE,
    (State.VALIDATE_MOVE, Event.GAME_ENDED): State.GAME_OVER,
    (State.ERROR_RECOVERY, Event.RESUME): State.ENGINE_THINK,
    (State.ERROR_RECOVERY, Event.RESET): State.START,
    (State.ESTOP, Event.RESET): State.START,
}

# States from which no further events are accepted.
TERMINAL: frozenset[State] = frozenset({State.GAME_OVER})


def next_state(state: State, event: Event) -> State:
    """Return the state resulting from ``event`` in ``state``. Pure; raises on invalid input.

    FAULT and ESTOP are valid from any non-terminal state (safety: always reachable).
    """
    if state in TERMINAL:
        raise InvalidTransition(state, event)
    if event is Event.ESTOP:
        return State.ESTOP
    if event is Event.FAULT:
        return State.ERROR_RECOVERY
    try:
        return TRANSITIONS[(state, event)]
    except KeyError:
        raise InvalidTransition(state, event) from None


class StateMachine:
    """A tiny stateful wrapper over :func:`next_state`, tracking history for debugging."""

    def __init__(self, state: State = State.START) -> None:
        self.state = state
        self.history: list[tuple[State, Event, State]] = []

    def send(self, event: Event) -> State:
        new = next_state(self.state, event)
        self.history.append((self.state, event, new))
        self.state = new
        return new

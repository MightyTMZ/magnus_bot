"""Arm assignment: single-arm when one arm reaches both ends, else a pad handoff."""

import pytest

from magnus_bot.kinematics.reach import ReachabilityMap
from magnus_bot.planning.assigner import HANDOFF, SINGLE, Unassignable, assign_transfer
from magnus_bot.planning.move_decomposer import Location, Transfer

ARMS = {"follower_1", "follower_2"}


def _transfer(src_sq, dst_sq):
    return Transfer(Location(kind="square", square=src_sq),
                    Location(kind="square", square=dst_sq), "P", "move")


def test_single_arm_when_one_arm_reaches_both():
    rmap = ReachabilityMap(
        reachable={"follower_1": {(0, 0), (7, 7)}, "follower_2": {(0, 0), (7, 7)}},
        pad_reach=ARMS,
    )
    a = assign_transfer(_transfer((0, 0), (7, 7)), rmap, ARMS)
    assert a.mode == SINGLE
    assert a.arm == "follower_1"  # deterministic tie-break by sorted id


def test_handoff_when_no_single_arm_reaches_both():
    rmap = ReachabilityMap(
        reachable={"follower_1": {(0, 0)}, "follower_2": {(7, 7)}},
        pad_reach=ARMS,
    )
    a = assign_transfer(_transfer((0, 0), (7, 7)), rmap, ARMS)
    assert a.mode == HANDOFF
    assert a.pick_arm == "follower_1"  # reaches the source + pad
    assert a.place_arm == "follower_2"  # reaches the pad + destination


def test_unassignable_when_pad_unreachable():
    rmap = ReachabilityMap(
        reachable={"follower_1": {(0, 0)}, "follower_2": {(7, 7)}},
        pad_reach=set(),
    )
    with pytest.raises(Unassignable):
        assign_transfer(_transfer((0, 0), (7, 7)), rmap, ARMS)

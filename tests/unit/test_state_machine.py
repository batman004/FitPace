"""Pure-logic tests for the goal state machine."""
from __future__ import annotations

from app.models.enums import GoalState
from app.services.state_machine import evaluate_transition


def test_on_track_at_high_pace() -> None:
    assert evaluate_transition(GoalState.ON_TRACK, 85.0) == GoalState.ON_TRACK


def test_at_risk_at_mid_pace() -> None:
    assert evaluate_transition(GoalState.ON_TRACK, 70.0) == GoalState.AT_RISK


def test_off_track_at_low_pace() -> None:
    assert evaluate_transition(GoalState.AT_RISK, 50.0) == GoalState.OFF_TRACK


def test_recovered_from_off_track_when_pace_crosses_70() -> None:
    assert (
        evaluate_transition(GoalState.OFF_TRACK, 72.0) == GoalState.RECOVERED
    )


def test_off_track_remains_when_pace_still_low() -> None:
    assert evaluate_transition(GoalState.OFF_TRACK, 55.0) == GoalState.OFF_TRACK


def test_on_track_falls_to_off_track_at_sub_60() -> None:
    assert evaluate_transition(GoalState.ON_TRACK, 30.0) == GoalState.OFF_TRACK

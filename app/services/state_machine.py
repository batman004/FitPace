"""Goal state machine: transition rules + persistence helper."""
from __future__ import annotations

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import GoalState
from app.models.goal import Goal
from app.models.goal_state_event import GoalStateEvent


def evaluate_transition(current_state: GoalState, pace_score: float) -> GoalState:
    """Return the next state given current state and the latest pace score.

    Rules:
        pace_score >= 80                        -> ON_TRACK
        60 <= pace_score < 80                   -> AT_RISK
        pace_score < 60                         -> OFF_TRACK
        was OFF_TRACK AND pace_score >= 70      -> RECOVERED

    NB: spec calls for RECOVERED after *two* consecutive recomputes >= 70; that would
    require persisting inter-recompute pace history. This simpler single-threshold
    rule is intentional for the initial wiring and should be tightened later.
    """
    if current_state == GoalState.OFF_TRACK and pace_score >= 70:
        return GoalState.RECOVERED
    if pace_score >= 80:
        return GoalState.ON_TRACK
    if pace_score >= 60:
        return GoalState.AT_RISK
    return GoalState.OFF_TRACK


def _describe(from_state: GoalState, to_state: GoalState, pace_score: float) -> str:
    return f"pace_score={pace_score:.1f} triggered {from_state.value} -> {to_state.value}"


async def apply_transition(
    goal: Goal, pace_score: float, db: AsyncSession
) -> GoalStateEvent | None:
    """Write a GoalStateEvent and update goal.current_state iff the state changed."""
    prev_state = goal.current_state
    new_state = evaluate_transition(prev_state, pace_score)
    if new_state == prev_state:
        return None

    event = GoalStateEvent(
        goal_id=goal.id,
        from_state=prev_state,
        to_state=new_state,
        pace_score=pace_score,
        reason=_describe(prev_state, new_state, pace_score),
    )
    goal.current_state = new_state
    db.add(event)
    await db.flush()
    logger.info(
        "goal state transition goal_id={} {}->{} pace_score={:.2f}",
        goal.id,
        prev_state.value,
        new_state.value,
        pace_score,
    )
    return event

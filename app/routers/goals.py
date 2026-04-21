"""Goal endpoints: CRUD + trajectory + history."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.enums import GoalState
from app.models.goal import Goal
from app.models.goal_state_event import GoalStateEvent
from app.models.progress_log import ProgressLog
from app.models.user import User
from app.schemas.goal import GoalCreate, GoalRead, GoalStateEventRead
from app.schemas.trajectory import TrajectoryRead
from app.services.trajectory_service import compute_trajectory

router = APIRouter(prefix="/goals", tags=["goals"])


@router.get("", response_model=list[GoalRead])
async def list_goals(
    user_id: UUID, db: AsyncSession = Depends(get_db)
) -> list[Goal]:
    """List goals for a user, newest first. Used by the dashboard UI."""
    result = await db.execute(
        select(Goal)
        .where(Goal.user_id == user_id)
        .order_by(Goal.created_at.desc())
    )
    goals = list(result.scalars().all())
    logger.debug("list_goals user_id={} n={}", user_id, len(goals))
    return goals


@router.post("", status_code=status.HTTP_201_CREATED, response_model=GoalRead)
async def create_goal(
    payload: GoalCreate, db: AsyncSession = Depends(get_db)
) -> Goal:
    goal = Goal(
        user_id=payload.user_id,
        goal_type=payload.goal_type,
        start_value=payload.start_value,
        target_value=payload.target_value,
        unit=payload.unit,
        start_date=payload.start_date,
        target_date=payload.target_date,
        current_state=GoalState.ON_TRACK,
    )
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    logger.info(
        "goal created goal_id={} user_id={} type={} {}->{} {} by={}",
        goal.id,
        goal.user_id,
        goal.goal_type.value if hasattr(goal.goal_type, "value") else goal.goal_type,
        goal.start_value,
        goal.target_value,
        goal.unit,
        goal.target_date,
    )
    return goal


@router.get("/{goal_id}", response_model=GoalRead)
async def get_goal(goal_id: UUID, db: AsyncSession = Depends(get_db)) -> Goal:
    goal = await db.get(Goal, goal_id)
    if goal is None:
        logger.info("goal lookup miss goal_id={}", goal_id)
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="goal not found")
    return goal


@router.get("/{goal_id}/trajectory", response_model=TrajectoryRead)
async def get_trajectory(
    goal_id: UUID, db: AsyncSession = Depends(get_db)
) -> TrajectoryRead:
    goal = await db.get(Goal, goal_id)
    if goal is None:
        logger.info("trajectory requested for missing goal_id={}", goal_id)
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="goal not found")

    result = await db.execute(
        select(ProgressLog)
        .where(ProgressLog.goal_id == goal_id)
        .order_by(ProgressLog.logged_at)
    )
    logs = result.scalars().all()
    user = await db.get(User, goal.user_id)
    traj = compute_trajectory(goal, logs, user=user)
    logger.info(
        "trajectory computed goal_id={} logs={} pace_score={} eta={} days_ahead={}",
        goal_id,
        len(logs),
        traj.pace_score,
        traj.eta_date,
        traj.days_ahead,
    )
    return TrajectoryRead(
        goal_id=traj.goal_id,
        pace_score=traj.pace_score,
        eta_date=traj.eta_date,
        days_ahead=traj.days_ahead,
        computed_at=traj.computed_at,
    )


@router.get("/{goal_id}/history", response_model=list[GoalStateEventRead])
async def get_history(
    goal_id: UUID, db: AsyncSession = Depends(get_db)
) -> list[GoalStateEvent]:
    goal = await db.get(Goal, goal_id)
    if goal is None:
        logger.info("history requested for missing goal_id={}", goal_id)
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="goal not found")

    result = await db.execute(
        select(GoalStateEvent)
        .where(GoalStateEvent.goal_id == goal_id)
        .order_by(GoalStateEvent.occurred_at)
    )
    events = list(result.scalars().all())
    logger.debug("goal history goal_id={} events={}", goal_id, len(events))
    return events

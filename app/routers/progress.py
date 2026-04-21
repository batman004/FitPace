"""Progress ingestion endpoints."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.goal import Goal
from app.models.progress_log import ProgressLog
from app.models.user import User
from app.schemas.progress_log import ProgressLogCreate, ProgressLogRead
from app.services.state_machine import apply_transition
from app.services.trajectory_service import compute_trajectory

router = APIRouter(prefix="/progress", tags=["progress"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProgressLogRead)
async def create_progress(
    payload: ProgressLogCreate, db: AsyncSession = Depends(get_db)
) -> ProgressLog:
    goal = await db.get(Goal, payload.goal_id)
    if goal is None:
        logger.info("progress rejected: goal not found goal_id={}", payload.goal_id)
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="goal not found")

    log = ProgressLog(
        goal_id=payload.goal_id,
        logged_at=payload.logged_at,
        value=payload.value,
        notes=payload.notes,
    )
    db.add(log)
    await db.flush()

    result = await db.execute(
        select(ProgressLog)
        .where(ProgressLog.goal_id == payload.goal_id)
        .order_by(ProgressLog.logged_at)
    )
    logs = list(result.scalars().all())

    # Need >=2 logs for a real slope; otherwise compute_trajectory returns its
    # neutral fallback (pace_score=50) and we must NOT feed that to the state
    # machine, or every first log flips the goal straight to OFF_TRACK.
    if len(logs) >= 2:
        user = await db.get(User, goal.user_id)
        traj = compute_trajectory(goal, logs, user=user)
        event = await apply_transition(goal, traj.pace_score, db)
        if event is not None:
            logger.info(
                "state transition goal_id={} {}->{} pace_score={} eta={}",
                goal.id,
                event.from_state.value,
                event.to_state.value,
                traj.pace_score,
                traj.eta_date,
            )
        else:
            logger.debug(
                "no state change goal_id={} state={} pace_score={}",
                goal.id,
                goal.current_state.value,
                traj.pace_score,
            )

    await db.commit()
    await db.refresh(log)
    logger.info(
        "progress logged log_id={} goal_id={} value={} logged_at={} logs_total={}",
        log.id,
        log.goal_id,
        log.value,
        log.logged_at,
        len(logs),
    )
    return log


@router.get("/{goal_id}", response_model=list[ProgressLogRead])
async def list_progress(
    goal_id: UUID, db: AsyncSession = Depends(get_db)
) -> list[ProgressLog]:
    result = await db.execute(
        select(ProgressLog)
        .where(ProgressLog.goal_id == goal_id)
        .order_by(ProgressLog.logged_at)
    )
    logs = list(result.scalars().all())
    logger.debug("list_progress goal_id={} n={}", goal_id, len(logs))
    return logs

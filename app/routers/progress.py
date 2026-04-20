"""Progress ingestion endpoints."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
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
        await apply_transition(goal, traj.pace_score, db)

    await db.commit()
    await db.refresh(log)
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
    return list(result.scalars().all())

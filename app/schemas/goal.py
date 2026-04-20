"""Goal request/response schemas."""
from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.enums import GoalState, GoalType


class GoalCreate(BaseModel):
    user_id: UUID
    goal_type: GoalType
    start_value: float
    target_value: float
    unit: str
    start_date: date
    target_date: date


class GoalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    goal_type: GoalType
    start_value: float
    target_value: float
    unit: str
    start_date: date
    target_date: date
    current_state: GoalState
    created_at: datetime


class GoalStateEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    goal_id: UUID
    from_state: GoalState
    to_state: GoalState
    pace_score: float
    occurred_at: datetime
    reason: str

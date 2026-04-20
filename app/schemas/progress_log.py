"""Progress log request/response schemas."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ProgressLogCreate(BaseModel):
    goal_id: UUID
    logged_at: datetime
    value: float
    notes: str | None = None


class ProgressLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    goal_id: UUID
    logged_at: datetime
    value: float
    notes: str | None

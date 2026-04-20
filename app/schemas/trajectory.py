"""Trajectory (pace score + ETA) response schema."""
from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel


class TrajectoryRead(BaseModel):
    goal_id: UUID
    pace_score: float
    eta_date: date | None
    days_ahead: int
    computed_at: datetime

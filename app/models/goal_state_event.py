"""Goal state transition audit trail."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import GoalState

if TYPE_CHECKING:
    from app.models.goal import Goal


class GoalStateEvent(Base):
    __tablename__ = "goal_state_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_state: Mapped[GoalState] = mapped_column(
        Enum(GoalState, name="goalstate", native_enum=True, create_type=False),
        nullable=False,
    )
    to_state: Mapped[GoalState] = mapped_column(
        Enum(GoalState, name="goalstate", native_enum=True, create_type=False),
        nullable=False,
    )
    pace_score: Mapped[float] = mapped_column(Float, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    goal: Mapped["Goal"] = relationship("Goal", back_populates="state_events")

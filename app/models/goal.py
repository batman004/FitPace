"""Goal ORM model."""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import GoalState, GoalType

if TYPE_CHECKING:
    from app.models.goal_state_event import GoalStateEvent
    from app.models.progress_log import ProgressLog
    from app.models.user import User


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    goal_type: Mapped[GoalType] = mapped_column(
        Enum(GoalType, name="goaltype", native_enum=True, create_type=False),
        nullable=False,
    )
    start_value: Mapped[float] = mapped_column(Float, nullable=False)
    target_value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    current_state: Mapped[GoalState] = mapped_column(
        Enum(GoalState, name="goalstate", native_enum=True, create_type=False),
        nullable=False,
        default=GoalState.ON_TRACK,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="goals")
    progress_logs: Mapped[list["ProgressLog"]] = relationship(
        "ProgressLog",
        back_populates="goal",
        cascade="all, delete-orphan",
        order_by="ProgressLog.logged_at",
    )
    state_events: Mapped[list["GoalStateEvent"]] = relationship(
        "GoalStateEvent",
        back_populates="goal",
        cascade="all, delete-orphan",
        order_by="GoalStateEvent.occurred_at",
    )

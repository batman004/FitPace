"""initial schema: users, goals, progress_logs, goal_state_events

Revision ID: 001_initial
Revises:
Create Date: 2026-04-20

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    goaltype = postgresql.ENUM(
        "weight_loss",
        "strength_gain",
        "step_goal",
        name="goaltype",
        create_type=True,
    )
    goalstate = postgresql.ENUM(
        "ON_TRACK",
        "AT_RISK",
        "OFF_TRACK",
        "RECOVERED",
        name="goalstate",
        create_type=True,
    )
    bind = op.get_bind()
    goaltype.create(bind, checkfirst=True)
    goalstate.create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "goals",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("goal_type", goaltype, nullable=False),
        sa.Column("start_value", sa.Float(), nullable=False),
        sa.Column("target_value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("current_state", goalstate, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_goals_user_id"), "goals", ["user_id"], unique=False)
    op.create_table(
        "progress_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("logged_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["goal_id"], ["goals.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_progress_logs_goal_id"), "progress_logs", ["goal_id"], unique=False
    )
    op.create_index(
        op.f("ix_progress_logs_logged_at"), "progress_logs", ["logged_at"], unique=False
    )
    op.create_table(
        "goal_state_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("from_state", goalstate, nullable=False),
        sa.Column("to_state", goalstate, nullable=False),
        sa.Column("pace_score", sa.Float(), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["goals.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_goal_state_events_goal_id"),
        "goal_state_events",
        ["goal_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_goal_state_events_goal_id"), table_name="goal_state_events")
    op.drop_table("goal_state_events")
    op.drop_index(op.f("ix_progress_logs_logged_at"), table_name="progress_logs")
    op.drop_index(op.f("ix_progress_logs_goal_id"), table_name="progress_logs")
    op.drop_table("progress_logs")
    op.drop_index(op.f("ix_goals_user_id"), table_name="goals")
    op.drop_table("goals")
    op.drop_table("users")

    goalstate = postgresql.ENUM(name="goalstate", create_type=False)
    goalstate.drop(op.get_bind(), checkfirst=True)
    goaltype = postgresql.ENUM(name="goaltype", create_type=False)
    goaltype.drop(op.get_bind(), checkfirst=True)

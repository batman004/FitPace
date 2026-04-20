"""user profile fields + password hash

Revision ID: 002_user_profile
Revises: 001_initial
Create Date: 2026-04-20

Splits the single `name` column into `first_name` / `last_name`, adds a required
`password_hash`, and optional fitness profile columns (DOB, height, weight, sex).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002_user_profile"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    sex_enum = postgresql.ENUM(
        "male",
        "female",
        "other",
        "prefer_not_to_say",
        name="sex",
        create_type=True,
    )
    sex_enum.create(op.get_bind(), checkfirst=True)

    # Any pre-existing rows won't have first/last/password_hash; the assignment
    # app has no production data yet so we drop & recreate columns rather than
    # backfilling. Adjust this migration if real data exists.
    op.drop_column("users", "name")
    op.add_column(
        "users",
        sa.Column("first_name", sa.String(length=100), nullable=False, server_default=""),
    )
    op.add_column(
        "users",
        sa.Column("last_name", sa.String(length=100), nullable=False, server_default=""),
    )
    op.add_column(
        "users",
        sa.Column("password_hash", sa.String(length=255), nullable=False, server_default=""),
    )
    op.add_column(
        "users", sa.Column("date_of_birth", sa.Date(), nullable=True)
    )
    op.add_column("users", sa.Column("height_cm", sa.Float(), nullable=True))
    op.add_column("users", sa.Column("weight_kg", sa.Float(), nullable=True))
    op.add_column("users", sa.Column("sex", sex_enum, nullable=True))

    # Drop server_defaults now that the columns exist (they were only needed
    # to satisfy NOT NULL against any historical rows).
    op.alter_column("users", "first_name", server_default=None)
    op.alter_column("users", "last_name", server_default=None)
    op.alter_column("users", "password_hash", server_default=None)


def downgrade() -> None:
    op.drop_column("users", "sex")
    op.drop_column("users", "weight_kg")
    op.drop_column("users", "height_cm")
    op.drop_column("users", "date_of_birth")
    op.drop_column("users", "password_hash")
    op.drop_column("users", "last_name")
    op.drop_column("users", "first_name")
    op.add_column(
        "users",
        sa.Column("name", sa.String(length=255), nullable=False, server_default=""),
    )
    op.alter_column("users", "name", server_default=None)

    sex_enum = postgresql.ENUM(name="sex", create_type=False)
    sex_enum.drop(op.get_bind(), checkfirst=True)

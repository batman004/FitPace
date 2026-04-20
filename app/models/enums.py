"""Shared PostgreSQL-backed enum types for ORM models."""
from __future__ import annotations

from enum import Enum


class GoalType(str, Enum):
    weight_loss = "weight_loss"
    strength_gain = "strength_gain"
    step_goal = "step_goal"


class GoalUnit(str, Enum):
    """Measurement units for a goal. Kept as a plain string in the DB (VARCHAR)
    but exposed as an enum in API schemas so Swagger renders a dropdown."""

    kg = "kg"
    kg_1rm = "kg_1rm"
    steps = "steps"


class GoalState(str, Enum):
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    OFF_TRACK = "OFF_TRACK"
    RECOVERED = "RECOVERED"


class Sex(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    prefer_not_to_say = "prefer_not_to_say"

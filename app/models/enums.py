"""Shared PostgreSQL-backed enum types for ORM models."""
from __future__ import annotations

from enum import Enum


class GoalType(str, Enum):
    weight_loss = "weight_loss"
    strength_gain = "strength_gain"
    step_goal = "step_goal"


class GoalState(str, Enum):
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    OFF_TRACK = "OFF_TRACK"
    RECOVERED = "RECOVERED"

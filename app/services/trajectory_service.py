"""Core pacing logic: pace score (via ML) + ETA projection."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import joblib
import numpy as np

from app.ml.features import (
    FEATURE_ORDER,
    age_from_dob,
    build_feature_vector,
    ground_truth_pace_score,
    sex_to_code,
)
from app.models.enums import Sex

MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "model.pkl"

_model_cache: Any = None
_model_mtime: float | None = None


def _get_model() -> Any | None:
    """Lazy-load model.pkl with mtime-based cache invalidation."""
    global _model_cache, _model_mtime
    if not MODEL_PATH.is_file():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _model_cache is None or mtime != _model_mtime:
        _model_cache = joblib.load(MODEL_PATH)
        _model_mtime = mtime
    return _model_cache


@dataclass
class TrajectoryResult:
    goal_id: uuid.UUID
    pace_score: float
    eta_date: date | None
    days_ahead: int
    computed_at: datetime


class _GoalLike(Protocol):
    id: uuid.UUID
    start_value: float
    target_value: float
    start_date: date
    target_date: date


class _LogLike(Protocol):
    logged_at: datetime
    value: float


class _UserLike(Protocol):
    date_of_birth: date | None
    height_cm: float | None
    weight_kg: float | None
    sex: Sex | None


def compute_trajectory(
    goal: _GoalLike,
    progress_logs: Sequence[_LogLike],
    today: date | None = None,
    user: _UserLike | None = None,
) -> TrajectoryResult:
    """Compute pace score + ETA for a goal given its ordered progress logs.

    If `user` is provided, profile fields (age from DOB, height, weight, sex)
    are fed into the model. Missing fields fall back to population defaults.
    """
    now = datetime.now(timezone.utc)
    today = today or now.date()

    if len(progress_logs) < 2:
        return TrajectoryResult(
            goal_id=goal.id,
            pace_score=50.0,
            eta_date=goal.target_date,
            days_ahead=0,
            computed_at=now,
        )

    values = [float(log.value) for log in progress_logs]
    dates = [
        log.logged_at.date() if hasattr(log.logged_at, "date") else log.logged_at
        for log in progress_logs
    ]

    if user is not None:
        user_age = age_from_dob(user.date_of_birth, today)
        user_height_cm = user.height_cm
        user_weight_kg = user.weight_kg
        user_sex_code = sex_to_code(user.sex)
    else:
        user_age = user_height_cm = user_weight_kg = user_sex_code = None

    feats = build_feature_vector(
        values=values,
        logged_dates=dates,
        start_value=goal.start_value,
        target_value=goal.target_value,
        start_date=goal.start_date,
        target_date=goal.target_date,
        today=today,
        user_age=user_age,
        user_height_cm=user_height_cm,
        user_weight_kg=user_weight_kg,
        user_sex_code=user_sex_code,
    )

    model = _get_model()
    if model is None:
        total_days = max(1, (goal.target_date - goal.start_date).days)
        pace_score = ground_truth_pace_score(
            feats["rolling_7d_slope"], goal.start_value, goal.target_value, total_days
        )
    else:
        vec = np.array([[feats[k] for k in FEATURE_ORDER]])
        pace_score = float(np.clip(model.predict(vec)[0], 0.0, 100.0))

    slope = feats["rolling_7d_slope"]
    remaining_delta = goal.target_value - feats["current_value"]
    eta_date: date | None
    if remaining_delta == 0:
        eta_date = today
    elif slope == 0 or slope * remaining_delta < 0:
        eta_date = None
    else:
        days_needed = remaining_delta / slope
        eta_date = today + timedelta(days=int(round(days_needed)))

    days_ahead = (goal.target_date - eta_date).days if eta_date is not None else 0

    return TrajectoryResult(
        goal_id=goal.id,
        pace_score=round(pace_score, 2),
        eta_date=eta_date,
        days_ahead=days_ahead,
        computed_at=now,
    )

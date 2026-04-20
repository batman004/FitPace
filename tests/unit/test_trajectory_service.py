"""Tests for trajectory_service.compute_trajectory."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from app.ml.features import (
    DEFAULT_AGE,
    DEFAULT_HEIGHT_CM,
    DEFAULT_SEX_CODE,
    DEFAULT_WEIGHT_KG,
    age_from_dob,
    build_feature_vector,
    sex_to_code,
)
from app.models.enums import Sex
from app.services.trajectory_service import compute_trajectory


@dataclass
class _FakeUser:
    date_of_birth: date | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    sex: Sex | None = None


@dataclass
class _FakeGoal:
    id: uuid.UUID
    start_value: float
    target_value: float
    start_date: date
    target_date: date


@dataclass
class _FakeLog:
    logged_at: datetime
    value: float


def _make_goal(start: float, target: float, total_days: int) -> _FakeGoal:
    start_date = date(2026, 1, 1)
    return _FakeGoal(
        id=uuid.uuid4(),
        start_value=start,
        target_value=target,
        start_date=start_date,
        target_date=start_date + timedelta(days=total_days),
    )


def _on_pace_logs(goal: _FakeGoal, n: int) -> list[_FakeLog]:
    total_days = (goal.target_date - goal.start_date).days
    logs: list[_FakeLog] = []
    for i in range(n):
        logged_at = datetime.combine(
            goal.start_date + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc
        )
        value = goal.start_value + (goal.target_value - goal.start_value) * (i / total_days)
        logs.append(_FakeLog(logged_at=logged_at, value=value))
    return logs


def test_less_than_two_logs_returns_default_50() -> None:
    goal = _make_goal(85.0, 80.0, 60)
    result = compute_trajectory(goal, [], today=goal.start_date)
    assert result.pace_score == 50.0
    assert result.eta_date == goal.target_date


def test_on_pace_scores_high() -> None:
    goal = _make_goal(85.0, 80.0, 60)
    on_pace_logs = _on_pace_logs(goal, 10)

    flat_goal = _make_goal(85.0, 80.0, 60)
    flat_logs = [
        _FakeLog(
            logged_at=datetime.combine(
                flat_goal.start_date + timedelta(days=i),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ),
            value=85.0,
        )
        for i in range(10)
    ]

    on_pace_score = compute_trajectory(
        goal, on_pace_logs, today=goal.start_date + timedelta(days=9)
    ).pace_score
    flat_score = compute_trajectory(
        flat_goal, flat_logs, today=flat_goal.start_date + timedelta(days=9)
    ).pace_score

    # The model's absolute calibration has MAE~20; the useful contract is
    # *directional*: on-pace logs score materially higher than a plateau.
    assert on_pace_score > 50.0
    assert flat_score < 50.0
    assert on_pace_score - flat_score >= 20.0


def test_plateau_has_no_eta() -> None:
    goal = _make_goal(85.0, 80.0, 60)
    logs = [
        _FakeLog(
            logged_at=datetime.combine(
                goal.start_date + timedelta(days=i),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ),
            value=85.0,
        )
        for i in range(10)
    ]
    result = compute_trajectory(goal, logs, today=goal.start_date + timedelta(days=9))
    assert result.eta_date is None


def test_eta_none_when_slope_opposes_target() -> None:
    goal = _make_goal(85.0, 80.0, 60)
    logs = [
        _FakeLog(
            logged_at=datetime.combine(
                goal.start_date + timedelta(days=i),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ),
            value=85.0 + i * 0.2,  # going up when we need to go down
        )
        for i in range(7)
    ]
    result = compute_trajectory(goal, logs, today=goal.start_date + timedelta(days=6))
    assert result.eta_date is None


def test_feature_vector_with_only_two_logs() -> None:
    feats = build_feature_vector(
        values=[85.0, 84.8],
        logged_dates=[date(2026, 1, 1), date(2026, 1, 2)],
        start_value=85.0,
        target_value=80.0,
        start_date=date(2026, 1, 1),
        target_date=date(2026, 3, 2),
        today=date(2026, 1, 2),
    )
    assert feats["rolling_7d_slope"] < 0
    assert feats["current_value"] == 84.8


def test_feature_vector_defaults_when_user_missing() -> None:
    feats = build_feature_vector(
        values=[85.0, 84.8],
        logged_dates=[date(2026, 1, 1), date(2026, 1, 2)],
        start_value=85.0,
        target_value=80.0,
        start_date=date(2026, 1, 1),
        target_date=date(2026, 3, 2),
        today=date(2026, 1, 2),
    )
    assert feats["user_age"] == DEFAULT_AGE
    assert feats["user_height_cm"] == DEFAULT_HEIGHT_CM
    assert feats["user_weight_kg"] == DEFAULT_WEIGHT_KG
    assert feats["user_sex_code"] == float(DEFAULT_SEX_CODE)


def test_feature_vector_uses_user_profile() -> None:
    feats = build_feature_vector(
        values=[85.0, 84.8],
        logged_dates=[date(2026, 1, 1), date(2026, 1, 2)],
        start_value=85.0,
        target_value=80.0,
        start_date=date(2026, 1, 1),
        target_date=date(2026, 3, 2),
        today=date(2026, 1, 2),
        user_age=42.0,
        user_height_cm=180.0,
        user_weight_kg=88.0,
        user_sex_code=0,
    )
    assert feats["user_age"] == 42.0
    assert feats["user_height_cm"] == 180.0
    assert feats["user_weight_kg"] == 88.0
    assert feats["user_sex_code"] == 0.0


def test_age_from_dob_and_sex_to_code() -> None:
    assert age_from_dob(None, date(2026, 1, 1)) is None
    assert age_from_dob(date(1990, 1, 1), date(2026, 1, 1)) == (date(2026, 1, 1) - date(1990, 1, 1)).days / 365.25
    assert sex_to_code(None) is None
    assert sex_to_code(Sex.male) == 0
    assert sex_to_code(Sex.female) == 1
    assert sex_to_code(Sex.other) == 2
    assert sex_to_code(Sex.prefer_not_to_say) == 2


def test_compute_trajectory_accepts_user() -> None:
    goal = _make_goal(85.0, 80.0, 60)
    logs = _on_pace_logs(goal, 10)
    user = _FakeUser(
        date_of_birth=date(1990, 6, 1),
        height_cm=178.0,
        weight_kg=85.0,
        sex=Sex.male,
    )
    result = compute_trajectory(
        goal, logs, today=goal.start_date + timedelta(days=9), user=user
    )
    # Should still score directionally high when the logs are on pace.
    assert result.pace_score > 50.0


def test_feature_vector_with_many_logs_only_uses_last_seven() -> None:
    values = [85.0 - 0.1 * i for i in range(14)]
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(14)]
    feats = build_feature_vector(
        values=values,
        logged_dates=dates,
        start_value=85.0,
        target_value=80.0,
        start_date=date(2026, 1, 1),
        target_date=date(2026, 3, 2),
        today=dates[-1],
    )
    assert feats["rolling_7d_avg"] < 85.0
    assert abs(feats["rolling_7d_slope"] - (-0.1)) < 1e-9

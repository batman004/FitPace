"""Feature engineering shared between training and live inference."""
from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np

FEATURE_ORDER: tuple[str, ...] = (
    "days_elapsed",
    "current_value",
    "rolling_7d_avg",
    "rolling_7d_slope",
    "slope_ratio",
    "pct_progress",
    "days_remaining",
)


def build_feature_vector(
    values: Sequence[float],
    logged_dates: Sequence[date],
    start_value: float,
    target_value: float,
    start_date: date,
    target_date: date,
    today: date,
) -> dict[str, float]:
    """Return a feature dict for a single (goal, today) evaluation point.

    Assumes `values` and `logged_dates` are sorted ascending and have length >= 1.
    Slope is expressed in units-per-day over the last up-to-7 samples.
    """
    if len(values) == 0:
        raise ValueError("values must contain at least one log")

    n = min(7, len(values))
    recent = np.asarray(values[-n:], dtype=float)

    if n >= 2:
        slope, _ = np.polyfit(np.arange(n, dtype=float), recent, 1)
        rolling_7d_slope = float(slope)
    else:
        rolling_7d_slope = 0.0

    total_delta = target_value - start_value
    current_value = float(values[-1])
    if total_delta != 0:
        pct = (current_value - start_value) / total_delta
    else:
        pct = 1.0
    pct_progress = float(min(1.0, max(0.0, pct)))

    total_days = max(1, (target_date - start_date).days)
    required_rate = total_delta / total_days
    if required_rate != 0:
        slope_ratio = rolling_7d_slope / required_rate
    else:
        slope_ratio = 1.0

    return {
        "days_elapsed": float(max(0, (today - start_date).days)),
        "current_value": current_value,
        "rolling_7d_avg": float(np.mean(recent)),
        "rolling_7d_slope": rolling_7d_slope,
        "slope_ratio": float(slope_ratio),
        "pct_progress": pct_progress,
        "days_remaining": float(max(0, (target_date - today).days)),
    }


def ground_truth_pace_score(
    rolling_7d_slope: float,
    start_value: float,
    target_value: float,
    total_days: int,
) -> float:
    """Label used during training; also serves as a fallback when model.pkl is absent."""
    total_delta = target_value - start_value
    if total_days <= 0 or total_delta == 0:
        return 100.0
    required_rate = total_delta / total_days
    ratio = rolling_7d_slope / required_rate
    return float(np.clip(100.0 * ratio, 0.0, 100.0))

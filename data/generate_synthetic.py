"""Generate synthetic goal progress and emit a CSV of training rows.

Usage:
    python data/generate_synthetic.py
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from app.ml.features import FEATURE_ORDER, build_feature_vector, ground_truth_pace_score

ARCHETYPES: dict[str, dict[str, float | str | int]] = {
    "weight_loss": {
        "start": 85.0,
        "target": 80.0,
        "unit": "kg",
        "days": 60,
        "noise": 0.5,
    },
    "strength_gain": {
        "start": 60.0,
        "target": 80.0,
        "unit": "kg_1rm",
        "days": 90,
        "noise": 1.0,
    },
    "step_goal": {
        "start": 5000.0,
        "target": 8000.0,
        "unit": "steps",
        "days": 30,
        "noise": 300.0,
    },
}

CSV_PATH = Path(__file__).resolve().parent / "synthetic_progress.csv"


def _simulate_values(
    start: float,
    target: float,
    total_days: int,
    noise_std: float,
    rng: random.Random,
    progress_factor: float = 1.0,
) -> list[float]:
    """`progress_factor` scales the user's actual pace relative to the required
    linear rate. 1.0 = perfectly on pace; <1 = falling behind; >1 = ahead."""
    plateau_start = rng.randint(5, max(6, total_days - 10))
    plateau_len = rng.randint(5, 10)
    plateau_end = plateau_start + plateau_len

    values: list[float] = []
    plateau_anchor: float | None = None
    for day in range(total_days + 1):
        if plateau_start <= day < plateau_end and plateau_anchor is not None:
            v = plateau_anchor + rng.gauss(0.0, noise_std * 0.2)
        else:
            fraction = progress_factor * (day / total_days)
            ideal = start + (target - start) * fraction
            v = ideal + rng.gauss(0.0, noise_std)
            plateau_anchor = v
        values.append(v)
    return values


def _sample_user_profile(
    rng: random.Random, arch_name: str, start_value: float
) -> dict[str, float]:
    """Sample a plausible demographic profile. Older users make slower
    progress, giving the model a signal to pick up."""
    age = rng.randint(18, 65)
    sex_code = rng.choice([0, 1, 2])
    height_cm = max(140.0, rng.gauss(172.0 if sex_code == 0 else 162.0, 7.0))

    if arch_name == "weight_loss":
        weight_kg = start_value + rng.gauss(0.0, 4.0)
    elif arch_name == "strength_gain":
        weight_kg = rng.gauss(72.0 if sex_code == 0 else 63.0, 10.0)
    else:
        weight_kg = rng.gauss(72.0 if sex_code == 0 else 63.0, 12.0)
    weight_kg = max(40.0, weight_kg)

    return {
        "user_age": float(age),
        "user_height_cm": float(height_cm),
        "user_weight_kg": float(weight_kg),
        "user_sex_code": float(sex_code),
    }


def _rows_for_goal(
    arch_name: str, arch: dict[str, float | str | int], rng: random.Random
) -> list[dict[str, float]]:
    start_value = float(arch["start"])
    target_value = float(arch["target"])
    total_days = int(arch["days"])
    noise = float(arch["noise"])

    start_date = date(2026, 1, 1)
    target_date = start_date + timedelta(days=total_days)

    profile = _sample_user_profile(rng, arch_name, start_value)

    # Older users make slightly slower progress; this gives demographic
    # features real predictive value over and above the slope itself.
    age_penalty = max(0.0, profile["user_age"] - 30.0) * 0.006
    base_factor = 1.0 - age_penalty
    progress_factor = max(0.5, base_factor + rng.gauss(0.0, 0.15))

    values = _simulate_values(
        start_value, target_value, total_days, noise, rng, progress_factor
    )

    rows: list[dict[str, float]] = []
    for day in range(2, total_days + 1):
        window_values = values[: day + 1]
        window_dates = [start_date + timedelta(days=i) for i in range(len(window_values))]
        today = start_date + timedelta(days=day)

        feats = build_feature_vector(
            values=window_values,
            logged_dates=window_dates,
            start_value=start_value,
            target_value=target_value,
            start_date=start_date,
            target_date=target_date,
            today=today,
            user_age=profile["user_age"],
            user_height_cm=profile["user_height_cm"],
            user_weight_kg=profile["user_weight_kg"],
            user_sex_code=int(profile["user_sex_code"]),
        )
        pace = ground_truth_pace_score(
            rolling_7d_slope=feats["rolling_7d_slope"],
            start_value=start_value,
            target_value=target_value,
            total_days=total_days,
        )
        rows.append({**feats, "pace_score": pace})
    return rows


def main(n_users: int = 200, seed: int = 42, csv_path: Path = CSV_PATH) -> Path:
    rng = random.Random(seed)
    archetype_names = list(ARCHETYPES)

    all_rows: list[dict[str, float]] = []
    for idx in range(n_users):
        arch_name = archetype_names[idx % len(archetype_names)]
        all_rows.extend(_rows_for_goal(arch_name, ARCHETYPES[arch_name], rng))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FEATURE_ORDER) + ["pace_score"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows from {n_users} goals to {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()

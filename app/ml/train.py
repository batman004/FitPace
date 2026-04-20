"""Train a LinearRegression model on synthetic_progress.csv and save model.pkl.

Usage:
    python app/ml/train.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.features import FEATURE_ORDER

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic_progress.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def main(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> Pipeline:
    df = pd.read_csv(data_path)
    X = df[list(FEATURE_ORDER)].to_numpy()
    y = df["pace_score"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # StandardScaler handles wildly different feature scales across the three
    # goal archetypes (kg vs steps); LinearRegression alone would underflow.
    model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"R2={r2:.4f} MAE={mae:.4f} n_train={len(X_train)} n_test={len(X_test)}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    return model


if __name__ == "__main__":
    main()

"""Fit, save, and load the rate model."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from freight.config import SEED
from freight.metrics import clip_rate

GRID = [
    {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40},
    {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 40},
    {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20},
    {"learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 50},
]


def _as_lgb(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in categorical:
        out[col] = out[col].astype("category")
    return out


def predict_log_model(booster: lgb.LGBMRegressor, features: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    log_hat = booster.predict(_as_lgb(features, categorical))
    return clip_rate(np.expm1(log_hat))


def fit_one(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    categorical: list[str],
    extra: dict,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="mae",
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        **extra,
    )
    model.fit(
        _as_lgb(X_train, categorical),
        np.log1p(y_train.to_numpy(dtype=float)),
        eval_set=[
            (
                _as_lgb(X_holdout, categorical),
                np.log1p(y_holdout.to_numpy(dtype=float)),
            )
        ],
        eval_metric="l1",
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def importance_table(model: lgb.LGBMRegressor, names: list[str], top: int = 12) -> list[dict]:
    gain = model.booster_.feature_importance(importance_type="gain")
    rows = sorted(zip(names, gain), key=lambda kv: kv[1], reverse=True)[:top]
    return [{"feature": name, "gain": float(val)} for name, val in rows]


def save_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: Path) -> dict:
    return joblib.load(path)

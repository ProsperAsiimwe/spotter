"""Fit, save, and load the rate model."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from freight.config import SEED
from freight.metrics import clip_rate
from freight.progress import tree_bar

# Time-split grid. Not sklearn GridSearchCV (that would shuffle / k-fold).
# colsample_bytree / subsample are the gbdt version of dropout.
# DART drop_rate was tried separately; holdout MAE went to ~2400 so it is not in the grid.
GBDT_GRID = {
    "boosting_type": ["gbdt"],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [31, 63],
    "min_child_samples": [20, 40],
    "colsample_bytree": [0.7, 0.9],
    "subsample": [0.7, 0.9],
    "reg_lambda": [1.0],
}

GRID = list(ParameterGrid(GBDT_GRID))


def _as_lgb(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in categorical:
        out[col] = out[col].astype("category")
    return out


def _estimator(extra: dict, n_estimators: int) -> lgb.LGBMRegressor:
    params = {
        "objective": "mae",
        "n_estimators": int(n_estimators),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "subsample_freq": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(extra)
    if float(params.get("subsample", 1.0)) >= 1.0:
        params["subsample_freq"] = 0
    return lgb.LGBMRegressor(**params)


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
    dart = extra.get("boosting_type") == "dart"
    n_estimators = 400 if dart else 2000
    model = _estimator(extra, n_estimators)
    y_tr = np.log1p(y_train.to_numpy(dtype=float))
    y_ho = np.log1p(y_holdout.to_numpy(dtype=float))
    cb, pbar = tree_bar(n_estimators, desc="trees")
    callbacks = [cb]
    if not dart:
        callbacks.insert(0, lgb.early_stopping(50, verbose=False))
    try:
        model.fit(
            _as_lgb(X_train, categorical),
            y_tr,
            eval_set=[(_as_lgb(X_holdout, categorical), y_ho)],
            eval_metric="l1",
            categorical_feature=categorical,
            callbacks=callbacks,
        )
    finally:
        pbar.close()
    return model


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    extra: dict,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    model = _estimator(extra, n_estimators)
    cb, pbar = tree_bar(n_estimators, desc="final trees")
    try:
        model.fit(
            _as_lgb(X, categorical),
            np.log1p(y.to_numpy(dtype=float)),
            categorical_feature=categorical,
            callbacks=[cb],
        )
    finally:
        pbar.close()
    return model


def predict_loads(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    builder = artifact["builder"]
    features = builder.transform(frame)
    return predict_log_model(artifact["booster"], features, artifact["categorical"])


def importance_table(model: lgb.LGBMRegressor, names: list[str], top: int = 12) -> list[dict]:
    gain = model.booster_.feature_importance(importance_type="gain")
    rows = sorted(zip(names, gain), key=lambda kv: kv[1], reverse=True)[:top]
    return [{"feature": name, "gain": float(val)} for name, val in rows]


def save_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: Path) -> dict:
    return joblib.load(path)

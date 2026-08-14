"""Boosted trees. LightGBM, XGBoost, and CatBoost share the same time split."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from freight.config import SEED
from freight.metrics import clip_rate
from freight.paths import MODEL_ARCHIVES, MODEL_ARTIFACT, MODEL_CURRENT_JSON
from freight.progress import log, tree_bar

FAMILIES = ("lightgbm", "xgboost", "catboost")

# Same split for every family. Not k-fold.
GRIDS = {
    "lightgbm": list(
        ParameterGrid(
            {
                "learning_rate": [0.03, 0.05],
                "num_leaves": [31, 63],
                "min_child_samples": [20, 40],
                "colsample_bytree": [0.9],
                "subsample": [0.7],
                "reg_lambda": [1.0],
            }
        )
    ),
    "xgboost": list(
        ParameterGrid(
            {
                "learning_rate": [0.03, 0.05],
                "max_depth": [6, 8],
                "min_child_weight": [20, 40],
                "colsample_bytree": [0.9],
                "subsample": [0.7],
                "reg_lambda": [1.0],
            }
        )
    ),
    "catboost": list(
        ParameterGrid(
            {
                "learning_rate": [0.03, 0.05],
                "depth": [6, 8],
                "l2_leaf_reg": [1, 3],
                "rsm": [0.9],
                "subsample": [0.7],
            }
        )
    ),
}


def _as_category(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in categorical:
        out[col] = out[col].astype("category")
    return out


def _log_y(y: pd.Series) -> np.ndarray:
    return np.log1p(y.to_numpy(dtype=float))


def n_trees(family: str, model) -> int:
    if family == "lightgbm":
        return int(getattr(model, "best_iteration_", 0) or model.n_estimators_)
    if family == "xgboost":
        best = getattr(model, "best_iteration", None)
        if best is not None:
            return int(best) + 1
        return int(model.n_estimators)
    best = model.get_best_iteration()
    if best is not None and best >= 0:
        return int(best) + 1
    return int(model.tree_count_)


def fit_one(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    categorical: list[str],
    extra: dict,
):
    y_tr = _log_y(y_train)
    y_ho = _log_y(y_holdout)
    if family == "lightgbm":
        return _fit_lgb(X_train, y_tr, X_holdout, y_ho, categorical, extra, 2000, True)
    if family == "xgboost":
        return _fit_xgb(X_train, y_tr, X_holdout, y_ho, categorical, extra, 2000, True)
    if family == "catboost":
        return _fit_cat(X_train, y_tr, X_holdout, y_ho, categorical, extra, 2000, True)
    raise ValueError(f"unknown family {family}")


def fit_final(
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    extra: dict,
    n_estimators: int,
):
    y_log = _log_y(y)
    if family == "lightgbm":
        return _fit_lgb(X, y_log, None, None, categorical, extra, n_estimators, False)
    if family == "xgboost":
        return _fit_xgb(X, y_log, None, None, categorical, extra, n_estimators, False)
    if family == "catboost":
        return _fit_cat(X, y_log, None, None, categorical, extra, n_estimators, False)
    raise ValueError(f"unknown family {family}")


def _fit_lgb(X, y, X_ho, y_ho, categorical, extra, n_estimators, early):
    params = {
        "objective": "mae",
        "n_estimators": int(n_estimators),
        "subsample_freq": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    params.update(extra)
    if float(params.get("subsample", 1.0)) >= 1.0:
        params["subsample_freq"] = 0
    model = lgb.LGBMRegressor(**params)
    X_tr = _as_category(X, categorical)
    cb, pbar = tree_bar(n_estimators, desc="lgbm")
    fit_kw = {
        "categorical_feature": categorical,
        "callbacks": [cb],
    }
    if early and X_ho is not None:
        fit_kw["eval_set"] = [(_as_category(X_ho, categorical), y_ho)]
        fit_kw["eval_metric"] = "l1"
        fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False), cb]
    try:
        model.fit(X_tr, y, **fit_kw)
    finally:
        pbar.close()
    return model


def _fit_xgb(X, y, X_ho, y_ho, categorical, extra, n_estimators, early):
    import xgboost as xgb

    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": int(n_estimators),
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }
    params.update(extra)
    if early:
        params["early_stopping_rounds"] = 50
    model = xgb.XGBRegressor(**params)
    X_tr = _as_category(X, categorical)
    fit_kw = {"verbose": False}
    if early and X_ho is not None:
        fit_kw["eval_set"] = [(_as_category(X_ho, categorical), y_ho)]
    model.fit(X_tr, y, **fit_kw)
    return model


def _fit_cat(X, y, X_ho, y_ho, categorical, extra, n_estimators, early):
    from catboost import CatBoostRegressor

    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": int(n_estimators),
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        "thread_count": -1,
    }
    params.update(extra)
    if early:
        params["early_stopping_rounds"] = 50
    model = CatBoostRegressor(**params)
    cat_idx = [list(X.columns).index(c) for c in categorical]
    fit_kw = {"cat_features": cat_idx}
    if early and X_ho is not None:
        fit_kw["eval_set"] = (X_ho, y_ho)
        fit_kw["use_best_model"] = True
    model.fit(X, y, **fit_kw)
    return model


def predict_matrix(booster, features: pd.DataFrame, categorical: list[str], family: str) -> np.ndarray:
    if family == "lightgbm":
        log_hat = booster.predict(_as_category(features, categorical))
    elif family == "xgboost":
        log_hat = booster.predict(_as_category(features, categorical))
    elif family == "catboost":
        log_hat = booster.predict(features)
    else:
        raise ValueError(f"unknown family {family}")
    return clip_rate(np.expm1(np.asarray(log_hat, dtype=float)))


def predict_loads(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    builder = artifact["builder"]
    features = builder.transform(frame)
    family = artifact.get("family", "lightgbm")
    return predict_matrix(artifact["booster"], features, artifact["categorical"], family)


def importance_table(family: str, model, names: list[str], top: int = 12) -> list[dict]:
    if family == "lightgbm":
        gain = model.booster_.feature_importance(importance_type="gain")
    elif family == "xgboost":
        gain = np.asarray(model.feature_importances_, dtype=float)
    else:
        gain = np.asarray(model.get_feature_importance(), dtype=float)
    rows = sorted(zip(names, gain), key=lambda kv: kv[1], reverse=True)[:top]
    return [{"feature": name, "gain": float(val)} for name, val in rows]


def metadata(payload: dict) -> dict:
    skip = {"builder", "booster"}
    return {k: v for k, v in payload.items() if k not in skip}


def archive_filename(payload: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    family = payload.get("family")
    if not family:
        boosting = (payload.get("params") or {}).get("boosting_type")
        family = "lightgbm" if boosting in (None, "gbdt") else str(boosting)
    mae = payload.get("holdout_mae")
    trees = payload.get("n_estimators", "na")
    seed = payload.get("seed", SEED)
    mae_tag = f"mae{mae:.2f}".replace(".", "p") if isinstance(mae, (int, float)) else "maeNA"
    return f"{stamp}_{family}_{mae_tag}_trees{trees}_seed{seed}.joblib"


def save_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: Path) -> dict:
    return joblib.load(path)


def install_current(payload: dict) -> Path | None:
    """Write the winner to models/rate_model.joblib. Previous file goes to archives/."""
    MODEL_ARCHIVES.mkdir(parents=True, exist_ok=True)
    archived = None
    if MODEL_ARTIFACT.exists():
        try:
            old = load_artifact(MODEL_ARTIFACT)
        except Exception:
            old = {"family": "unknown"}
        name = archive_filename(old)
        archived = MODEL_ARCHIVES / name
        shutil.move(str(MODEL_ARTIFACT), str(archived))
        (archived.with_suffix(".json")).write_text(json.dumps(metadata(old), indent=2) + "\n")
        log(f"[train] archived {archived.name}")
    save_artifact(MODEL_ARTIFACT, payload)
    MODEL_CURRENT_JSON.write_text(json.dumps(metadata(payload), indent=2) + "\n")
    log(f"[train] current family={payload.get('family')}  holdout MAE={payload.get('holdout_mae')}")
    return archived

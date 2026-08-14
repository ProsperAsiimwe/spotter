"""Holdout baselines. LightGBM has to beat these before it ships."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_COLS = [
    "distance",
    "log_distance",
    "weight",
    "market_index",
    "quote_signal",
    "days_since_start",
]


class EquipmentRpmBaseline:
    """posted_rate ≈ distance * median $/mile for that equipment."""

    def __init__(self) -> None:
        self.rpm_by_eq: dict[str, float] = {}
        self.rpm_global: float = 0.0

    def fit(self, frame: pd.DataFrame) -> EquipmentRpmBaseline:
        rpm = frame["posted_rate"] / frame["distance"]
        self.rpm_by_eq = rpm.groupby(frame["equipment"]).median().to_dict()
        self.rpm_global = float(rpm.median())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rpm = frame["equipment"].map(self.rpm_by_eq).astype("float64")
        rpm = rpm.fillna(self.rpm_global)
        return (rpm * frame["distance"]).to_numpy(dtype=float)


class RidgeBaseline:
    """Ridge on a small numeric set, log1p target. Equipment is one-hot."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipe: Pipeline | None = None
        self.columns: list[str] = []

    def _design(self, features: pd.DataFrame, equipment: pd.Series, fit: bool) -> np.ndarray:
        num = features[RIDGE_COLS].to_numpy(dtype=float)
        dummies = pd.get_dummies(equipment.fillna("unknown"), prefix="eq")
        if fit:
            self.columns = list(dummies.columns)
        else:
            dummies = dummies.reindex(columns=self.columns, fill_value=0)
        return np.hstack([num, dummies.to_numpy(dtype=float)])

    def fit(self, features: pd.DataFrame, equipment: pd.Series, y: pd.Series) -> RidgeBaseline:
        X = self._design(features, equipment, fit=True)
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, solver="cholesky")),
            ]
        )
        self.pipe.fit(X, np.log1p(y.to_numpy(dtype=float)))
        return self

    def predict(self, features: pd.DataFrame, equipment: pd.Series) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("RidgeBaseline.fit() first")
        X = self._design(features, equipment, fit=False)
        return np.expm1(self.pipe.predict(X))

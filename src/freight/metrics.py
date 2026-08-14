from __future__ import annotations

import numpy as np
import pandas as pd

RATE_FLOOR = 0.01


def clip_rate(pred) -> np.ndarray:
    return np.maximum(np.asarray(pred, dtype=float), RATE_FLOOR)


def evaluate(y_true, y_pred, distance) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_rate(y_pred)
    distance = np.asarray(distance, dtype=float)
    err = np.abs(y_true - y_pred)
    rpm_err = np.abs(y_true / distance - y_pred / distance)
    return {
        "mae": float(np.mean(err)),
        "medae": float(np.median(err)),
        "mape": float(np.mean(err / np.clip(np.abs(y_true), RATE_FLOOR, None)) * 100.0),
        "rpm_mae": float(np.mean(rpm_err)),
    }

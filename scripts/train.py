from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.baselines import EquipmentRpmBaseline, RidgeBaseline
from freight.config import SEED, set_seed
from freight.features import FeatureBuilder
from freight.metrics import evaluate
from freight.model import GRID, fit_one, importance_table, predict_log_model, save_artifact
from freight.paths import METRICS_JSON, MODEL_ARTIFACT, TRAIN_TEST_CSV
from freight.split import time_split


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in metrics.items()}


def main() -> None:
    set_seed()
    labeled = pd.read_csv(TRAIN_TEST_CSV, parse_dates=["date"])
    train, holdout = time_split(labeled)
    y_tr = train["posted_rate"]
    y_ho = holdout["posted_rate"]

    builder = FeatureBuilder().fit(train)
    X_tr = builder.transform(train)
    X_ho = builder.transform(holdout)

    rpm = EquipmentRpmBaseline().fit(train)
    ridge = RidgeBaseline().fit(X_tr, train["equipment"], y_tr)
    rpm_ho = evaluate(y_ho, rpm.predict(holdout), holdout["distance"])
    ridge_ho = evaluate(y_ho, ridge.predict(X_ho, holdout["equipment"]), holdout["distance"])

    trials = []
    best = None
    for extra in GRID:
        booster = fit_one(X_tr, y_tr, X_ho, y_ho, builder.categorical, extra)
        pred_ho = predict_log_model(booster, X_ho, builder.categorical)
        pred_tr = predict_log_model(booster, X_tr, builder.categorical)
        ho = evaluate(y_ho, pred_ho, holdout["distance"])
        tr = evaluate(y_tr, pred_tr, train["distance"])
        row = {
            "params": extra,
            "best_iteration": int(getattr(booster, "best_iteration_", 0) or booster.n_estimators_),
            "train": _round(tr),
            "holdout": _round(ho),
        }
        trials.append(row)
        print("lgbm", extra, "holdout mae", round(ho["mae"], 2), "iter", row["best_iteration"])
        if best is None or ho["mae"] < best["holdout_mae"]:
            best = {
                "booster": booster,
                "holdout_mae": ho["mae"],
                "train": tr,
                "holdout": ho,
                "params": extra,
                "best_iteration": row["best_iteration"],
            }

    assert best is not None
    beats_ridge = best["holdout"]["mae"] < ridge_ho["mae"]
    beats_rpm = best["holdout"]["mae"] < rpm_ho["mae"]

    payload = {
        "builder": builder,
        "booster": best["booster"],
        "params": best["params"],
        "seed": SEED,
        "target": "log1p(posted_rate)",
        "objective": "mae",
        "fit_on": "jan_aug_train",
        "categorical": list(builder.categorical),
        "feature_names": list(builder.feature_names),
    }
    save_artifact(MODEL_ARTIFACT, payload)

    report = {
        "split": {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train": "2025-01-01 to 2025-08-31",
            "holdout": "2025-09-01 to 2025-10-31",
            "holdout_used_for_early_stopping_and_light_tune": True,
        },
        "seed": SEED,
        "equipment_rpm_holdout": _round(rpm_ho),
        "ridge_holdout": _round(ridge_ho),
        "lightgbm": {
            "params": best["params"],
            "target": "log1p(posted_rate)",
            "objective": "mae",
            "best_iteration": best["best_iteration"],
            "train": _round(best["train"]),
            "holdout": _round(best["holdout"]),
            "beats_equipment_rpm": beats_rpm,
            "beats_ridge": beats_ridge,
            "importance_gain": importance_table(best["booster"], builder.feature_names),
        },
        "grid": trials,
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print("best", best["params"], "holdout", _round(best["holdout"]))
    print("beats ridge", beats_ridge, "beats rpm", beats_rpm)
    print("wrote", METRICS_JSON)
    print("wrote", MODEL_ARTIFACT)


if __name__ == "__main__":
    main()

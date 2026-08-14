from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.baselines import EquipmentRpmBaseline, RidgeBaseline
from freight.config import SEED, set_seed
from freight.features import FeatureBuilder
from freight.metrics import evaluate
from freight.model import GRID, fit_final, fit_one, importance_table, predict_log_model, save_artifact
from freight.paths import METRICS_JSON, MODEL_ARTIFACT, TRAIN_TEST_CSV
from freight.progress import log
from freight.split import time_split


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in metrics.items()}


def _plain(params: dict) -> dict:
    out = {}
    for key, val in params.items():
        if isinstance(val, float):
            out[key] = float(val)
        elif isinstance(val, (int, bool)):
            out[key] = int(val) if not isinstance(val, bool) else val
        else:
            out[key] = val
    return out


def main() -> None:
    set_seed()
    log("[train] seed=42  loading labeled file")
    labeled = pd.read_csv(TRAIN_TEST_CSV, parse_dates=["date"])
    train, holdout = time_split(labeled)
    log(f"[train] jan-aug {len(train):,}  sep-oct holdout {len(holdout):,}")
    y_tr = train["posted_rate"]
    y_ho = holdout["posted_rate"]

    log("[train] fitting features on jan-aug")
    builder = FeatureBuilder().fit(train)
    X_tr = builder.transform(train)
    X_ho = builder.transform(holdout)

    log("[train] equipment rpm baseline")
    rpm = EquipmentRpmBaseline().fit(train)
    rpm_ho = evaluate(y_ho, rpm.predict(holdout), holdout["distance"])
    log(f"[train] rpm holdout MAE {rpm_ho['mae']:.2f}")

    log("[train] ridge baseline")
    ridge = RidgeBaseline().fit(X_tr, train["equipment"], y_tr)
    ridge_ho = evaluate(y_ho, ridge.predict(X_ho, holdout["equipment"]), holdout["distance"])
    log(f"[train] ridge holdout MAE {ridge_ho['mae']:.2f}")

    log(f"[train] grid search {len(GRID)} configs on sep-oct MAE")
    trials = []
    best = None
    for extra in tqdm(GRID, desc="grid search", unit="cfg"):
        booster = fit_one(X_tr, y_tr, X_ho, y_ho, builder.categorical, extra)
        pred_ho = predict_log_model(booster, X_ho, builder.categorical)
        pred_tr = predict_log_model(booster, X_tr, builder.categorical)
        ho = evaluate(y_ho, pred_ho, holdout["distance"])
        tr = evaluate(y_tr, pred_tr, train["distance"])
        iters = int(getattr(booster, "best_iteration_", 0) or booster.n_estimators_)
        row = {
            "params": _plain(extra),
            "best_iteration": iters,
            "train": _round(tr),
            "holdout": _round(ho),
        }
        trials.append(row)
        log(
            f"[train] MAE {ho['mae']:.2f}  iter {iters}  { _plain(extra) }"
        )
        if best is None or ho["mae"] < best["holdout_mae"]:
            best = {
                "booster": booster,
                "holdout_mae": ho["mae"],
                "train": tr,
                "holdout": ho,
                "params": _plain(extra),
                "best_iteration": iters,
            }

    assert best is not None
    beats_ridge = best["holdout"]["mae"] < ridge_ho["mae"]
    beats_rpm = best["holdout"]["mae"] < rpm_ho["mae"]
    log(f"[train] winner holdout MAE {best['holdout']['mae']:.2f}  {best['params']}")
    log(f"[train] beats ridge={beats_ridge}  beats rpm={beats_rpm}")

    log("[train] refit on all 48k labeled rows")
    full_builder = FeatureBuilder().fit(labeled)
    X_all = full_builder.transform(labeled)
    final = fit_final(
        X_all,
        labeled["posted_rate"],
        full_builder.categorical,
        best["params"],
        best["best_iteration"],
    )
    payload = {
        "builder": full_builder,
        "booster": final,
        "params": best["params"],
        "n_estimators": best["best_iteration"],
        "seed": SEED,
        "target": "log1p(posted_rate)",
        "objective": "mae",
        "fit_on": "all_labeled_jan_oct",
        "categorical": list(full_builder.categorical),
        "feature_names": list(full_builder.feature_names),
    }
    save_artifact(MODEL_ARTIFACT, payload)

    report = {
        "split": {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train": "2025-01-01 to 2025-08-31",
            "holdout": "2025-09-01 to 2025-10-31",
            "holdout_used_for_early_stopping_and_grid": True,
            "grid_size": len(GRID),
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
            "saved_model": {
                "fit_on": "all_labeled_jan_oct",
                "n_estimators": best["best_iteration"],
            },
        },
        "grid": trials,
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2) + "\n")
    log(f"[train] wrote {METRICS_JSON}")
    log(f"[train] wrote {MODEL_ARTIFACT}")


if __name__ == "__main__":
    main()

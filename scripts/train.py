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
from freight.model import (
    FAMILIES,
    GRIDS,
    fit_final,
    fit_one,
    importance_table,
    install_current,
    n_trees,
    predict_matrix,
)
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
        elif isinstance(val, bool):
            out[key] = val
        elif isinstance(val, int):
            out[key] = int(val)
        else:
            out[key] = val
    return out


def main() -> None:
    set_seed()
    log("[train] seed=42  families=" + ",".join(FAMILIES))
    log("[train] loading labeled file")
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

    all_trials = []
    family_best = []
    for family in FAMILIES:
        grid = GRIDS[family]
        log(f"[train] {family}: {len(grid)} configs")
        best = None
        for extra in tqdm(grid, desc=family, unit="cfg"):
            booster = fit_one(family, X_tr, y_tr, X_ho, y_ho, builder.categorical, extra)
            pred_ho = predict_matrix(booster, X_ho, builder.categorical, family)
            pred_tr = predict_matrix(booster, X_tr, builder.categorical, family)
            ho = evaluate(y_ho, pred_ho, holdout["distance"])
            tr = evaluate(y_tr, pred_tr, train["distance"])
            iters = n_trees(family, booster)
            row = {
                "family": family,
                "params": _plain(extra),
                "best_iteration": iters,
                "train": _round(tr),
                "holdout": _round(ho),
            }
            all_trials.append(row)
            log(f"[train] {family}  MAE {ho['mae']:.2f}  trees {iters}  {_plain(extra)}")
            if best is None or ho["mae"] < best["holdout_mae"]:
                best = {
                    "family": family,
                    "booster": booster,
                    "holdout_mae": ho["mae"],
                    "train": tr,
                    "holdout": ho,
                    "params": _plain(extra),
                    "best_iteration": iters,
                }
        assert best is not None
        log(f"[train] {family} winner MAE {best['holdout']['mae']:.2f}")
        family_best.append(best)

    winner = min(family_best, key=lambda item: item["holdout_mae"])
    log(
        f"[train] contest winner {winner['family']}  "
        f"MAE {winner['holdout']['mae']:.2f}  {winner['params']}"
    )
    for item in family_best:
        log(f"[train]   {item['family']}: holdout MAE {item['holdout']['mae']:.2f}")

    beats_ridge = winner["holdout"]["mae"] < ridge_ho["mae"]
    beats_rpm = winner["holdout"]["mae"] < rpm_ho["mae"]
    log(f"[train] beats ridge={beats_ridge}  beats rpm={beats_rpm}")

    log(f"[train] refit {winner['family']} on all 48k labeled rows")
    full_builder = FeatureBuilder().fit(labeled)
    X_all = full_builder.transform(labeled)
    final = fit_final(
        winner["family"],
        X_all,
        labeled["posted_rate"],
        full_builder.categorical,
        winner["params"],
        winner["best_iteration"],
    )
    payload = {
        "family": winner["family"],
        "builder": full_builder,
        "booster": final,
        "params": winner["params"],
        "n_estimators": winner["best_iteration"],
        "seed": SEED,
        "target": "log1p(posted_rate)",
        "objective": "mae",
        "fit_on": "all_labeled_jan_oct",
        "holdout_mae": round(winner["holdout"]["mae"], 4),
        "holdout_metrics": _round(winner["holdout"]),
        "categorical": list(full_builder.categorical),
        "feature_names": list(full_builder.feature_names),
    }
    install_current(payload)

    report = {
        "split": {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train": "2025-01-01 to 2025-08-31",
            "holdout": "2025-09-01 to 2025-10-31",
            "holdout_used_for_early_stopping_and_grid": True,
        },
        "seed": SEED,
        "equipment_rpm_holdout": _round(rpm_ho),
        "ridge_holdout": _round(ridge_ho),
        "contest": {item["family"]: _round(item["holdout"]) for item in family_best},
        "winner": {
            "family": winner["family"],
            "params": winner["params"],
            "best_iteration": winner["best_iteration"],
            "train": _round(winner["train"]),
            "holdout": _round(winner["holdout"]),
            "beats_equipment_rpm": beats_rpm,
            "beats_ridge": beats_ridge,
            "importance_gain": importance_table(
                winner["family"], winner["booster"], builder.feature_names
            ),
            "saved_model": {
                "path": str(MODEL_ARTIFACT),
                "fit_on": "all_labeled_jan_oct",
                "n_estimators": winner["best_iteration"],
            },
        },
        "grid": all_trials,
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2) + "\n")
    log(f"[train] wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()

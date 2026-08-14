from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.baselines import RIDGE_COLS, EquipmentRpmBaseline, RidgeBaseline
from freight.config import set_seed
from freight.features import FeatureBuilder
from freight.metrics import evaluate
from freight.paths import BASELINES_JSON, TRAIN_TEST_CSV
from freight.progress import log
from freight.split import time_split


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in metrics.items()}


def main() -> None:
    from tqdm import tqdm

    set_seed()
    log("[baselines] seed=42  loading labeled file")
    labeled = pd.read_csv(TRAIN_TEST_CSV, parse_dates=["date"])
    train, holdout = time_split(labeled)
    log(f"[baselines] jan-aug {len(train):,}  sep-oct {len(holdout):,}")
    y_tr = train["posted_rate"]
    y_ho = holdout["posted_rate"]

    steps = tqdm(total=2, desc="baselines")
    log("[baselines] equipment rpm")
    rpm = EquipmentRpmBaseline().fit(train)
    steps.update(1)

    log("[baselines] features + ridge")
    builder = FeatureBuilder().fit(train)
    X_tr = builder.transform(train)
    X_ho = builder.transform(holdout)
    ridge = RidgeBaseline().fit(X_tr, train["equipment"], y_tr)
    steps.update(1)
    steps.close()

    report = {
        "split": {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train": "2025-01-01 to 2025-08-31",
            "holdout": "2025-09-01 to 2025-10-31",
        },
        "equipment_rpm": {
            "train": _round(evaluate(y_tr, rpm.predict(train), train["distance"])),
            "holdout": _round(evaluate(y_ho, rpm.predict(holdout), holdout["distance"])),
            "rpm_by_eq": {k: round(v, 4) for k, v in rpm.rpm_by_eq.items()},
        },
        "ridge": {
            "train": _round(evaluate(y_tr, ridge.predict(X_tr, train["equipment"]), train["distance"])),
            "holdout": _round(
                evaluate(y_ho, ridge.predict(X_ho, holdout["equipment"]), holdout["distance"])
            ),
            "cols": list(RIDGE_COLS),
        },
    }

    BASELINES_JSON.write_text(json.dumps(report, indent=2) + "\n")
    log(f"[baselines] equipment rpm holdout {report['equipment_rpm']['holdout']}")
    log(f"[baselines] ridge holdout         {report['ridge']['holdout']}")
    log(f"[baselines] wrote {BASELINES_JSON}")


if __name__ == "__main__":
    main()

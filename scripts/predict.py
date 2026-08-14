from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.model import load_artifact, predict_loads
from freight.paths import (
    DECEMBER_INPUTS_CSV,
    MODEL_ARTIFACT,
    VALIDATION_CSV,
    VALIDATION_PREDICTIONS_CSV,
    VALIDATION_TEMPLATE_CSV,
)
from freight.progress import log


def main() -> None:
    log(f"[predict] loading {MODEL_ARTIFACT}")
    artifact = load_artifact(MODEL_ARTIFACT)
    log(f"[predict] fit_on={artifact.get('fit_on')}  params={artifact.get('params')}")

    steps = tqdm(total=2, desc="predict")
    log("[predict] scoring validation.csv (12,000 rows)")
    val = pd.read_csv(VALIDATION_CSV)
    template = pd.read_csv(VALIDATION_TEMPLATE_CSV)
    rates = np.round(predict_loads(artifact, val), 2)
    pred = pd.Series(rates, index=val["load_id"].astype(str))
    out = template.copy()
    out["predicted_rate"] = out["load_id"].astype(str).map(pred)
    if out["predicted_rate"].isna().any() or (out["predicted_rate"] <= 0).any():
        raise SystemExit("validation predictions missing or not positive")
    out.to_csv(VALIDATION_PREDICTIONS_CSV, index=False)
    log(
        f"[predict] wrote {VALIDATION_PREDICTIONS_CSV}  {len(out)} rows  "
        f"{rates.min():.2f} to {rates.max():.2f}"
    )
    steps.update(1)

    log("[predict] scoring december-chart-inputs.csv")
    december = pd.read_csv(DECEMBER_INPUTS_CSV)
    december["predicted_rate"] = np.round(predict_loads(artifact, december), 2)
    if (december["predicted_rate"] <= 0).any():
        raise SystemExit("december predictions not positive")
    december.to_csv(DECEMBER_INPUTS_CSV, index=False)
    log(
        f"[predict] december {float(december['predicted_rate'].min()):.2f} -> "
        f"{float(december['predicted_rate'].max()):.2f}  "
        f"nunique={int(december['predicted_rate'].nunique())}"
    )
    log(f"[predict] wrote {DECEMBER_INPUTS_CSV}")
    steps.update(1)
    steps.close()


if __name__ == "__main__":
    main()

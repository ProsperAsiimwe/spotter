"""Paths for the train/predict scripts.

The PDF uses underscores in a few filenames; the actual files use hyphens.
`december-chart-inputs.csv` sits in the repo root, not under data/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SCORER_RESULTS_DIR = ROOT / "scorer_results"

TRAIN_TEST_CSV = DATA_DIR / "train-test.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_CSV = DATA_DIR / "validation-predictions-template.csv"

DECEMBER_INPUTS_CSV = ROOT / "december-chart-inputs.csv"
VALIDATION_PREDICTIONS_CSV = ROOT / "validation_predictions.csv"

MODEL_ARTIFACT = MODELS_DIR / "rate_model.joblib"
METRICS_JSON = REPORTS_DIR / "metrics.json"
EDA_MD = REPORTS_DIR / "eda.md"
DECEMBER_CHART_PNG = SCORER_RESULTS_DIR / "candidate_december.png"

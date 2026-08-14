# Freight rate prediction

Predicts `posted_rate` for truck loads. The labeled file (`data/train-test.csv`)
runs Jan through Oct 2025. `data/validation.csv` is Nov-Dec and has no target.
There's also a 31-day Lexington to Fort Wayne series used for the December chart.

The original question is in `freight-rate-ml-assessment.pdf`.

## Setup

Python 3.9+. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src
```

On macOS, LightGBM needs OpenMP. If `import lightgbm` fails with an OpenMP error:

```bash
brew list libomp || brew install libomp
```

`PYTHONPATH=src` is required so `import freight` works.

## Run

```bash
python3 scripts/eda.py
python3 scripts/baselines.py
python3 scripts/train.py
python3 scripts/predict.py
python3 score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

`train.py` fits LightGBM, XGBoost, and CatBoost on the same Jan-Aug / Sep-Oct
split, keeps the lowest holdout MAE, refits that family on all 48k labeled rows,
and writes `models/rate_model.joblib`. The previous current model is moved to
`models/archives/` with family, MAE, tree count, seed, and timestamp in the
filename. `predict.py` always loads `models/rate_model.joblib`.

`score.py` shipped with the prompt; leave it alone. It checks both prediction
files and writes `scorer_results/candidate_december.png`. If you only want to
re-check the committed CSVs, skip to that command.

`december-chart-inputs.csv` is in the repo root, not under `data/`. A few names
in the PDF use underscores (`train_test.csv`); the files on disk use hyphens.

## Layout

- `src/freight/`: split, features, model. Train and predict both import from here.
- `models/rate_model.joblib`: current winner
- `models/archives/`: previous winners (name has family, MAE, trees, seed, time)
- `scripts/`: `eda.py`, `baselines.py`, `train.py`, `predict.py`
- `data/train-test.csv`: 48k labeled rows
- `data/validation.csv`: 12k rows to score
- `data/validation-predictions-template.csv`: `load_id` list for the submission file
- `reports/`: EDA notes, writeup, holdout metrics
- `scorer_results/candidate_december.png`: chart from `score.py`

## Model

Date split: Jan-Aug train, Sep-Oct holdout. After that looks fine, refit on all
48k labeled rows and predict the 12k file.

LightGBM, XGBoost, and CatBoost all train on `log1p(posted_rate)` (MAE objective)
and invert with `expm1`. `random_state` / `random_seed` is 42. Each family gets
its own time-split grid (`ParameterGrid` on Sep-Oct MAE, not k-fold). Column and
row dropout is `colsample_bytree` / `subsample` (CatBoost: `rsm` / `subsample`).
The winner of that contest is the current model until the next `train.py` run.
It still has to beat the miles-by-equipment and Ridge baselines.

Missing `weight` and `market_index` are filled with training-split medians.
December rows don't include `market_index` or `quote_signal`, so those columns
are optional in the same feature code (otherwise the chart goes flat).

`validation_predictions.csv` is `load_id,predicted_rate` for `TE-000001` through
`TE-012000`. All rates are positive.

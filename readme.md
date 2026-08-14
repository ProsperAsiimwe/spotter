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

`train.py` fits on `data/train-test.csv`, writes `reports/metrics.json`, and
saves `models/rate_model.joblib`. `predict.py` writes `validation_predictions.csv`
and fills `predicted_rate` on `december-chart-inputs.csv`.

`score.py` shipped with the prompt; leave it alone. It checks both prediction
files and writes `scorer_results/candidate_december.png`. If you only want to
re-check the committed CSVs, skip to that command.

`december-chart-inputs.csv` is in the repo root, not under `data/`. A few names
in the PDF use underscores (`train_test.csv`); the files on disk use hyphens.

## Layout

- `src/freight/`: split, features, model. Train and predict both import from here.
- `scripts/`: `eda.py`, `baselines.py`, `train.py`, `predict.py`
- `data/train-test.csv`: 48k labeled rows
- `data/validation.csv`: 12k rows to score
- `data/validation-predictions-template.csv`: `load_id` list for the submission file
- `reports/`: EDA notes, writeup, holdout metrics
- `scorer_results/candidate_december.png`: chart from `score.py`

## Model

Date split: Jan-Aug train, Sep-Oct holdout. After that looks fine, refit on all
48k labeled rows and predict the 12k file.

LightGBM on `log1p(posted_rate)`, inverted with `expm1`. `random_state` is 42
(`freight.config.SEED`), and `set_seed(42)` is called before fit. First baseline is
distance times median $/mile by equipment type; the tree model has to beat that
before I use it for the submission.

Missing `weight` and `market_index` are filled with training-split medians.
December rows don't include `market_index` or `quote_signal`, so those columns
are optional in the same feature code (otherwise the chart goes flat).

`validation_predictions.csv` is `load_id,predicted_rate` for `TE-000001` through
`TE-012000`. All rates are positive.

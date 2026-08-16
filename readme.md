# Freight rate prediction

This project predicts `posted_rate` for truck loads using the labeled data in
`data/train-test.csv`, which covers January through October 2025. The
unlabeled `data/validation.csv` contains November and December observations
for which the target column, posted_rate is not provided. A separate 31-day Lexington-to-Fort
Wayne series, `./december-chart-inputs.csv` is also included for the December prediction chart.

The original assessment question is provided in `freight-rate-ml-assessment.pdf`.

## Live demo

The repository includes a live deployment of the final selected model for
interactive inspection. During model selection, LightGBM, XGBoost, and
CatBoost were evaluated using the same January-August training and
September-October holdout split. Those three are the usual gradient-boosted
tree families for this kind of table: mixed numeric and high-cardinality
categorical columns, nonlinear interactions (equipment and distance, city,
month), and irregular event rows rather than a dense panel. LightGBM is
built for fast histogram splits on large tabular sets. XGBoost is the
standard regularized boosting baseline. CatBoost is the one designed around
categorical features, which matters here because pickup, delivery, and
equipment are categoricals. LightGBM achieved the lowest holdout MAE
($108.58, compared with $109.90 for XGBoost and $116.51 for CatBoost) and was
therefore selected as the final model.

The deployed application serves this frozen LightGBM artifact directly. It
does not retrain the model or select a different model family at request time.

**Live demo:** [prosperasiimwe.dev/freight](https://prosperasiimwe.dev/freight)

The interface provides test cases designed to exercise conditions handled by
the same training and prediction pipeline:

- **Presets:** Lexington to Fort Wayne, missing weight, an unseen city
  (Laredo), a long Reefer haul, and Christmas.
- **December mode:** removes `market_index` and `quote_signal`, matching the
  December chart data.
- **Prediction output:** predicted freight rate, implied dollars per mile,
  inference latency, and the equipment-by-miles baseline for comparison.
- **Model diagnostics:** the December prediction series and LightGBM gain
  feature importance are generated from the live API.
- **Cold-start handling:** the application displays a wake/retry message when
  the underlying free Hugging Face Space has been sleeping. The first request
  after a period of inactivity can take up to a minute.

The GitHub repository remains the source of truth for the assessment and
contains the complete training and prediction pipeline. Training data is not
uploaded to the deployment. The live application serves only the locked
`models/rate_model.joblib` artifact.

## Hugging Face

The deployment uses two Hugging Face repositories: one for the serialized
model artifact and one for the HTTP inference service.

**Model:** [huggingface.co/Byteroot/lane-rate-lgbm](https://huggingface.co/Byteroot/lane-rate-lgbm)

- `rate_model.joblib` (~2.3 MB): serialized dictionary containing the feature
  builder, fitted LightGBM booster, model family, parameters, and holdout
  metrics. This is the same artifact loaded locally by `predict.py`.
- `current.json`: lightweight model metadata containing the model family,
  number of trees, random seed, holdout MAE, and the fact that the final model
  was refit on all 48k labeled observations.

**Space:** [huggingface.co/spaces/Byteroot/lane-rate](https://huggingface.co/spaces/Byteroot/lane-rate)

The inference service runs as a Docker Space using FastAPI on port 7860. Its
public host is [byteroot-lane-rate.hf.space](https://byteroot-lane-rate.hf.space).
At startup, the service downloads `rate_model.joblib` from the model
repository and loads it once. A copy of `src/freight/` is included in the
Space so that the serialized feature builder and model can be resolved
correctly.

The FastAPI OpenAPI documentation is available at `/docs`.

For Example:

```bash
curl -s https://byteroot-lane-rate.hf.space/predict \
  -H 'content-type: application/json' \
  -d '{"pickup":"Lexington","delivery":"Fort Wayne","distance":360,"equipment":"Dry Van","weight":32000,"date":"2025-12-01"}'
```

That Lexington row should come back near `$839.50`, which matches
`december-chart-inputs.csv` for 2025-12-01.

## Setup

The project requires Python 3.9 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src
```

On macOS, LightGBM requires OpenMP. If `import lightgbm` fails with an
OpenMP-related error, install `libomp` with Homebrew:

```bash
brew list libomp || brew install libomp
```

`PYTHONPATH=src` is required so that the shared `freight` package can be
imported by the training and prediction scripts.

## Run

The full workflow can be reproduced with:

```bash
python3 scripts/eda.py
python3 scripts/baselines.py
python3 scripts/train.py
python3 scripts/predict.py
python3 score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

The analysis begins with exploratory data analysis and baseline models before
moving to model selection and final prediction.

`train.py` evaluates LightGBM, XGBoost, and CatBoost using the same
January-August training period and September-October holdout period. The model
with the lowest holdout MAE is selected, then refit on all 48k labeled rows.
The resulting model is saved to `models/rate_model.joblib`.

When a new model replaces the current one, the previous model is moved to
`models/archives/`. Its filename records the model family, holdout MAE, tree
count, random seed, and training timestamp so that previous runs can be
identified and compared. `predict.py` always loads the current model from
`models/rate_model.joblib`.

`score.py` was supplied with the assessment and should not be modified. It
evaluates both prediction files and produces
`scorer_results/candidate_december.png`. If the committed CSVs are already
available and only the final scoring needs to be reproduced, the earlier
commands can be skipped.

`december-chart-inputs.csv` is located in the repository root rather than
under `data/`. The assessment PDF also refers to some files using underscores,
such as `train_test.csv`; the corresponding files in the repository use
hyphens.

## Layout

* `src/freight/`: shared split, feature, and model code used by training and prediction
* `models/rate_model.joblib`: current selected model
* `models/archives/`: previous models, with family, MAE, tree count, seed, and timestamp in the filename
* `scripts/`: `eda.py`, `baselines.py`, `train.py`, and `predict.py`
* `data/train-test.csv`: 48k labeled observations from January through October
* `data/validation.csv`: 12k unlabeled observations from November and December
* `data/validation-predictions-template.csv`: `load_id` values for the submission file
* `reports/`: EDA notes, modelling writeup, and holdout metrics
* `scorer_results/candidate_december.png`: December chart generated by `score.py`

## Models

The modelling strategy follows the temporal structure of the data. January
through August is used for training and September through October is held out
for model selection. Once the best-performing configuration has been
identified, that configuration is refit on all 48k labeled observations and
used to predict the 12k validation rows.

LightGBM, XGBoost, and CatBoost are all trained on `log1p(posted_rate)` with an
MAE objective, and predictions are transformed back to the original rate scale
with `expm1`. The random seed is fixed at 42 (`random_state` or
`random_seed`, depending on the model).

Each model family has its own time-based hyperparameter grid. Candidate
configurations are evaluated with `ParameterGrid` using September-October MAE,
rather than k-fold cross-validation, so that the evaluation remains
consistent with the chronological nature of the forecasting problem.

Column and row subsampling are controlled through `colsample_bytree` and
`subsample`; for CatBoost, the corresponding parameters are `rsm` and
`subsample`. The configuration with the lowest holdout MAE becomes the current
model until the next `train.py` run. The selected model must also outperform
the miles-by-equipment and Ridge baselines.

Missing `weight` and `market_index` values are filled using medians calculated
from the training split. The December data does not contain `market_index` or
`quote_signal`, so these columns are treated as optional by the shared feature
code. This is necessary for the December predictions to vary appropriately
rather than producing a flat chart.

`validation_predictions.csv` contains `load_id,predicted_rate` for
`TE-000001` through `TE-012000`. All predicted rates are positive.

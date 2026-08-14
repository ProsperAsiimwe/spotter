from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.paths import (
    DECEMBER_INPUTS_CSV,
    EDA_MD,
    REPORTS_DIR,
    TRAIN_TEST_CSV,
    VALIDATION_CSV,
)


def _missing(frame: pd.DataFrame) -> pd.Series:
    return frame.isna().sum().loc[lambda s: s > 0]


def _cities(frame: pd.DataFrame) -> set[str]:
    return set(frame["pickup"]) | set(frame["delivery"])


def _corr(a: pd.Series, b: pd.Series) -> float:
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_TEST_CSV, parse_dates=["date"])
    val = pd.read_csv(VALIDATION_CSV, parse_dates=["date"])
    december = pd.read_csv(DECEMBER_INPUTS_CSV, parse_dates=["date"])
    train["rpm"] = train["posted_rate"] / train["distance"]
    return train, val, december


def _write_plots(train: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    monthly = (
        train.set_index("date")
        .resample("ME")["posted_rate"]
        .median()
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_title("Median posted_rate by month (labeled file)")
    ax.set_ylabel("USD")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_monthly_rate.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    train.boxplot(column="rpm", by="equipment", ax=ax, showfliers=False)
    ax.set_title("Rate per mile by equipment")
    ax.set_xlabel("")
    ax.set_ylabel("USD / mile")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_rpm_by_equipment.png", dpi=140)
    plt.close(fig)


def _report(train: pd.DataFrame, val: pd.DataFrame, december: pd.DataFrame) -> str:
    train_cities = _cities(train)
    val_cities = _cities(val)
    new_cities = sorted(val_cities - train_cities)
    new_pickup = sorted(set(val["pickup"]) - set(train["pickup"]))
    new_delivery = sorted(set(val["delivery"]) - set(train["delivery"]))

    lfw = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
    lanes = train.groupby(["pickup", "delivery", "equipment"]).size()
    od = train.groupby(["pickup", "delivery"]).size()

    month = train["date"].dt.strftime("%Y-%m")
    monthly_med = train.groupby(month)["posted_rate"].median().rename_axis("month")
    monthly_rpm = train.groupby(month)["rpm"].median().rename_axis("month")
    labeled_months = train.groupby(month).size().rename_axis("month")
    val_months = (
        val.groupby(val["date"].dt.strftime("%Y-%m")).size().rename_axis("month")
    )

    weekend = train["date"].dt.dayofweek >= 5
    city_coord_n = (
        pd.concat(
            [
                train[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
                ),
                train[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
                ),
            ]
        )
        .drop_duplicates()
        .groupby("city")
        .size()
    )

    train_miss = _missing(train)
    val_miss = _missing(val)
    train_miss_txt = train_miss.to_string() if len(train_miss) else "none"
    val_miss_txt = val_miss.to_string() if len(val_miss) else "none"

    dist_corr = _corr(train["distance"], train["posted_rate"])
    qs_rpm_corr = _corr(train["quote_signal"], train["rpm"])
    mi_rpm_corr = _corr(train["market_index"], train["rpm"])
    qs_dist_corr = _corr(train["quote_signal"] * train["distance"], train["posted_rate"])
    weight_by_eq = train.groupby("equipment")["weight"].apply(lambda s: int(s.isna().sum()))
    market_by_eq = train.groupby("equipment")["market_index"].apply(
        lambda s: int(s.isna().sum())
    )

    return f"""# EDA notes

From `scripts/eda.py`. Numbers come off the CSVs, not the PDF.

## Files

- labeled: {len(train):,} rows, {train['date'].min().date()} to {train['date'].max().date()} ({train['date'].nunique()} days)
- validation: {len(val):,} rows, {val['date'].min().date()} to {val['date'].max().date()} ({val['date'].nunique()} days), no `posted_rate`
- December chart: {len(december)} rows, {december['pickup'].iloc[0]} to {december['delivery'].iloc[0]}, {december['distance'].iloc[0]:g} miles, {december['equipment'].iloc[0]}, {december['weight'].iloc[0]:g} lb. No lat/lon, `market_index`, or `quote_signal`.

Load ids: labeled `{train['load_id'].iloc[0]}`..`{train['load_id'].iloc[-1]}`, validation `{val['load_id'].iloc[0]}`..`{val['load_id'].iloc[-1]}`. Duplicates in labeled: {int(train['load_id'].duplicated().sum())}, validation: {int(val['load_id'].duplicated().sum())}.

## Missing

Labeled:
```
{train_miss_txt}
```

Validation:
```
{val_miss_txt}
```

Labeled missing `weight` by equipment:
```
{weight_by_eq.to_string()}
```

Labeled missing `market_index` by equipment:
```
{market_by_eq.to_string()}
```

Same holes show up in validation, so dropping rows is not an option for the 12k file.

## Target and distance

posted_rate: min {train['posted_rate'].min():.2f}, median {train['posted_rate'].median():.2f}, mean {train['posted_rate'].mean():.2f}, max {train['posted_rate'].max():.2f}

distance: min {train['distance'].min():.1f}, median {train['distance'].median():.1f}, mean {train['distance'].mean():.1f}, max {train['distance'].max():.1f}

USD/mile: min {train['rpm'].min():.3f}, median {train['rpm'].median():.3f}, mean {train['rpm'].mean():.3f}, max {train['rpm'].max():.3f}

corr(distance, posted_rate) = {dist_corr:.3f}
corr(quote_signal, usd/mile) = {qs_rpm_corr:.3f}
corr(market_index, usd/mile) = {mi_rpm_corr:.3f}
corr(quote_signal * distance, posted_rate) = {qs_dist_corr:.3f}

quote_signal is not a ready-made $/mile. Distance dominates the dollar amount. Multiplying by quote_signal does not beat distance alone.

## Equipment

Labeled: {train['equipment'].value_counts().to_dict()}
Validation: {val['equipment'].value_counts().to_dict()}

Median USD/mile by equipment:
```
{train.groupby('equipment')['rpm'].median().sort_values().to_string()}
```

## Time

Loads per month (labeled):
```
{labeled_months.to_string()}
```

Validation:
```
{val_months.to_string()}
```

Median posted_rate by month:
```
{monthly_med.to_string()}
```

Median USD/mile by month:
```
{monthly_rpm.to_string()}
```

Weekday median rate {train.loc[~weekend, 'posted_rate'].median():.2f} vs weekend {train.loc[weekend, 'posted_rate'].median():.2f}.
Weekday median USD/mile {train.loc[~weekend, 'rpm'].median():.3f} vs weekend {train.loc[weekend, 'rpm'].median():.3f}.
Weekend vs weekday is basically flat. The month pattern is the one to keep (USD/mile peaks in June, lower in Jan).

Sep-Oct is the last labeled window before the Nov-Dec file, so that is the holdout. A random row split would mix those months.

## Cities and lanes

Labeled cities: {len(train_cities)}. Validation cities: {len(val_cities)}. New in validation: {new_cities or 'none'}
New pickups: {new_pickup or 'none'}
New deliveries: {new_delivery or 'none'}

Unique pickup-delivery pairs: {len(od)} (median {od.median():.0f} loads, max {od.max()})
Unique pickup-delivery-equipment lanes: {len(lanes)} (median {lanes.median():.0f} loads, max {lanes.max()})

Most lanes are too thin for a per-lane time-series model.

Each city name maps to one lat/lon in the labeled file (max {int(city_coord_n.max())} pair per city).

## Lexington to Fort Wayne

Labeled rows on that OD: {len(lfw)} across {lfw['date'].nunique()} days.
Equipment: {lfw['equipment'].value_counts().to_dict() if len(lfw) else {}}
Dry Van median rate {lfw.loc[lfw['equipment']=='Dry Van', 'posted_rate'].median() if len(lfw) else float('nan'):.2f} (n={int((lfw['equipment']=='Dry Van').sum()) if len(lfw) else 0}), median USD/mile {lfw.loc[lfw['equipment']=='Dry Van', 'rpm'].median() if len(lfw) else float('nan'):.3f}.
Chart distance is {december['distance'].iloc[0]:g} vs labeled Dry Van distances {lfw.loc[lfw['equipment']=='Dry Van', 'distance'].min() if len(lfw) else float('nan'):.1f}-{lfw.loc[lfw['equipment']=='Dry Van', 'distance'].max() if len(lfw) else float('nan'):.1f}.

Not much history on the exact chart lane. Calendar features have to do some of the work in December, and `market_index` / `quote_signal` cannot be required.

## Plots

- `reports/eda_monthly_rate.png`
- `reports/eda_rpm_by_equipment.png`
"""


def main() -> None:
    train, val, december = _load()
    _write_plots(train)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    text = _report(train, val, december)
    EDA_MD.write_text(text)
    print(text)
    print(f"wrote {EDA_MD}")
    print(f"wrote {REPORTS_DIR / 'eda_monthly_rate.png'}")
    print(f"wrote {REPORTS_DIR / 'eda_rpm_by_equipment.png'}")


if __name__ == "__main__":
    main()

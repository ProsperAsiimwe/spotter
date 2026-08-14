"""Feature matrix for training and inference.

December doesn't have market_index or quote_signal, so those have to be optional
here. Same transform for the 12k file and the chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("2025-01-01")
UNKNOWN = "unknown"

# 2025 US federal holidays.
HOLIDAYS = frozenset(
    pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-20",
            "2025-02-17",
            "2025-05-26",
            "2025-06-19",
            "2025-07-04",
            "2025-09-01",
            "2025-10-13",
            "2025-11-11",
            "2025-11-27",
            "2025-12-25",
        ]
    ).date
)

FEATURE_NAMES = [
    "distance",
    "log_distance",
    "equipment",
    "weight",
    "weight_missing",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "pickup",
    "delivery",
    "pickup_unseen",
    "delivery_unseen",
    "dow",
    "month",
    "week",
    "day",
    "is_weekend",
    "is_holiday",
    "days_since_start",
    "market_index",
    "market_missing",
    "quote_signal",
    "quote_missing",
]

CATEGORICAL = ["equipment", "pickup", "delivery"]


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def _str(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(UNKNOWN, index=frame.index)
    return frame[name].fillna(UNKNOWN).astype(str)


class FeatureBuilder:
    """Fit medians and city tables on train only, then transform any load table."""

    def __init__(self) -> None:
        self.equipment_codes: dict[str, int] = {}
        self.city_codes: dict[str, int] = {}
        self.city_lat: dict[str, float] = {}
        self.city_lon: dict[str, float] = {}
        self.weight_medians: dict[str, float] = {}
        self.market_medians: dict[str, float] = {}
        self.quote_medians: dict[str, float] = {}
        self.weight_global: float = 0.0
        self.market_global: float = 0.0
        self.quote_global: float = 0.0
        self.default_lat: float = 0.0
        self.default_lon: float = 0.0
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.categorical: list[str] = list(CATEGORICAL)

    def fit(self, frame: pd.DataFrame) -> FeatureBuilder:
        equipment = _str(frame, "equipment")
        pickup = _str(frame, "pickup")
        delivery = _str(frame, "delivery")
        weight = _num(frame, "weight")
        market = _num(frame, "market_index")
        quote = _num(frame, "quote_signal")

        eq_levels = sorted(set(equipment) | {UNKNOWN})
        self.equipment_codes = {name: i for i, name in enumerate(eq_levels)}

        cities = sorted(set(pickup) | set(delivery) | {UNKNOWN})
        self.city_codes = {name: i for i, name in enumerate(cities)}

        coords = pd.concat(
            [
                pd.DataFrame(
                    {
                        "city": pickup,
                        "lat": _num(frame, "pickup_lat"),
                        "lon": _num(frame, "pickup_lon"),
                    }
                ),
                pd.DataFrame(
                    {
                        "city": delivery,
                        "lat": _num(frame, "delivery_lat"),
                        "lon": _num(frame, "delivery_lon"),
                    }
                ),
            ]
        ).dropna()
        coords = coords[coords["city"] != UNKNOWN]
        city_xy = coords.groupby("city")[["lat", "lon"]].median()
        self.city_lat = city_xy["lat"].to_dict()
        self.city_lon = city_xy["lon"].to_dict()
        self.default_lat = float(coords["lat"].median()) if len(coords) else 0.0
        self.default_lon = float(coords["lon"].median()) if len(coords) else 0.0

        tmp = pd.DataFrame(
            {"equipment": equipment, "weight": weight, "market": market, "quote": quote}
        )
        self.weight_medians = tmp.groupby("equipment")["weight"].median().dropna().to_dict()
        self.market_medians = tmp.groupby("equipment")["market"].median().dropna().to_dict()
        self.quote_medians = tmp.groupby("equipment")["quote"].median().dropna().to_dict()
        self.weight_global = float(weight.median()) if weight.notna().any() else 0.0
        self.market_global = float(market.median()) if market.notna().any() else 0.0
        self.quote_global = float(quote.median()) if quote.notna().any() else 0.0
        return self

    def _impute(self, raw: pd.Series, equipment: pd.Series, medians: dict[str, float], global_med: float) -> pd.Series:
        fill = equipment.map(medians).astype("float64").fillna(global_med)
        return raw.where(raw.notna(), fill)

    def _coord(self, frame: pd.DataFrame, city: pd.Series, lat_col: str, lon_col: str) -> tuple[pd.Series, pd.Series]:
        lat = _num(frame, lat_col)
        lon = _num(frame, lon_col)
        lat = lat.where(lat.notna(), city.map(self.city_lat).astype("float64"))
        lon = lon.where(lon.notna(), city.map(self.city_lon).astype("float64"))
        lat = lat.fillna(self.default_lat)
        lon = lon.fillna(self.default_lon)
        return lat, lon

    def _codes(self, names: pd.Series, table: dict[str, int]) -> pd.Series:
        unknown = table[UNKNOWN]
        return names.map(table).fillna(unknown).astype(int)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.city_codes:
            raise RuntimeError("FeatureBuilder.fit() first")

        equipment = _str(frame, "equipment")
        pickup = _str(frame, "pickup")
        delivery = _str(frame, "delivery")
        distance = _num(frame, "distance").clip(lower=1.0)
        weight = _num(frame, "weight")
        market = _num(frame, "market_index")
        quote = _num(frame, "quote_signal")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("date missing or unparseable")

        pickup_lat, pickup_lon = self._coord(frame, pickup, "pickup_lat", "pickup_lon")
        delivery_lat, delivery_lon = self._coord(frame, delivery, "delivery_lat", "delivery_lon")

        out = pd.DataFrame(index=frame.index)
        out["distance"] = distance
        out["log_distance"] = np.log(distance)
        out["equipment"] = self._codes(equipment, self.equipment_codes)
        out["weight_missing"] = weight.isna().astype(int)
        out["weight"] = self._impute(weight, equipment, self.weight_medians, self.weight_global)
        out["pickup_lat"] = pickup_lat
        out["pickup_lon"] = pickup_lon
        out["delivery_lat"] = delivery_lat
        out["delivery_lon"] = delivery_lon
        out["pickup"] = self._codes(pickup, self.city_codes)
        out["delivery"] = self._codes(delivery, self.city_codes)
        known = set(self.city_codes) - {UNKNOWN}
        out["pickup_unseen"] = (~pickup.isin(known)).astype(int)
        out["delivery_unseen"] = (~delivery.isin(known)).astype(int)
        out["dow"] = dates.dt.dayofweek
        out["month"] = dates.dt.month
        out["week"] = dates.dt.isocalendar().week.astype(int)
        out["day"] = dates.dt.day
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out["is_holiday"] = dates.dt.date.isin(HOLIDAYS).astype(int)
        out["days_since_start"] = (dates - ORIGIN).dt.days
        out["market_missing"] = market.isna().astype(int)
        out["market_index"] = self._impute(market, equipment, self.market_medians, self.market_global)
        out["quote_missing"] = quote.isna().astype(int)
        out["quote_signal"] = self._impute(quote, equipment, self.quote_medians, self.quote_global)
        return out[self.feature_names]

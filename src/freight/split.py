"""Jan-Aug train / Sep-Oct holdout on the labeled file."""

from __future__ import annotations

import pandas as pd

HOLDOUT_START = pd.Timestamp("2025-09-01")
LABELED_END = pd.Timestamp("2025-10-31")


def time_split(
    frame: pd.DataFrame, date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut on calendar date. No shuffling.

    Train: date < 2025-09-01
    Holdout: 2025-09-01 through 2025-10-31
    """
    dates = pd.to_datetime(frame[date_col])
    train_mask = dates < HOLDOUT_START
    holdout_mask = (dates >= HOLDOUT_START) & (dates <= LABELED_END)
    train = frame.loc[train_mask].copy()
    holdout = frame.loc[holdout_mask].copy()
    if train.empty or holdout.empty:
        raise ValueError("time_split produced an empty side")
    if dates.loc[train_mask].max() >= dates.loc[holdout_mask].min():
        raise ValueError("train and holdout dates overlap")
    return train, holdout

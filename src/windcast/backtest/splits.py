"""Rolling-origin (walk-forward) folds."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_origin_folds(start, end, initial_train_days: int, test_days: int,
                         step_days: int | None = None, expanding: bool = True,
                         min_test_days: int = 7) -> list[Fold]:
    """Consecutive test blocks, each trained on everything before it.

    ``expanding=True`` keeps the training start fixed; ``False`` slides a window of
    ``initial_train_days``. A final partial block shorter than ``min_test_days`` is dropped.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    step = pd.Timedelta(days=step_days or test_days)
    test_len = pd.Timedelta(days=test_days)
    folds, test_start = [], start + pd.Timedelta(days=initial_train_days)
    while test_start + pd.Timedelta(days=min_test_days) <= end:
        train_start = start if expanding else test_start - pd.Timedelta(days=initial_train_days)
        folds.append(Fold(len(folds), train_start, test_start, min(test_start + test_len, end)))
        test_start += step
    if not folds:
        raise ValueError("date range too short for the requested backtest layout")
    return folds


def split(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test rows for a fold.

    Training keeps only rows whose VALID time precedes the test start. That is an
    embargo as long as the longest lead: no training label overlaps the test period.
    """
    train = frame[(frame["issue_time"] >= fold.train_start) & (frame["valid_time"] < fold.test_start)]
    test = frame[(frame["issue_time"] >= fold.test_start) & (frame["issue_time"] < fold.test_end)]
    return train, test

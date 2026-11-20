"""Reference forecasts every model has to beat: persistence, climatology, ARIMA."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def persistence(frame: pd.DataFrame) -> np.ndarray:
    """Last observed farm power at issue time, held for every lead."""
    return frame["obs_power_t0"].to_numpy()


class Climatology:
    """Mean and quantiles of normalised power by hour of day, from training data only."""

    def __init__(self, quantiles: list[float]):
        self.quantiles = list(quantiles)

    def fit(self, frame: pd.DataFrame) -> Climatology:
        df = frame.dropna(subset=["y"])
        hours = df["valid_time"].dt.hour
        self.mean_ = df["y"].groupby(hours).mean().reindex(range(24)).fillna(df["y"].mean())
        self.q_ = (df["y"].groupby(hours).quantile(self.quantiles).unstack()
                   .reindex(range(24)).fillna(df["y"].quantile(self.quantiles)))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.mean_.reindex(frame["valid_time"].dt.hour).to_numpy()

    def predict_quantiles(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self.q_.reindex(frame["valid_time"].dt.hour)
        out.index = frame.index
        out.columns = [f"clim_q{int(round(q * 100)):02d}" for q in self.quantiles]
        return out


def arima_forecasts(series: pd.Series, train_end: pd.Timestamp, issue_times,
                    lead_steps: list[int], order=(2, 0, 1), history_steps: int = 400) -> pd.DataFrame:
    """ARIMA baseline via statsmodels.

    Parameters are estimated once on data before ``train_end``. For each origin the
    fitted model is re-applied to the most recent ``history_steps`` observations
    (no re-estimation) and forecast forward. ``series`` must be on a regular grid
    stamped at period end; NaNs are handled by the Kalman filter.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rows = []
    train = series[series.index < train_end]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(train.to_numpy(), order=tuple(order), trend="c",
                         enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=100)
        max_step = max(lead_steps)
        pos = pd.Series(np.arange(len(series)), index=series.index)
        for t0 in issue_times:
            if t0 not in pos.index:
                continue
            end = int(pos[t0]) + 1
            hist = series.iloc[max(0, end - history_steps):end].to_numpy()
            if np.isfinite(hist).sum() < 20:
                continue
            fc = fitted.apply(hist).forecast(max_step)
            for step in lead_steps:
                rows.append((t0, step, float(np.clip(fc[step - 1], 0.0, 1.0))))
    return pd.DataFrame(rows, columns=["issue_time", "lead_step", "arima"])

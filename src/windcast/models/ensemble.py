"""Stacked ensemble: non-negative, lead-dependent weights over member forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from windcast.utils import LEAD_LABELS, lead_bucket

MIN_SAMPLES = 150


class StackedBlender:
    """Convex combination of members, fitted on a time-ordered hold-out.

    Weights depend on lead time because the best mix changes with horizon:
    persistence-like members dominate the first steps, NWP-driven members later.
    With a handful of distinct leads (intra-hour products) there is one weight set
    per lead; otherwise one per lead bucket. A member that is NaN on a row is
    dropped and the remaining weights renormalised.
    """

    MAX_PER_LEAD = 8

    def __init__(self, members: list[str]):
        self.members = list(members)
        self.per_lead_ = False
        self.default_ = np.full(len(self.members), 1.0 / len(self.members))
        self.weights_: dict = {}

    def _keys(self, lead_h) -> np.ndarray:
        lead_h = np.asarray(lead_h, float)
        return np.round(lead_h, 4) if self.per_lead_ else lead_bucket(lead_h).astype(float)

    @staticmethod
    def _solve(P: np.ndarray, y: np.ndarray) -> np.ndarray:
        w, _ = nnls(P, y)
        return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1.0 / P.shape[1])

    def fit(self, preds: pd.DataFrame, y, lead_h) -> StackedBlender:
        P, y = preds[self.members].to_numpy(float), np.asarray(y, float)
        ok = np.isfinite(P).all(axis=1) & np.isfinite(y)
        if ok.sum() < MIN_SAMPLES:
            return self
        self.per_lead_ = len(np.unique(np.round(np.asarray(lead_h, float), 4))) <= self.MAX_PER_LEAD
        keys = self._keys(lead_h)
        self.default_ = self._solve(P[ok], y[ok])
        for key in np.unique(keys):
            m = ok & (keys == key)
            if m.sum() >= MIN_SAMPLES:
                self.weights_[float(key)] = self._solve(P[m], y[m])
        return self

    def predict(self, preds: pd.DataFrame, lead_h) -> np.ndarray:
        P = preds[self.members].to_numpy(float)
        W = np.vstack([self.weights_.get(float(k), self.default_) for k in self._keys(lead_h)]) \
            if len(P) else np.empty((0, len(self.members)))
        W = np.where(np.isfinite(P), W, 0.0)
        total = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.nansum(P * W, axis=1) / total
            fallback = np.nansum(P, axis=1) / np.maximum(np.isfinite(P).sum(axis=1), 1)
        return np.clip(np.where(total > 0, out, fallback), 0.0, 1.0)

    def table(self) -> pd.DataFrame:
        keys = sorted(self.weights_)
        labels = [f"{k:g}h" for k in keys] if self.per_lead_ else [LEAD_LABELS[int(k)] for k in keys]
        if not keys:
            keys, labels = [None], ["all"]
        rows = [self.weights_.get(k, self.default_) for k in keys]
        return pd.DataFrame(rows, index=labels, columns=self.members).round(3)

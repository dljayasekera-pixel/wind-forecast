"""Provider blending weights per lead bucket."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from windcast.utils import LEAD_LABELS, lead_bucket

MIN_SAMPLES = 200


def _weights(F: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    k = F.shape[1]
    if method == "equal" or len(y) < MIN_SAMPLES:
        return np.full(k, 1.0 / k)
    if method == "inverse_mse":
        w = 1.0 / np.maximum(np.mean((F - y[:, None]) ** 2, axis=0), 1e-9)
    elif method == "nnls":
        # Non-negative least squares, then normalise: a convex combination that
        # accounts for error correlation between providers.
        w, _ = nnls(F, y)
        if w.sum() <= 0:
            w = np.ones(k)
    else:
        raise ValueError(f"unknown blend method '{method}'")
    return w / w.sum()


class ProviderBlender:
    """Convex blend of bias-corrected provider forecasts.

    Weights are learned on rows where every provider delivered. At prediction time
    the weights of missing providers are dropped and the rest renormalised, so the
    blend degrades gracefully instead of failing.
    """

    def __init__(self, providers: list[str], method: str = "nnls"):
        self.providers, self.method = list(providers), method
        k = len(self.providers)
        self.weights_ = {b: np.full(k, 1.0 / k) for b in range(len(LEAD_LABELS))}

    def fit(self, forecasts: np.ndarray, observed, lead_h) -> ProviderBlender:
        F, y = np.asarray(forecasts, float), np.asarray(observed, float)
        ok = np.isfinite(F).all(axis=1) & np.isfinite(y)
        buckets = lead_bucket(lead_h)
        overall = _weights(F[ok], y[ok], self.method)
        for b in range(len(LEAD_LABELS)):
            m = ok & (buckets == b)
            self.weights_[b] = _weights(F[m], y[m], self.method) if m.sum() >= MIN_SAMPLES else overall
        return self

    def transform(self, forecasts: np.ndarray, lead_h) -> np.ndarray:
        F = np.asarray(forecasts, float)
        W = np.vstack([self.weights_[b] for b in range(len(LEAD_LABELS))])[lead_bucket(lead_h)]
        W = np.where(np.isfinite(F), W, 0.0)
        total = W.sum(axis=1)
        # If only zero-weight providers are present, fall back to their plain mean.
        present = np.isfinite(F).sum(axis=1)
        plain = np.where(present > 0, np.nansum(F, axis=1) / np.maximum(present, 1), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            blended = np.nansum(F * W, axis=1) / total
        return np.where(total > 0, blended, plain)

    def table(self):
        import pandas as pd

        return pd.DataFrame([self.weights_[b] for b in range(len(LEAD_LABELS))],
                            index=LEAD_LABELS, columns=self.providers).round(3)

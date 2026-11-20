"""Bias correction of NWP wind speed against on-site observations."""
from __future__ import annotations

import numpy as np

from windcast.utils import LEAD_LABELS, lead_bucket

MIN_SAMPLES = 200


class MOSCorrector:
    """Model Output Statistics: obs = a + b * forecast, fitted per lead bucket.

    Buckets with too little history fall back to the all-leads fit.
    """

    def __init__(self):
        self.coef_: dict[int, tuple[float, float]] = {}
        self.global_: tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def _ols(x, y) -> tuple[float, float]:
        slope, intercept = np.polyfit(x, y, 1)
        return float(intercept), float(slope)

    def fit(self, forecast, observed, lead_h) -> MOSCorrector:
        f, o = np.asarray(forecast, float), np.asarray(observed, float)
        ok = np.isfinite(f) & np.isfinite(o)
        if ok.sum() < 10:
            return self
        self.global_ = self._ols(f[ok], o[ok])
        buckets = lead_bucket(lead_h)
        for b in range(len(LEAD_LABELS)):
            m = ok & (buckets == b)
            if m.sum() >= MIN_SAMPLES:
                self.coef_[b] = self._ols(f[m], o[m])
        return self

    def transform(self, forecast, lead_h) -> np.ndarray:
        f = np.asarray(forecast, float)
        buckets = lead_bucket(lead_h)
        a = np.array([self.coef_.get(b, self.global_)[0] for b in range(len(LEAD_LABELS))])[buckets]
        s = np.array([self.coef_.get(b, self.global_)[1] for b in range(len(LEAD_LABELS))])[buckets]
        return np.clip(a + s * f, 0.0, None)


class QuantileMapper:
    """Empirical quantile mapping: makes the forecast distribution match the observed one.

    Fixes distribution shape (e.g. too few high-wind hours) that a linear fit cannot,
    at the cost of not reducing random error.
    """

    def __init__(self, n_quantiles: int = 101):
        self.n_quantiles = n_quantiles
        self.fq_ = self.oq_ = None

    def fit(self, forecast, observed, lead_h=None) -> QuantileMapper:
        f, o = np.asarray(forecast, float), np.asarray(observed, float)
        f, o = f[np.isfinite(f)], o[np.isfinite(o)]
        if len(f) < 10 or len(o) < 10:
            return self
        probs = np.linspace(0, 1, self.n_quantiles)
        fq, oq = np.quantile(f, probs), np.quantile(o, probs)
        self.fq_ = fq + np.arange(len(fq)) * 1e-9  # strictly increasing for interp
        self.oq_ = oq
        return self

    def transform(self, forecast, lead_h=None) -> np.ndarray:
        f = np.asarray(forecast, float)
        if self.fq_ is None:
            return f
        return np.where(np.isfinite(f), np.interp(f, self.fq_, self.oq_), np.nan)


class Identity:
    def fit(self, forecast, observed, lead_h=None):
        return self

    def transform(self, forecast, lead_h=None):
        return np.asarray(forecast, float)


def make_corrector(method: str):
    try:
        return {"mos": MOSCorrector, "quantile_mapping": QuantileMapper, "none": Identity}[method]()
    except KeyError:
        raise ValueError(f"unknown bias correction '{method}'") from None

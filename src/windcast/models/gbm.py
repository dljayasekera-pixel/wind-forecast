"""Gradient-boosted models: LightGBM quantile regression and an XGBoost point model."""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


def resolve_jobs(n_jobs: int = 0) -> int:
    """0 means every available core, capped at 8. Asking for more threads than cores slows boosting."""
    return n_jobs if n_jobs and n_jobs > 0 else min(8, os.cpu_count() or 1)


def qcol(q: float, prefix: str = "lgbm") -> str:
    return f"{prefix}_q{int(round(q * 100)):02d}"


class LGBMQuantile:
    """One LightGBM model per quantile, trained on the pinball loss.

    Rows for every lead are pooled with ``lead_h`` as a feature, so the trees learn
    how much to trust persistence against NWP as the horizon grows. Predictions are
    sorted across quantiles to remove any crossing.
    """

    def __init__(self, quantiles: list[float], params: dict | None = None, seed: int = 7,
                 n_jobs: int = 0):
        self.quantiles, self.params, self.seed = list(quantiles), dict(params or {}), seed
        self.n_jobs = resolve_jobs(n_jobs)
        self.models_: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y) -> LGBMQuantile:
        self.features_ = list(X.columns)
        # Every quantile starts boosting from the median. LightGBM's default start is the
        # target quantile itself; for an extreme quantile of a bounded target (many hours
        # at zero or at rated power) that start sits on the bound, all gradients are equal,
        # no split has gain and training stops at a constant.
        self.base_ = float(np.median(y))
        init = np.full(len(X), self.base_)
        for q in self.quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed,
                                      deterministic=True, force_row_wise=True, n_jobs=self.n_jobs,
                                      verbose=-1, **self.params)
            self.models_[q] = model.fit(X, y, init_score=init)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = self.base_ + np.column_stack([self.models_[q].predict(X[self.features_])
                                            for q in self.quantiles])
        raw = np.clip(np.sort(raw, axis=1), 0.0, 1.0)
        return pd.DataFrame(raw, index=X.index, columns=[qcol(q) for q in self.quantiles])

    def importance(self, q: float = 0.5) -> pd.Series:
        booster = self.models_[q].booster_
        gain = booster.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=self.features_).sort_values(ascending=False)


class XGBPoint:
    """XGBoost squared-error model: a second boosted learner with a different loss
    and tree-growing policy, which gives the stacked ensemble something to diversify over."""

    def __init__(self, params: dict | None = None, seed: int = 7, n_jobs: int = 0):
        self.params, self.seed, self.n_jobs = dict(params or {}), seed, resolve_jobs(n_jobs)

    def fit(self, X: pd.DataFrame, y) -> XGBPoint:
        self.features_ = list(X.columns)
        self.model_ = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                       random_state=self.seed, n_jobs=self.n_jobs, **self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(X[self.features_]), 0.0, 1.0)


class QuantileCalibrator:
    """Conformal-style recalibration of quantile forecasts on a hold-out set.

    For each quantile level q and lead bucket, the forecast is shifted by the
    empirical q-quantile of (actual - forecast) on the hold-out, so that the stated
    level is met there. Boosted quantile models are typically too narrow in the
    tails; this corrects the level without retraining them.
    """

    def __init__(self, quantiles: list[float], prefix: str = "lgbm", min_samples: int = 150):
        self.quantiles, self.prefix, self.min_samples = list(quantiles), prefix, min_samples
        self.shift_: dict[int, np.ndarray] = {}
        self.global_ = np.zeros(len(self.quantiles))

    @property
    def cols(self) -> list[str]:
        return [qcol(q, self.prefix) for q in self.quantiles]

    def _shifts(self, resid: np.ndarray) -> np.ndarray:
        return np.array([np.quantile(resid[:, i], q) for i, q in enumerate(self.quantiles)])

    def fit(self, q_preds: pd.DataFrame, y, lead_h) -> QuantileCalibrator:
        from windcast.utils import LEAD_LABELS, lead_bucket

        resid = np.asarray(y, float)[:, None] - q_preds[self.cols].to_numpy(float)
        ok = np.isfinite(resid).all(axis=1)
        if ok.sum() < self.min_samples:
            return self
        self.global_ = self._shifts(resid[ok])
        buckets = lead_bucket(lead_h)
        for b in range(len(LEAD_LABELS)):
            m = ok & (buckets == b)
            if m.sum() >= self.min_samples:
                self.shift_[b] = self._shifts(resid[m])
        return self

    def transform(self, q_preds: pd.DataFrame, lead_h) -> pd.DataFrame:
        from windcast.utils import LEAD_LABELS, lead_bucket

        table = np.vstack([self.shift_.get(b, self.global_) for b in range(len(LEAD_LABELS))])
        adj = q_preds[self.cols].to_numpy(float) + table[lead_bucket(lead_h)]
        adj = np.clip(np.sort(adj, axis=1), 0.0, 1.0)
        return pd.DataFrame(adj, index=q_preds.index, columns=self.cols)

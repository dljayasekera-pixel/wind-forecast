from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from windcast.backtest import metrics as M
from windcast.backtest.splits import rolling_origin_folds, split
from windcast.models.baselines import Climatology, arima_forecasts
from windcast.models.ensemble import StackedBlender
from windcast.models.gbm import LGBMQuantile, QuantileCalibrator, XGBPoint, resolve_jobs


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.normal(0, 1, n)})
    y = np.clip(X["a"] + rng.normal(0, 0.1 + 0.2 * X["a"], n), 0, 1)  # noise grows with a
    return X, y


def test_pinball_and_crps_reference_values():
    assert M.pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert M.pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)
    y = np.array([0.3, 0.6])
    qs = [0.1, 0.5, 0.9]
    perfect = np.column_stack([y, y, y])
    assert M.crps_from_quantiles(y, perfect, qs) == pytest.approx(0.0)
    wide = np.column_stack([y - 0.3, y, y + 0.3])
    assert M.crps_from_quantiles(y, wide, qs) > 0
    assert M.skill(0.5, 1.0) == pytest.approx(0.5) and np.isnan(M.skill(1.0, 0.0))
    assert M.bias([1.0, 1.0], [1.5, 1.5]) == pytest.approx(0.5)


def test_dm_test_detects_the_better_forecast_and_stays_quiet_otherwise():
    rng = np.random.default_rng(3)
    e_good, e_bad = rng.normal(0, 1.0, 800), rng.normal(0, 1.5, 800)
    stat, p = M.dm_test(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.001
    stat, p = M.dm_test(e_good, rng.normal(0, 1.0, 800), h=4)
    assert p > 0.01
    assert np.isnan(M.dm_test([0.1] * 5, [0.2] * 5)[0])


def test_metric_table_scores_every_model_on_the_same_rows():
    df = pd.DataFrame({"y": [0.5] * 10, "persistence": [0.7] * 10,
                       "m": [0.6] * 5 + [np.nan] * 5, "lead_h": [1.0] * 10})
    out = M.metric_table(df, ["persistence", "m"], ["lead_h"]).set_index("model")
    assert out.loc["m", "n"] == 5 and out.loc["m", "mae"] == pytest.approx(0.1)
    assert out.loc["m", "skill_mae"] == pytest.approx(0.5)
    assert out.loc["persistence", "skill_rmse"] == pytest.approx(0.0)


def test_reliability_counts_ties_as_half():
    df = pd.DataFrame({"y": [0.0] * 50 + [0.5] * 50, "lgbm_q50": [0.0] * 50 + [0.6] * 50})
    rel = M.reliability_table(df, [0.5])
    assert rel["observed"].iloc[0] == pytest.approx(0.75)


def test_rolling_folds_have_no_label_overlap(frame):
    folds = rolling_origin_folds(frame["issue_time"].min(), frame["issue_time"].max(), 60, 20)
    assert len(folds) >= 2
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end == b.test_start
    for fold in folds:
        train, test = split(frame, fold)
        assert train["valid_time"].max() < fold.test_start <= test["issue_time"].min()
        assert test["issue_time"].max() < fold.test_end
    with pytest.raises(ValueError):
        rolling_origin_folds("2024-01-01", "2024-01-20", 60, 20)


def test_sliding_window_moves_the_training_start():
    folds = rolling_origin_folds("2024-01-01", "2024-06-01", 60, 20, expanding=False)
    assert folds[1].train_start > folds[0].train_start
    assert all((f.test_start - f.train_start).days == 60 for f in folds)


def test_lgbm_quantiles_are_ordered_calibrated_and_reproducible():
    X, y = _toy()
    Xt, yt = _toy(seed=9)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    params = {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 8, "min_child_samples": 40}
    model = LGBMQuantile(qs, params, seed=1, n_jobs=1).fit(X, y)
    pred = model.predict(Xt)
    assert (np.diff(pred.to_numpy(), axis=1) >= 0).all()
    assert pred.to_numpy().min() >= 0 and pred.to_numpy().max() <= 1
    coverage = ((yt >= pred["lgbm_q05"]) & (yt <= pred["lgbm_q95"])).mean()
    assert 0.75 < coverage < 0.97  # raw boosted quantiles run narrow; QuantileCalibrator fixes the level
    # Heteroscedastic by construction: intervals must widen with 'a'.
    width = pred["lgbm_q95"] - pred["lgbm_q05"]
    assert width[Xt["a"].between(0.6, 0.8)].mean() > width[Xt["a"].between(0.1, 0.3)].mean()
    again = LGBMQuantile(qs, params, seed=1, n_jobs=1).fit(X, y).predict(Xt)
    pd.testing.assert_frame_equal(pred, again)
    assert model.importance().index[0] == "a"


def test_xgb_point_model_learns_and_clips():
    X, y = _toy()
    Xt, yt = _toy(seed=5)
    pred = XGBPoint({"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1}, n_jobs=1).fit(X, y).predict(Xt)
    assert np.mean(np.abs(pred - yt)) < np.mean(np.abs(yt.mean() - yt)) * 0.7
    assert pred.min() >= 0 and pred.max() <= 1
    assert resolve_jobs(3) == 3 and 1 <= resolve_jobs(0) <= 8


def test_calibrator_restores_coverage_of_overconfident_quantiles():
    rng = np.random.default_rng(4)
    n, qs = 6000, [0.05, 0.5, 0.95]
    y = np.clip(0.5 + rng.normal(0, 0.15, n), 0, 1)
    narrow = pd.DataFrame({"lgbm_q05": 0.45, "lgbm_q50": 0.5, "lgbm_q95": 0.55}, index=range(n))
    lead = np.full(n, 12.0)
    cal = QuantileCalibrator(qs).fit(narrow[:3000], y[:3000], lead[:3000])
    fixed = cal.transform(narrow[3000:], lead[3000:])
    before = ((y[3000:] >= 0.45) & (y[3000:] <= 0.55)).mean()
    after = ((y[3000:] >= fixed["lgbm_q05"]) & (y[3000:] <= fixed["lgbm_q95"])).mean()
    assert before < 0.35 and 0.86 < after < 0.94
    assert (np.diff(fixed.to_numpy(), axis=1) >= 0).all()
    untouched = QuantileCalibrator(qs).transform(narrow, np.full(n, 12.0))  # never fitted: identity
    pd.testing.assert_frame_equal(untouched, narrow)


def test_stacker_weights_follow_lead_and_tolerate_missing_members():
    rng = np.random.default_rng(5)
    n = 4000
    y = rng.uniform(0, 1, n)
    lead = np.repeat([2.0, 60.0], n // 2)
    short = lead < 6
    preds = pd.DataFrame({
        "persistence": y + rng.normal(0, np.where(short, 0.03, 0.4)),
        "nwp": y + rng.normal(0, np.where(short, 0.15, 0.08)),
    })
    stacker = StackedBlender(["persistence", "nwp"]).fit(preds, y, lead)
    table = stacker.table()
    assert table.loc["2h", "persistence"] > 0.8 and table.loc["60h", "nwp"] > 0.8
    assert np.allclose(table.sum(axis=1), 1.0, atol=0.01)
    out = stacker.predict(preds, lead)
    assert np.mean((out - y) ** 2) <= min(np.mean((preds[c] - y) ** 2) for c in preds) + 1e-6
    holes = preds.copy()
    holes.loc[:9, "persistence"] = np.nan
    assert np.allclose(stacker.predict(holes, lead)[:10], np.clip(holes["nwp"][:10], 0, 1))


def test_stacker_uses_one_weight_set_per_lead_for_intra_hour_products():
    rng = np.random.default_rng(6)
    y = rng.uniform(0, 1, 3000)
    lead = np.tile([1 / 6, 0.5, 1.0], 1000)
    preds = pd.DataFrame({"a": y + rng.normal(0, 0.02 + 0.2 * lead), "b": y + rng.normal(0, 0.1, 3000)})
    stacker = StackedBlender(["a", "b"]).fit(preds, y, lead)
    table = stacker.table()
    assert stacker.per_lead_ and len(table) == 3
    assert table["a"].iloc[0] > table["a"].iloc[-1]


def test_climatology_by_hour(frame):
    qs = [0.05, 0.5, 0.95]
    clim = Climatology(qs).fit(frame)
    mean, q = clim.predict(frame), clim.predict_quantiles(frame)
    assert np.isfinite(mean).all() and list(q.columns) == ["clim_q05", "clim_q50", "clim_q95"]
    assert (q["clim_q05"] <= q["clim_q50"]).all() and (q["clim_q50"] <= q["clim_q95"]).all()
    hour0 = frame[frame["valid_time"].dt.hour == 0]
    assert clim.predict(hour0.head(1))[0] == pytest.approx(hour0["y"].mean())


def test_arima_baseline_on_an_ar1_series():
    rng = np.random.default_rng(7)
    n, phi = 1500, 0.9
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.05)
    series = pd.Series(np.clip(0.5 + x, 0, 1), index=pd.date_range("2024-01-01", periods=n, freq="h"))
    origins = series.index[1200:1400:10]
    fc = arima_forecasts(series, series.index[1000], origins, [1, 6], order=(1, 0, 0), history_steps=200)
    assert len(fc) == len(origins) * 2 and fc["arima"].between(0, 1).all()
    one = fc[fc["lead_step"] == 1].set_index("issue_time")["arima"]
    actual = series.shift(-1).reindex(one.index)
    persistence = series.reindex(one.index)
    # One step ahead an AR(1) forecast should be about as good as persistence, and far better than the mean.
    assert (one - actual).abs().mean() < (persistence - actual).abs().mean() * 1.15
    assert (one - actual).abs().mean() < (series.mean() - actual).abs().mean() * 0.7

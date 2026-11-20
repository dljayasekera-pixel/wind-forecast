from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from windcast.models.bundle import fit_power_curve
from windcast.providers.bias_correction import MOSCorrector, QuantileMapper, make_corrector
from windcast.providers.blend import ProviderBlender
from windcast.providers.fallback import apply_fallback
from windcast.providers.scorecard import accuracy_scorecard, delivery_scorecard, provider_impact
from windcast.providers.stack import WeatherStack


def test_mos_removes_linear_bias_per_lead_bucket():
    rng = np.random.default_rng(0)
    obs = rng.uniform(2, 15, 6000)
    lead = np.repeat([3.0, 36.0], 3000)
    fcst = np.where(lead < 6, 1.2 * obs + 1.0, 0.8 * obs - 0.5) + rng.normal(0, 0.3, 6000)
    mos = MOSCorrector().fit(fcst, obs, lead)
    corrected = mos.transform(fcst, lead)
    for value in (3.0, 36.0):
        m = lead == value
        assert abs(np.mean(corrected[m] - obs[m])) < 0.02
        assert np.mean(np.abs(corrected[m] - obs[m])) < np.mean(np.abs(fcst[m] - obs[m])) / 2
    # A lead bucket never seen in training uses the pooled fit rather than failing.
    assert np.isfinite(mos.transform([8.0], [120.0])).all()
    assert (mos.transform([-5.0], [3.0]) >= 0).all()


def test_quantile_mapping_matches_the_observed_distribution():
    rng = np.random.default_rng(1)
    obs = rng.weibull(2.0, 20_000) * 8
    fcst = obs * 1.3 + 1.0
    qm = QuantileMapper().fit(fcst, obs)
    mapped = qm.transform(fcst)
    for q in (0.1, 0.5, 0.9):
        assert np.quantile(mapped, q) == pytest.approx(np.quantile(obs, q), rel=0.02)
    assert np.isnan(qm.transform([np.nan])[0])


def test_make_corrector_rejects_unknown_method():
    assert make_corrector("none").transform([1.0, 2.0]).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        make_corrector("magic")


@pytest.mark.parametrize("method", ["nnls", "inverse_mse"])
def test_blender_favours_the_better_provider_and_beats_it(method):
    rng = np.random.default_rng(2)
    truth = rng.uniform(3, 14, 5000)
    good, poor = truth + rng.normal(0, 0.5, 5000), truth + rng.normal(0, 2.0, 5000)
    F, lead = np.column_stack([good, poor]), np.full(5000, 12.0)
    blender = ProviderBlender(["good", "poor"], method).fit(F, truth, lead)
    w = blender.weights_[1]
    assert w[0] > 0.8 and w.sum() == pytest.approx(1.0)
    rmse = lambda x: np.sqrt(np.mean((x - truth) ** 2))  # noqa: E731
    assert rmse(blender.transform(F, lead)) <= rmse(good) + 1e-9


def test_blender_renormalises_when_a_provider_is_missing():
    blender = ProviderBlender(["a", "b", "c"], "equal")
    F = np.array([[6.0, 9.0, 12.0], [6.0, np.nan, 12.0], [np.nan, np.nan, 12.0], [np.nan, np.nan, np.nan]])
    out = blender.transform(F, np.full(4, 12.0))
    assert out[:3].tolist() == [9.0, 9.0, 12.0] and np.isnan(out[3])


def test_fallback_tiers(frame, cfg):
    providers = list(cfg["providers"])
    sample = frame.dropna(subset=[f"{p}_ws_hub" for p in providers]).head(4).copy()
    sample.iloc[1, sample.columns.get_loc("icon_age_h")] = 100.0   # stale
    sample.iloc[2, sample.columns.get_loc("gfs_ws_hub")] = np.nan  # missing
    for p in providers:                                            # everything down
        sample.iloc[3, sample.columns.get_loc(f"{p}_ws_hub")] = np.nan
    out = apply_fallback(sample, providers, max_age_h=30.0)
    assert out["fallback_tier"].tolist() == [0, 1, 1, 2]
    assert out["n_providers"].tolist() == [3, 2, 2, 0]
    assert np.isnan(out["icon_ws_hub"].iloc[1]) and np.isnan(out["icon_ramp_3h"].iloc[1])
    assert sample["icon_ws_hub"].notna().iloc[1]  # input frame untouched


def test_weather_stack_improves_on_raw_nwp_and_survives_outage(frame, farm, cfg):
    providers = list(cfg["providers"])
    cut = pd.Timestamp("2024-03-15")
    train, test = frame[frame["valid_time"] < cut], frame[frame["issue_time"] >= cut]
    pc = fit_power_curve(farm, cut, cfg["site"])
    stack = WeatherStack(providers, pc).fit(train)
    out = stack.transform(test)
    obs = out["tgt_ws_obs"]
    raw_mae = (out["gfs_ws_hub"] - obs).abs().mean()
    bc_mae = (out["gfs_ws_bc"] - obs).abs().mean()
    blend_mae = (out["blend_ws"] - obs).abs().mean()
    assert bc_mae < raw_mae            # the synthetic GFS has a planted +7% / +0.3 m/s bias
    assert blend_mae < min((out[f"{p}_ws_bc"] - obs).abs().mean() for p in providers)
    assert out["blend_pc"].between(0, 1.05).all() or out["blend_pc"].isna().any()

    dead = test.copy()
    for p in providers:
        dead[f"{p}_ws_hub"] = np.nan
        dead[f"{p}_ws_hub_now"] = np.nan
    out_dead = stack.transform(dead)
    assert (out_dead["fallback_tier"] == 2).all() and out_dead["blend_pc"].isna().all()


def test_scorecards(frame, farm, data, cfg):
    providers = list(cfg["providers"])
    delivery = delivery_scorecard(data["nwp"], cfg["providers"])
    assert set(delivery.index) == set(providers)
    assert delivery.loc["ecmwf", "latency_mean_h"] > delivery.loc["icon", "latency_mean_h"]
    assert delivery.loc["icon", "max_lead_h"] < delivery.loc["gfs", "max_lead_h"]
    assert (delivery["run_coverage"] <= 1.0).all() and delivery.loc["icon", "run_coverage"] < 0.99
    assert delivery.loc["icon", "longest_gap_h"] > 24  # the planted multi-day outage

    cut = pd.Timestamp("2024-03-15")
    pc = fit_power_curve(farm, cut, cfg["site"])
    stack = WeatherStack(providers, pc).fit(frame[frame["valid_time"] < cut])
    ev = stack.transform(frame[frame["issue_time"] >= cut])
    acc = accuracy_scorecard(ev, providers)
    gfs = acc[acc["provider"] == "gfs"]
    assert (gfs["ws_bias_raw"] > 0.3).all() and (gfs["ws_bias_bc"].abs() < gfs["ws_bias_raw"].abs()).all()
    impact = provider_impact(ev, providers)
    assert (impact["nmae_vs_blend_pct"] > 0).mean() > 0.8  # single providers are worse than the blend

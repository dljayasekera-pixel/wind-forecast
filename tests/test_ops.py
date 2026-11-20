"""Root-cause analysis, monitoring, SQL layer, config."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from windcast import storage
from windcast.config import leads, load_config
from windcast.monitoring import health
from windcast.monitoring.cloudwatch import CloudWatchEmitter
from windcast.rca import attribution
from windcast.utils import config_hash, lead_bucket_label, to_ns


def _preds(**over) -> pd.DataFrame:
    base = {"issue_time": pd.Timestamp("2024-06-01"), "valid_time": pd.Timestamp("2024-06-02"), "lead_h": 24.0,
            "y": 0.5, "stack": 0.5, "p_obs": 0.5, "blend_pc": 0.5, "tgt_availability": 1.0,
            "tgt_derated_frac": 0.0, "tgt_dq": False, "fallback_tier": 0}
    cases = {
        "weather": {"y": 0.9, "p_obs": 0.9, "blend_pc": 0.4, "stack": 0.42},     # wind was stronger than forecast
        "model": {"y": 0.6, "p_obs": 0.6, "blend_pc": 0.6, "stack": 0.2},        # NWP was right, model ignored it
        "operations": {"y": 0.3, "p_obs": 0.9, "blend_pc": 0.88, "stack": 0.9,   # turbines were down
                       "tgt_availability": 0.33},
        "data_quality": {"y": np.nan, "tgt_dq": True},
        "provider_outage": {"y": 0.8, "p_obs": 0.8, "blend_pc": np.nan, "stack": 0.3, "fallback_tier": 2},
    }
    rows = [{**base, **v, "case": k} for k, v in cases.items()]
    return pd.DataFrame(rows).assign(**over)


def test_rca_components_sum_to_the_error_and_label_each_case():
    dec = attribution.decompose(_preds())
    ok = dec[["c_weather", "c_model", "c_operations"]].notna().all(axis=1) & dec["error"].notna()
    total = dec.loc[ok, ["c_weather", "c_model", "c_operations"]].sum(axis=1)
    assert np.allclose(total, dec.loc[ok, "error"])
    assert dec["cause"].astype(str).tolist() == ["weather", "model", "operations", "data_quality", "provider_outage"]
    misses = attribution.big_misses(dec, 0.25)
    assert misses["error"].abs().is_monotonic_decreasing and len(misses) == 4
    summary = attribution.summary(dec, 0.25).set_index("cause")
    assert summary["share_of_big_misses"].sum() == pytest.approx(1.0)
    assert summary.loc["weather", "n_big_misses"] == 1


def test_rca_by_lead_excludes_undecomposable_rows():
    out = attribution.variance_shares(attribution.decompose(_preds()))
    assert list(out.columns) == ["weather_mae", "model_mae", "operations_mae"]
    assert out.loc[24.0, "operations_mae"] == pytest.approx(0.6 / 3)


def _health_inputs(now):
    idx = pd.date_range(now - pd.Timedelta(days=20), now, freq="h")
    farm = pd.DataFrame({"power_norm": 0.4}, index=idx)
    nwp = pd.DataFrame({"provider": ["gfs", "icon"], "available_time": [now - pd.Timedelta("3h")] * 2})
    rng = np.random.default_rng(0)
    n = 400
    y = rng.uniform(0.1, 0.9, n)
    preds = pd.DataFrame({"valid_time": pd.date_range(now - pd.Timedelta(days=10), periods=n, freq="30min"),
                          "y": y, "stack": y + rng.normal(0, 0.05, n), "persistence": y + rng.normal(0, 0.2, n),
                          "lgbm_q05": y - 0.15, "lgbm_q95": y + 0.15, "tgt_clean": True, "fallback_tier": 0})
    return farm, nwp, preds


def test_healthy_system_raises_no_alerts():
    now = pd.Timestamp("2024-08-01")
    farm, nwp, preds = _health_inputs(now)
    thresholds = load_config()["monitoring"]
    perf = health.performance(preds, [0.05, 0.5, 0.95], now)
    assert perf["nmae"] == pytest.approx(0.04, abs=0.01) and perf["skill_vs_persistence"] > 0.5
    fresh = health.freshness(farm, nwp, now)
    assert fresh == {"scada_lag_h": 0.0, "provider_lag_h_gfs": 3.0, "provider_lag_h_icon": 3.0}
    alerts = health.evaluate_alerts(perf, fresh, thresholds, baseline_nmae=0.04)
    assert alerts == [] and not health.should_retrain(alerts)


def test_alert_rules_separate_feed_problems_from_model_problems():
    now = pd.Timestamp("2024-08-01")
    farm, nwp, preds = _health_inputs(now)
    thresholds = load_config()["monitoring"]

    farm.loc[farm.index > now - pd.Timedelta("8h"), "power_norm"] = np.nan
    nwp["available_time"] = now - pd.Timedelta("40h")
    fresh = health.freshness(farm, nwp, now)
    perf = health.performance(preds, [0.05, 0.5, 0.95], now)
    alerts = health.evaluate_alerts(perf, fresh, thresholds, 0.04)
    names = {a.name: a.severity for a in alerts}
    assert names == {"scada_stale": "CRITICAL", "provider_stale_gfs": "WARN", "provider_stale_icon": "WARN",
                     "all_providers_stale": "CRITICAL"}
    assert not health.should_retrain(alerts)  # a data outage is not a reason to retrain

    preds["stack"] = preds["y"] + 0.12
    preds["lgbm_q95"] = preds["y"] + 0.01
    preds["lgbm_q05"] = preds["y"] + 0.005
    perf = health.performance(preds, [0.05, 0.5, 0.95], now)
    alerts = health.evaluate_alerts(perf, {"scada_lag_h": 0.0}, thresholds, 0.04)
    assert {a.name for a in alerts} == {"nmae_degraded", "bias_drift", "intervals_too_narrow"}
    assert health.should_retrain(alerts)
    assert health.performance(preds.head(0), [0.05, 0.5, 0.95], now) == {"n": 0}


def test_cloudwatch_emitter_batches_and_skips_non_finite():
    class Client:
        def __init__(self):
            self.calls = []

        def put_metric_data(self, Namespace, MetricData):
            self.calls.append((Namespace, MetricData))

    client = Client()
    metrics = {f"m{i}": float(i) for i in range(45)} | {"bad": float("inf"), "text": "x", "n": 7}
    sent = CloudWatchEmitter("WindCast", "kelmarsh", client=client, dry_run=False).put(metrics)
    assert sent == 46 and [len(c[1]) for c in client.calls] == [20, 20, 6]
    datum = client.calls[0][1][0]
    assert datum["Dimensions"] == [{"Name": "Site", "Value": "kelmarsh"}]
    assert CloudWatchEmitter(dry_run=True).put({"nmae": 0.1}) == 1  # dry run: nothing sent, nothing raised


def test_sql_layer_round_trip(cfg, data, flagged, farm):
    conn = storage.connect()
    site = cfg["site"]["name"]
    storage.load_site(conn, cfg)
    assert storage.load_farm(conn, farm, site) == len(farm)
    assert storage.load_scada(conn, flagged[0].head(5000), site) == 5000
    nwp = data["nwp"][data["nwp"]["valid_time"] < "2024-02-01"]
    assert storage.load_nwp(conn, nwp, site) == len(nwp)
    assert storage.load_nwp(conn, nwp, site) == len(nwp)  # reload is idempotent
    assert conn.execute("SELECT COUNT(*) FROM nwp_forecasts").fetchone()[0] == len(nwp)

    idx = farm.index[200:400]
    preds = pd.DataFrame({"issue_time": idx - pd.Timedelta("24h"), "valid_time": idx, "lead_h": 24.0,
                          "persistence": farm["power_norm"].shift(24).reindex(idx).to_numpy(),
                          "stack": farm["power_norm"].reindex(idx).to_numpy() + 0.01,
                          "lgbm_q05": 0.0, "lgbm_q95": 1.1})
    storage.load_forecasts(conn, preds, site, ["lgbm_q05", "lgbm_q95"], ("persistence", "stack"))

    params = {
        "provider_error_by_lead": {}, "provider_latency_and_gaps": {}, "daily_energy": {},
        "latest_forecast_as_of": {"as_of": "2024-01-15 06:00:00"},
        "ramp_events": {"threshold": 0.4}, "forecast_skill_by_lead": {}, "rolling_7d_mae": {},
        "provider_disagreement": {"provider_a": "gfs", "provider_b": "icon", "lead_h": 24},
        "interval_coverage": {},
    }
    assert set(params) == set(storage.named_queries())
    results = {name: storage.query(conn, name, site=site, **extra) for name, extra in params.items()}
    assert all(len(df) > 0 for df in results.values()), {k: len(v) for k, v in results.items()}

    as_of = results["latest_forecast_as_of"]
    assert (as_of["run_age_h"] > 0).all() and as_of.groupby(["provider", "valid_time"]).size().max() == 1
    skill = results["forecast_skill_by_lead"].set_index("model")
    assert skill.loc["stack", "nmae"] == pytest.approx(0.01, abs=1e-3)
    assert skill.loc["stack", "skill_vs_persistence"] > 0.5
    assert results["interval_coverage"]["coverage"].iloc[0] == 1.0
    gfs = results["provider_error_by_lead"].query("provider == 'gfs'")
    assert (gfs["ws_bias"] > 0).all()
    with pytest.raises(KeyError):
        storage.query(conn, "nope")


def test_config_defaults_overrides_and_validation(tmp_path):
    cfg = load_config()
    assert cfg["site"]["n_turbines"] == 6 and len(leads(cfg)) == 19
    path = tmp_path / "c.yaml"
    path.write_text("providers:\n  only: {openmeteo_model: gfs_seamless}\nforecast:\n  issue_freq: 6h\n")
    custom = load_config(path)
    assert list(custom["providers"]) == ["only"]            # providers replace, they do not merge
    assert custom["forecast"]["issue_freq"] == "6h" and custom["forecast"]["resolution"] == "1h"
    assert config_hash(custom) != config_hash(cfg) and config_hash(cfg) == config_hash(load_config())
    for bad in ({"forecast": {"quantiles": [0.1, 0.9]}}, {"forecast": {"leads": ["90min"]}},
                {"data": {"source": "oracle"}}, {"providers": {}}, {"forecast": {"issue_freq": "90min"}}):
        with pytest.raises(ValueError):
            load_config(overrides=bad)


def test_time_helpers():
    assert to_ns("2024-01-01T06:00:00+02:00") == pd.Timestamp("2024-01-01 04:00")
    assert str(to_ns(pd.Series(["2024-01-01"])).dtype) == "datetime64[ns]"
    assert lead_bucket_label([1, 6, 7, 24, 25, 96, 97]).tolist() == ["0-6h", "0-6h", "6-24h", "6-24h", "24-48h",
                                                                   "48-96h", "96h+"]

"""End-to-end tests: the whole pipeline on a small synthetic site. Marked slow (they train models)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from windcast import pipeline, registry, storage
from windcast.cli import main

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    c = copy.deepcopy(cfg)
    c["data"]["root"], c["paths"]["artifacts"] = str(root / "data"), str(root / "artifacts")
    return c, root, pipeline.run_all(c)


def test_backtest_beats_baselines_and_is_calibrated(run):
    c, root, out = run
    s = out["backtest"]
    assert s["n_folds"] >= 2
    assert s["stack_nmae"] < 0.6 * s["persistence_nmae"]
    assert s["stack_skill_rmse_vs_persistence"] > 0.3
    assert abs(s["stack_bias"]) < 0.03
    assert s["crpss_vs_climatology"] > 0.4
    assert 0.75 < s["interval_coverage"] < 0.98

    tables = root / "artifacts" / "backtest" / "tables"
    by_lead = pd.read_csv(tables / "metrics_by_lead.csv")
    stack = by_lead[by_lead["model"] == "stack"].set_index("lead_h")
    pers = by_lead[by_lead["model"] == "persistence"].set_index("lead_h")
    # With only two months of training the first hour is a near tie with persistence; beyond it, a clear win.
    assert (stack["mae"] <= pers["mae"] * 1.10).all()
    assert (stack.loc[3.0:, "mae"] < 0.7 * pers.loc[3.0:, "mae"]).all()
    assert stack.loc[48.0, "mae"] > stack.loc[1.0, "mae"]     # error grows with horizon
    assert "arima" in set(by_lead["model"])
    dm = pd.read_csv(tables / "dm_tests.csv")
    day_ahead = dm[(dm["lead_h"] == 24) & (dm["versus"] == "persistence")].iloc[0]
    assert day_ahead["dm_stat"] < 0 and day_ahead["p_value"] < 0.01
    for name in ("provider_accuracy", "provider_delivery", "rca_summary", "reliability", "blend_weights",
                 "stack_weights", "feature_importance", "metrics_by_season", "metrics_by_regime"):
        assert len(pd.read_csv(tables / f"{name}.csv")) > 0


def test_no_test_row_was_predicted_by_a_model_that_saw_it(run):
    c, root, _ = run
    preds = pd.read_parquet(root / "artifacts" / "backtest" / "preds.parquet")
    assert not preds.duplicated(["issue_time", "valid_time"]).any()
    starts = preds.groupby("fold")["issue_time"].min()
    assert starts.is_monotonic_increasing
    q = [c_ for c_ in preds if c_.startswith("lgbm_q")]
    assert (np.diff(preds[q].to_numpy(), axis=1) >= 0).all()
    assert preds[["stack", *q]].stack().between(0, 1).all()


def test_planted_curtailment_is_attributed_to_operations(run):
    c, root, _ = run
    preds = pd.read_parquet(root / "artifacts" / "backtest" / "preds.parquet")
    from windcast.rca import attribution

    dec = attribution.decompose(preds)
    constrained = dec[(dec["tgt_availability"] < 0.7) & (dec["error"].abs() > 0.25) & (dec["fallback_tier"] < 2)]
    if len(constrained) >= 5:
        assert (constrained["cause"] == "operations").mean() > 0.5
    decomposable = dec.dropna(subset=["error", "c_weather", "c_model", "c_operations"])
    total = decomposable[["c_weather", "c_model", "c_operations"]].sum(axis=1)
    assert np.allclose(total, decomposable["error"], atol=1e-9)


def test_registry_and_inference(run):
    c, root, out = run
    models = root / "artifacts" / "models"
    bundle, meta = registry.load(models)
    assert meta["version"] == out["train"]["version"] and meta["metrics"]["n_folds"] >= 2
    assert meta["lightgbm"] and len(meta["features"]) == len(bundle.features)

    fc = pipeline.predict(c, "2024-04-10T06:00")
    assert len(fc) == len(c["forecast"]["leads"]) and (fc["issue_time"] == pd.Timestamp("2024-04-10 06:00")).all()
    assert fc["p50_mw"].between(0, 12.3).all() and (fc["q05_mw"] <= fc["q95_mw"]).all()
    assert (fc["model_version"] == meta["version"]).all()
    again = pipeline.predict(c, "2024-04-10T06:00")
    pd.testing.assert_frame_equal(fc, again)  # same inputs, same forecast

    # A second, unpromoted version can be loaded explicitly (shadow) while 'latest' is unchanged.
    shadow = pipeline.train(c, promote=False)["version"]
    assert shadow != meta["version"]
    assert registry.load(models)[1]["version"] == meta["version"]
    assert registry.load(models, shadow)[1]["version"] == shadow
    with pytest.raises(FileNotFoundError):
        registry.promote_version(models, "does-not-exist")


def test_inference_degrades_instead_of_failing(run, monkeypatch):
    c, root, _ = run
    real = pipeline._load_curated

    def no_weather(cfg):
        farm, nwp = real(cfg)
        return farm, nwp.iloc[0:0]

    monkeypatch.setattr(pipeline, "_load_curated", no_weather)
    fc = pipeline.predict(c, "2024-04-10T06:00")
    assert (fc["fallback_tier"] == 2).all() and fc["p50_mw"].notna().all()

    def stale_scada(cfg):
        farm, nwp = real(cfg)
        farm = farm.copy()
        farm.loc[farm.index > "2024-04-09 00:00", ["power_norm", "ws_obs", "availability", "derated_frac"]] = np.nan
        return farm, nwp

    monkeypatch.setattr(pipeline, "_load_curated", stale_scada)
    fc = pipeline.predict(c, "2024-04-10T06:00")
    assert fc["p50_mw"].notna().all() and (fc["fallback_tier"] == 0).all()


def test_monitor_and_sql_outputs(run):
    c, root, out = run
    report = json.loads((root / "artifacts" / "monitoring" / "health.json").read_text())
    assert report["performance"]["n"] > 20 and "scada_lag_h" in report["freshness"]
    assert isinstance(report["retrain_recommended"], bool)
    late = pipeline.monitor(c, now="2024-05-03T00:00")  # two days after the last data point
    assert "scada_stale" in {a["name"] for a in late["alerts"]}

    conn = storage.connect(root / "artifacts" / "windcast.db")
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("sites", "farm_hourly", "scada_10min", "nwp_forecasts", "forecasts")}
    assert counts["sites"] == 1 and all(v > 0 for v in counts.values())
    skill = storage.query(conn, "forecast_skill_by_lead", site=c["site"]["name"])
    sql_stack = skill[(skill["model"] == "stack") & (skill["lead_h"] == 24)]["nmae"].iloc[0]
    csv = pd.read_csv(root / "artifacts" / "backtest" / "tables" / "metrics_by_lead.csv")
    py_stack = csv[(csv["model"] == "stack") & (csv["lead_h"] == 24)]["mae"].iloc[0]
    assert sql_stack == pytest.approx(py_stack, abs=2e-3)  # SQL and Python agree on the same metric


def test_backtest_is_reproducible(run):
    c, root, out = run
    again = pipeline.backtest(c)
    assert again == out["backtest"]


def test_cli_returns_nonzero_on_failure_and_zero_on_success(run, tmp_path, capsys):
    c, root, _ = run
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(c))
    assert main(["monitor", "--config", str(path)]) == 0
    assert "retrain_recommended" in capsys.readouterr().out
    assert main(["predict", "--config", str(path), "--version", "missing"]) == 1


def test_dashboard_renders(run, monkeypatch):
    c, root, _ = run
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WINDCAST_ARTIFACTS", c["paths"]["artifacts"])
    app = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) == 5 and len(at.tabs) == 5

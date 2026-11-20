"""Pipeline steps. Each is idempotent, reads and writes Parquet, and maps to one
state in the Step Functions workflow (infra/statemachine.asl.json).

    ingest   -> data/raw        sources to canonical frames
    prepare  -> data/curated    QC, farm aggregation
    backtest -> artifacts/backtest
    train    -> artifacts/models
    predict  -> artifacts/forecasts
    monitor  -> artifacts/monitoring
    load-sql -> artifacts/windcast.db
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from windcast import registry, storage
from windcast.backtest.engine import run_backtest
from windcast.config import capacity_kw
from windcast.features.build import build_frame
from windcast.models.bundle import fit_bundle, predict_bundle
from windcast.monitoring import health
from windcast.monitoring.cloudwatch import CloudWatchEmitter
from windcast.scada.aggregate import to_farm
from windcast.scada.qc import clean_scada, qc_summary
from windcast.utils import get_logger, to_ns

log = get_logger("windcast.pipeline")


def _dirs(cfg: dict) -> dict[str, Path]:
    root, art = Path(cfg["data"]["root"]), Path(cfg["paths"]["artifacts"])
    d = {"raw": root / "raw", "curated": root / "curated", "backtest": art / "backtest",
         "models": art / "models", "forecasts": art / "forecasts", "monitoring": art / "monitoring"}
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    d["db"] = art / "windcast.db"
    return d


def ingest(cfg: dict) -> dict:
    """Pull SCADA, status log and NWP into the raw zone."""
    d = _dirs(cfg)
    if cfg["data"]["source"] == "synthetic":
        from windcast.connectors import synthetic

        data = synthetic.generate(cfg)
        data["events"].to_parquet(d["raw"] / "events.parquet", index=False)
        scada, status, nwp = data["scada"], data["status"], data["nwp"]
    else:
        from windcast.connectors import kelmarsh, openmeteo

        scada, status = kelmarsh.fetch(cfg, d["raw"] / "kelmarsh")
        nwp = openmeteo.fetch_all(cfg, d["raw"] / "openmeteo")
    scada.to_parquet(d["raw"] / "scada.parquet", index=False)
    status.to_parquet(d["raw"] / "status.parquet", index=False)
    nwp.to_parquet(d["raw"] / "nwp.parquet", index=False)
    out = {"scada_rows": len(scada), "status_rows": len(status), "nwp_rows": len(nwp),
           "providers": sorted(nwp["provider"].unique())}
    log.info("ingest done", extra={"ctx": out})
    return out


def prepare(cfg: dict) -> dict:
    """QC the turbine data and build the farm-level series."""
    d = _dirs(cfg)
    scada = pd.read_parquet(d["raw"] / "scada.parquet")
    status = pd.read_parquet(d["raw"] / "status.parquet")
    flagged, curve = clean_scada(scada, status, cfg["site"])
    farm = to_farm(flagged, cfg["site"], cfg["forecast"]["resolution"])
    flagged.to_parquet(d["curated"] / "scada_flagged.parquet", index=False)
    farm.to_parquet(d["curated"] / "farm.parquet")
    qc_summary(flagged).to_csv(d["curated"] / "qc_summary.csv")
    (d["curated"] / "turbine_power_curve.json").write_text(json.dumps(curve.to_dict()))
    out = {"farm_rows": len(farm), "capacity_factor": round(float(farm["power_norm"].mean()), 4),
           "bad_data_share": round(float(farm["dq"].mean()), 4),
           "operating_ok_share": round(float(flagged["operating_ok"].mean()), 4)}
    log.info("prepare done", extra={"ctx": out})
    return out


def _load_curated(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _dirs(cfg)
    farm = pd.read_parquet(d["curated"] / "farm.parquet")
    farm.index = to_ns(farm.index)
    nwp = pd.read_parquet(d["raw"] / "nwp.parquet")
    for col in ("run_time", "valid_time", "available_time"):
        nwp[col] = to_ns(nwp[col])
    return farm, nwp


def backtest(cfg: dict) -> dict:
    """Rolling-origin backtest; writes predictions and every report table."""
    d = _dirs(cfg)
    farm, nwp = _load_curated(cfg)
    result = run_backtest(farm, nwp, cfg)
    result.preds.to_parquet(d["backtest"] / "preds.parquet", index=False)
    result.eval_frame.to_parquet(d["backtest"] / "eval_frame.parquet", index=False)
    tables = d["backtest"] / "tables"
    tables.mkdir(exist_ok=True)
    for name, table in result.tables.items():
        table.to_csv(tables / f"{name}.csv", index=False)
    overall = result.tables["metrics_overall_clean"].set_index("model")
    prob = result.tables["prob_overall"].iloc[0]
    summary = {
        "n_folds": len(result.folds), "n_predictions": int(len(result.preds)),
        "test_start": str(result.folds[0].test_start), "test_end": str(result.folds[-1].test_end),
        "stack_nmae": float(overall.loc["stack", "mae"]), "stack_nrmse": float(overall.loc["stack", "rmse"]),
        "stack_bias": float(overall.loc["stack", "bias"]),
        "stack_skill_rmse_vs_persistence": float(overall.loc["stack", "skill_rmse"]),
        "lgbm_q50_nmae": float(overall.loc["lgbm_q50", "mae"]),
        "persistence_nmae": float(overall.loc["persistence", "mae"]),
        "crps": float(prob["crps"]), "crpss_vs_climatology": float(prob["crpss"]),
        "interval_coverage": float(prob["coverage"]), "nominal_coverage": float(prob["nominal_coverage"]),
    }
    (d["backtest"] / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("backtest done", extra={"ctx": summary})
    return summary


def train(cfg: dict, promote: bool = True) -> dict:
    """Fit on all available history and register the bundle."""
    d = _dirs(cfg)
    farm, nwp = _load_curated(cfg)
    frame = build_frame(farm, nwp, cfg)
    train_end = farm.index.max() + pd.Timedelta(cfg["forecast"]["resolution"])
    bundle = fit_bundle(frame[frame["valid_time"] < train_end], farm, cfg, train_end)
    summary_path = d["backtest"] / "summary.json"
    metrics = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    version = registry.save(bundle, cfg, d["models"], metrics, promote)
    out = {"version": version, "promoted": promote, **bundle.meta}
    log.info("train done", extra={"ctx": out})
    return out


def predict(cfg: dict, issue_time: str | None = None, version: str | None = None) -> pd.DataFrame:
    """Issue one forecast. Uses only SCADA stamped <= issue_time and NWP available by then.

    Degrades rather than fails: stale SCADA drops the persistence features, missing
    providers drop out of the blend, and both are logged with the fallback tier.
    """
    d = _dirs(cfg)
    farm, nwp = _load_curated(cfg)
    bundle, meta = registry.load(d["models"], version)
    res = pd.Timedelta(cfg["forecast"]["resolution"])
    t0 = to_ns(pd.Timestamp(issue_time)) if issue_time else farm.index.max()
    t0 = t0.floor(res)
    farm_now = farm[farm.index <= t0]
    nwp_now = nwp[nwp["available_time"] <= t0]
    if farm_now.empty:
        raise ValueError(f"no SCADA at or before {t0}")
    good = farm_now.index[farm_now["power_norm"].notna()]
    scada_lag_h = float((t0 - good.max()) / pd.Timedelta("1h")) if len(good) else float("inf")
    if scada_lag_h > cfg["monitoring"]["max_scada_lag_h"]:
        log.warning("SCADA stale at issue time", extra={"ctx": {"issue_time": str(t0), "lag_h": scada_lag_h}})

    frame = build_frame(farm_now, nwp_now, cfg, issue_times=[t0])
    preds, _ = predict_bundle(bundle, frame)
    cap_mw = capacity_kw(cfg) / 1000.0
    qcols = [c for c in preds.columns if c.startswith("lgbm_q")]
    out = preds[["issue_time", "valid_time", "lead_h", "fallback_tier", "n_providers"]].copy()
    out["p50_norm"] = preds["stack"]
    out["p50_mw"] = preds["stack"] * cap_mw
    for col in qcols:
        out[f"{col.removeprefix('lgbm_')}_mw"] = preds[col] * cap_mw
    out["model_version"] = meta["version"]
    tier = int(out["fallback_tier"].max())
    if tier > 0:
        log.warning("forecast issued in fallback mode", extra={"ctx": {"issue_time": str(t0), "tier": tier}})
    path = d["forecasts"] / f"forecast_{t0:%Y%m%dT%H%M}.parquet"
    out.to_parquet(path, index=False)
    log.info("forecast issued", extra={"ctx": {"issue_time": str(t0), "rows": len(out), "path": str(path),
                                               "max_fallback_tier": tier, "scada_lag_h": scada_lag_h}})
    return out


def monitor(cfg: dict, now: str | None = None, emit: bool = False) -> dict:
    """Score recent forecasts, check feed freshness, raise alerts, publish metrics."""
    d = _dirs(cfg)
    farm, nwp = _load_curated(cfg)
    preds = pd.read_parquet(d["backtest"] / "preds.parquet")
    mcfg = cfg["monitoring"]
    now_ts = to_ns(pd.Timestamp(now)) if now else farm.index.max()
    perf = health.performance(preds, cfg["forecast"]["quantiles"], now_ts, mcfg["window_days"])
    fresh = health.freshness(farm[farm.index <= now_ts], nwp, now_ts)
    summary_path = d["backtest"] / "summary.json"
    baseline = json.loads(summary_path.read_text())["stack_nmae"] if summary_path.exists() else None
    alerts = health.evaluate_alerts(perf, fresh, mcfg, baseline)
    report = {"now": str(now_ts), "performance": perf, "freshness": fresh, "baseline_nmae": baseline,
              "alerts": [a.to_dict() for a in alerts], "retrain_recommended": health.should_retrain(alerts)}
    (d["monitoring"] / "health.json").write_text(json.dumps(report, indent=2, default=str))
    emitter = CloudWatchEmitter(mcfg["namespace"], cfg["site"]["name"], dry_run=not emit)
    emitter.put({**perf, **fresh, "alerts": len(alerts)})
    for alert in alerts:
        log.warning("alert", extra={"ctx": alert.to_dict()})
    log.info("monitor done", extra={"ctx": {"alerts": len(alerts), "retrain": report["retrain_recommended"]}})
    return report


def load_sql(cfg: dict, nwp_days: int = 60) -> dict:
    """Load curated data and backtest forecasts into the SQL store."""
    d = _dirs(cfg)
    site = cfg["site"]["name"]
    farm, nwp = _load_curated(cfg)
    conn = storage.connect(d["db"])
    storage.load_site(conn, cfg)
    counts = {"farm_hourly": storage.load_farm(conn, farm, site)}
    flagged_path = d["curated"] / "scada_flagged.parquet"
    if flagged_path.exists():
        counts["scada_10min"] = storage.load_scada(conn, pd.read_parquet(flagged_path), site)
    recent = nwp[nwp["valid_time"] > nwp["valid_time"].max() - pd.Timedelta(days=nwp_days)]
    counts["nwp_forecasts"] = storage.load_nwp(conn, recent, site)
    preds_path = d["backtest"] / "preds.parquet"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        qcols = [c for c in preds.columns if c.startswith("lgbm_q")]
        counts["forecasts"] = storage.load_forecasts(conn, preds, site, qcols)
    conn.close()
    log.info("sql load done", extra={"ctx": counts})
    return counts


STEPS = {"ingest": ingest, "prepare": prepare, "backtest": backtest, "train": train,
         "monitor": monitor, "load-sql": load_sql}


def run_all(cfg: dict) -> dict:
    out = {name: STEPS[name](cfg) for name in ("ingest", "prepare", "backtest", "train")}
    # Demo forecast: issued far enough before the end of the data that every lead can be checked.
    farm_end = pd.read_parquet(_dirs(cfg)["curated"] / "farm.parquet").index.max()
    issue = farm_end - max(pd.to_timedelta(cfg["forecast"]["leads"]))
    out["predict_rows"] = len(predict(cfg, str(issue)))
    out["monitor"] = monitor(cfg)
    out["load-sql"] = load_sql(cfg)
    return out

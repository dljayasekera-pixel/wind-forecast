"""Operational health: recent accuracy, data freshness, alert rules, retraining trigger."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from windcast.backtest import metrics as M


@dataclass
class Alert:
    name: str
    severity: str  # WARN | CRITICAL
    value: float
    threshold: float
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def performance(preds: pd.DataFrame, quantiles: list[float], now: pd.Timestamp,
                window_days: int = 14, model: str = "stack") -> dict:
    """Accuracy over forecasts whose valid time fell in the trailing window, on unconstrained hours."""
    recent = preds[(preds["valid_time"] > now - pd.Timedelta(days=window_days))
                   & (preds["valid_time"] <= now) & preds["tgt_clean"]]
    recent = recent.dropna(subset=["y", model, "persistence"])
    if len(recent) < 20:
        return {"n": int(len(recent))}
    lo, hi = (f"lgbm_q{int(round(q * 100)):02d}" for q in (quantiles[0], quantiles[-1]))
    y = recent["y"]
    return {
        "n": int(len(recent)),
        "nmae": M.mae(y, recent[model]),
        "nrmse": M.rmse(y, recent[model]),
        "bias": M.bias(y, recent[model]),
        "skill_vs_persistence": M.skill(M.rmse(y, recent[model]), M.rmse(y, recent["persistence"])),
        "interval_coverage": float(((y >= recent[lo]) & (y <= recent[hi])).mean()),
        "fallback_share": float((recent["fallback_tier"] > 0).mean()),
    }


def freshness(farm: pd.DataFrame, nwp: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Hours since the last good SCADA sample and since each provider last delivered."""
    good = farm.index[farm["power_norm"].notna()]
    out = {"scada_lag_h": float((now - good.max()) / pd.Timedelta("1h")) if len(good) else np.inf}
    seen = nwp[nwp["available_time"] <= now].groupby("provider")["available_time"].max()
    for provider in nwp["provider"].unique():
        last = seen.get(provider)
        out[f"provider_lag_h_{provider}"] = float((now - last) / pd.Timedelta("1h")) if last is not None else np.inf
    return out


def evaluate_alerts(perf: dict, fresh: dict, thresholds: dict, baseline_nmae: float | None) -> list[Alert]:
    """Apply the alert rules. Data-feed problems are CRITICAL because they need action now;
    accuracy drift is WARN and feeds the retraining decision."""
    alerts: list[Alert] = []
    if fresh["scada_lag_h"] > thresholds["max_scada_lag_h"]:
        alerts.append(Alert("scada_stale", "CRITICAL", fresh["scada_lag_h"], thresholds["max_scada_lag_h"],
                            "SCADA feed is stale; forecasts are running without persistence features"))
    lags = {k.removeprefix("provider_lag_h_"): v for k, v in fresh.items() if k.startswith("provider_lag_h_")}
    stale = {p: v for p, v in lags.items() if v > thresholds["max_provider_lag_h"]}
    for provider, lag in stale.items():
        alerts.append(Alert(f"provider_stale_{provider}", "WARN", lag, thresholds["max_provider_lag_h"],
                            f"NWP provider {provider} has not delivered; blend is running without it"))
    if lags and len(stale) == len(lags):
        alerts.append(Alert("all_providers_stale", "CRITICAL", min(stale.values()),
                            thresholds["max_provider_lag_h"], "no NWP provider is delivering"))
    if perf.get("n", 0) >= 20:
        if baseline_nmae and perf["nmae"] > thresholds["nmae_degradation_ratio"] * baseline_nmae:
            alerts.append(Alert("nmae_degraded", "WARN", perf["nmae"],
                                thresholds["nmae_degradation_ratio"] * baseline_nmae,
                                "recent nMAE is well above the backtest baseline"))
        if abs(perf["bias"]) > thresholds["max_abs_bias"]:
            alerts.append(Alert("bias_drift", "WARN", perf["bias"], thresholds["max_abs_bias"],
                                "persistent over- or under-forecasting"))
        if perf["interval_coverage"] < thresholds["min_coverage_90"]:
            alerts.append(Alert("intervals_too_narrow", "WARN", perf["interval_coverage"],
                                thresholds["min_coverage_90"], "prediction intervals are under-covering"))
    return alerts


def should_retrain(alerts: list[Alert]) -> bool:
    """Retrain when the model, not the data feed, is the problem."""
    return any(a.name in {"nmae_degraded", "bias_drift", "intervals_too_narrow"} for a in alerts)

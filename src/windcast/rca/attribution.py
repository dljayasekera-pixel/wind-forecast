"""Attribute forecast error to weather, model, operations or data quality.

Every error is split exactly (the three terms sum to actual - forecast):

    actual - forecast = (actual - p_obs)      plant term
                      + (p_obs  - p_nwp)      weather term
                      + (p_nwp  - forecast)   model term

    p_obs  power the farm should make from the wind that was measured (power curve)
    p_nwp  power implied by the blended, bias-corrected NWP wind (power curve)

The plant term is then labelled: 'operations' when turbines were unavailable or
derated at the valid time, otherwise 'conversion' (power-curve residual, which is
model error). Rows with bad telemetry are 'data_quality' and rows forecast with
no NWP at all are 'provider_outage'; neither can be decomposed meaningfully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAUSES = ["weather", "model", "operations", "data_quality", "provider_outage"]


def decompose(preds: pd.DataFrame, model: str = "stack") -> pd.DataFrame:
    out = preds[["issue_time", "valid_time", "lead_h", "y", model, "p_obs", "blend_pc",
                 "tgt_availability", "tgt_derated_frac", "tgt_dq", "fallback_tier"]].copy()
    out = out.rename(columns={model: "forecast", "blend_pc": "p_nwp"})
    out["error"] = out["y"] - out["forecast"]
    plant = out["y"] - out["p_obs"]
    constrained = (out["tgt_availability"] < 0.999) | (out["tgt_derated_frac"] > 0.001)
    out["c_operations"] = plant.where(constrained, 0.0)
    out["c_conversion"] = plant.where(~constrained, 0.0)
    out["c_weather"] = out["p_obs"] - out["p_nwp"]
    out["c_model"] = (out["p_nwp"] - out["forecast"]) + out["c_conversion"]

    comps = out[["c_weather", "c_model", "c_operations"]].abs()
    dominant = comps.idxmax(axis=1, skipna=True).str.removeprefix("c_")
    undecomposable = comps.isna().any(axis=1)
    cause = np.where(out["tgt_dq"], "data_quality",
                     np.where(out["fallback_tier"] == 2, "provider_outage",
                              np.where(undecomposable, "data_quality", dominant)))
    out["cause"] = pd.Categorical(cause, CAUSES)
    return out


def big_misses(decomposed: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """Rows whose absolute error exceeds ``threshold`` of installed capacity."""
    df = decomposed[decomposed["error"].abs() >= threshold]
    return df.sort_values("error", key=np.abs, ascending=False).reset_index(drop=True)


def summary(decomposed: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """Per cause: share of big misses and of total absolute error."""
    df = decomposed.dropna(subset=["error"])
    misses = df[df["error"].abs() >= threshold]
    total_abs = df["error"].abs().sum()
    rows = []
    for cause in CAUSES:
        sub, sub_miss = df[df["cause"] == cause], misses[misses["cause"] == cause]
        rows.append({"cause": cause, "n_rows": len(sub), "n_big_misses": len(sub_miss),
                     "share_of_big_misses": len(sub_miss) / max(1, len(misses)),
                     "share_of_abs_error": sub["error"].abs().sum() / total_abs if total_abs else np.nan,
                     "mean_abs_error": sub["error"].abs().mean() if len(sub) else np.nan})
    return pd.DataFrame(rows).round(4)


def variance_shares(decomposed: pd.DataFrame, by_lead: bool = True) -> pd.DataFrame:
    """Mean absolute size of each component by lead time: shows where weather error takes over."""
    df = decomposed[~decomposed["tgt_dq"] & (decomposed["fallback_tier"] < 2)]
    cols = ["c_weather", "c_model", "c_operations"]
    grouped = df.groupby("lead_h")[cols] if by_lead else df[cols]
    out = grouped.agg(lambda s: s.abs().mean())
    return out.rename(columns=lambda c: c.removeprefix("c_") + "_mae").round(4)

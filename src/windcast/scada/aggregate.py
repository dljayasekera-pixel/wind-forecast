"""Turbine SCADA to a farm-level series at the forecast resolution.

Output timestamps mark the END of each averaging period, so the value stamped
``t`` is known at ``t``. Every feature and target relies on that convention.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SCADA_STEP = pd.Timedelta("10min")


def to_farm(flagged: pd.DataFrame, site: dict, resolution: str = "1h") -> pd.DataFrame:
    n_t, rated = int(site["n_turbines"]), float(site["rated_kw"])
    df = flagged
    ok = df["qc_ok"]
    rad = np.deg2rad(df["wind_dir"])
    work = pd.DataFrame({
        "time": df["time"],
        "p": df["power_kw"].where(ok).clip(lower=0),
        "ws": df["wind_speed"].where(ok),
        "temp": df["ambient_temp"],
        "sin": np.sin(rad).where(ok), "cos": np.cos(rad).where(ok),
        "rep": ok.astype(float),
        "avail": (ok & ~df["flag_outage"]).astype(float),
        "der": (ok & df["flag_derated"] & ~df["flag_outage"]).astype(float),
    })
    g = work.groupby("time").agg(p=("p", "sum"), ws=("ws", "mean"), temp=("temp", "mean"),
                                 sin=("sin", "mean"), cos=("cos", "mean"), rep=("rep", "sum"),
                                 avail=("avail", "sum"), der=("der", "sum"))
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq=SCADA_STEP))
    g["rep"] = g["rep"].fillna(0)
    enough = g["rep"] >= math.ceil(0.5 * n_t)
    # Scale up over reporting turbines so a comms drop on one unit is not read as lost power.
    g["power_norm"] = (g["p"] / (g["rep"] * rated)).where(enough)
    g["availability"] = (g["avail"] / g["rep"]).where(enough)
    g["derated_frac"] = (g["der"] / g["rep"]).where(enough)
    g["coverage"] = g["rep"] / n_t
    g["ws"] = g["ws"].where(enough)
    g.index = g.index + SCADA_STEP  # period start -> period end

    res = pd.Timedelta(resolution)
    steps = int(res / SCADA_STEP)
    cols = ["power_norm", "ws", "temp", "sin", "cos", "availability", "derated_frac", "coverage"]
    if steps > 1:
        r = g[cols].resample(res, label="right", closed="right")
        out = r.mean()
        count = r["power_norm"].count()
        out.loc[count < math.ceil(0.5 * steps), ["power_norm", "ws", "availability", "derated_frac"]] = np.nan
    else:
        out = g[cols].copy()
    out["wd_obs"] = np.rad2deg(np.arctan2(out["sin"], out["cos"])) % 360
    out = out.rename(columns={"ws": "ws_obs", "temp": "temp_obs"}).drop(columns=["sin", "cos"])
    out["dq"] = out["power_norm"].isna() | (out["coverage"] < 0.8)
    out["power_kw"] = out["power_norm"] * rated * n_t
    out.index.name = "time"
    return out

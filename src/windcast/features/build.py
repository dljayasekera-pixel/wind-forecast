"""Build the modelling frame: one row per (issue_time, lead).

Column groups
    keys      issue_time, valid_time, lead_h
    target    y (normalised farm power averaged over the period ending at valid_time)
    tgt_*     things only known after the fact (observed wind, availability, flags).
              Never features; used for fitting bias correction on history, for
              filtering and for root-cause analysis.
    obs_*     SCADA-derived values known at issue_time
    <p>_*     raw NWP from provider p, selected as of issue_time
    vt_*      calendar terms of the valid time

Fold-dependent transforms (bias correction, blending, power curve) are NOT applied
here. They are fitted on training rows only, in ``windcast.providers.stack``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.features.nwp import prepare_nwp, select_latest
from windcast.utils import to_ns

KEYS = ["issue_time", "valid_time", "lead_h"]


def _grid(farm: pd.DataFrame, resolution: str) -> pd.DataFrame:
    full = pd.date_range(farm.index.min(), farm.index.max(), freq=pd.Timedelta(resolution))
    return farm.reindex(full)


def obs_features(farm: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """SCADA features indexed by issue time. Row t uses only farm rows stamped <= t."""
    fc = cfg["forecast"]
    g = _grid(farm, fc["resolution"])
    p = g["power_norm"]
    limit = int(fc["stale_limit_steps"])
    idx = np.arange(len(g))
    last_seen = pd.Series(np.where(p.notna(), idx, np.nan), index=g.index).ffill()
    out = pd.DataFrame(index=g.index)
    out["obs_power_t0"] = p.ffill(limit=limit)
    out["obs_stale_steps"] = idx - last_seen
    for k in fc["lag_steps"]:
        out[f"obs_power_lag{k}"] = p.shift(k)
    for w in fc["roll_steps"]:
        roll = p.rolling(w, min_periods=max(1, w // 2))
        out[f"obs_roll_mean{w}"] = roll.mean()
        out[f"obs_roll_std{w}"] = roll.std()
    out["obs_ramp1"] = p - p.shift(1)
    out["obs_ramp3"] = p - p.shift(3)
    out["obs_ws_t0"] = g["ws_obs"].ffill(limit=limit)
    out["obs_avail_t0"] = g["availability"].ffill(limit=limit)
    out["obs_derated_t0"] = g["derated_frac"].ffill(limit=limit)
    return out


def calendar_features(valid_time: pd.Series, seasonal: bool = False) -> pd.DataFrame:
    hour = valid_time.dt.hour + valid_time.dt.minute / 60
    doy = valid_time.dt.dayofyear
    out = pd.DataFrame(index=valid_time.index)
    for k in (1, 2):
        out[f"vt_hour_sin{k}"] = np.sin(2 * np.pi * k * hour / 24)
        out[f"vt_hour_cos{k}"] = np.cos(2 * np.pi * k * hour / 24)
    if seasonal:
        out["vt_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["vt_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def build_frame(farm: pd.DataFrame, nwp: pd.DataFrame, cfg: dict,
                issue_times=None) -> pd.DataFrame:
    """Assemble the frame. ``issue_times`` restricts to given origins (used at inference)."""
    fc = cfg["forecast"]
    res = pd.Timedelta(fc["resolution"])
    grid = _grid(farm, fc["resolution"])
    if issue_times is None:
        issue_freq = pd.Timedelta(fc["issue_freq"])
        origin = grid.index.min().normalize()
        issue_times = grid.index[((grid.index - origin) % issue_freq) == pd.Timedelta(0)]
    issue_times = to_ns(pd.DatetimeIndex(issue_times))
    leads = pd.to_timedelta(fc["leads"])

    # Valid times come from exact timedeltas; going through float hours would smear
    # sub-hourly leads by nanoseconds and miss the target grid.
    frame = pd.DataFrame({"issue_time": np.repeat(issue_times.to_numpy(), len(leads))})
    frame["valid_time"] = frame["issue_time"] + np.tile(leads.to_numpy(), len(issue_times))
    frame["lead_h"] = (frame["valid_time"] - frame["issue_time"]) / pd.Timedelta("1h")
    frame = frame[KEYS]

    # Targets and after-the-fact context at the valid time.
    tgt = grid.reindex(frame["valid_time"])
    frame["y"] = tgt["power_norm"].to_numpy()
    frame["tgt_ws_obs"] = tgt["ws_obs"].to_numpy()
    frame["tgt_temp_obs"] = tgt["temp_obs"].to_numpy()
    frame["tgt_availability"] = tgt["availability"].to_numpy()
    frame["tgt_derated_frac"] = tgt["derated_frac"].to_numpy()
    frame["tgt_dq"] = tgt["dq"].astype("boolean").fillna(True).astype(bool).to_numpy()
    frame["tgt_clean"] = (frame["y"].notna() & ~frame["tgt_dq"]
                          & (frame["tgt_availability"] >= cfg["clean"]["min_availability"])
                          & (frame["tgt_derated_frac"] <= cfg["clean"]["max_derated_frac"]))

    obs = obs_features(farm, cfg).reindex(frame["issue_time"])
    for col in obs.columns:
        frame[col] = obs[col].to_numpy()

    # NWP is hourly; sub-hourly valid times use the enclosing hour.
    valid_hour = frame["valid_time"].dt.ceil("h") if res < pd.Timedelta("1h") else frame["valid_time"]
    issue_hour = frame["issue_time"].dt.floor("h")
    hub = float(cfg["site"]["hub_height_m"])
    for name in cfg["providers"]:
        prepared = prepare_nwp(nwp[nwp["provider"] == name], hub)
        at_valid = select_latest(prepared, frame["issue_time"], valid_hour)
        for col in at_valid.columns:
            frame[f"{name}_{col}"] = at_valid[col].to_numpy()
        # Forecast for 'now': lets the model see how wrong each provider currently is.
        at_issue = select_latest(prepared, frame["issue_time"], issue_hour)
        frame[f"{name}_ws_hub_now"] = at_issue["ws_hub"].to_numpy()
        frame[f"{name}_rho_now"] = at_issue["rho"].to_numpy()

    cal = calendar_features(frame["valid_time"], bool(fc.get("seasonal_terms", False)))
    frame = pd.concat([frame, cal], axis=1)
    return frame.reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs: everything except keys, the target, tgt_* and fold bookkeeping."""
    drop = set(KEYS) - {"lead_h"} | {"y", "fold", "fallback_tier"}
    return [c for c in frame.columns
            if c not in drop and not c.startswith("tgt_") and not c.endswith("_rho_now")]

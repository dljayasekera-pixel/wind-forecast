"""NWP preparation and as-of selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.scada.power_curve import air_density, extrapolate_to_hub, shear_exponent
from windcast.utils import to_ns

NWP_FEATURES = ["ws_hub", "rho", "temp", "wd_sin", "wd_cos", "shear", "ramp_3h"]


def prepare_nwp(nwp: pd.DataFrame, hub_height_m: float) -> pd.DataFrame:
    """Derive hub-height wind, shear, density, direction components and ramp for one provider."""
    df = nwp.sort_values(["run_time", "valid_time"]).reset_index(drop=True)
    alpha = shear_exponent(df["ws_10m"], df["ws_100m"], 10.0, 100.0)
    out = pd.DataFrame({
        "run_time": df["run_time"], "valid_time": df["valid_time"],
        "available_time": df["available_time"],
        "ws_hub": extrapolate_to_hub(df["ws_100m"], alpha, 100.0, hub_height_m),
        "shear": alpha,
        "rho": air_density(df["pressure_hpa"], df["temp_2m"]),
        "temp": df["temp_2m"],
        "wd_sin": np.sin(np.deg2rad(df["wd_100m"])),
        "wd_cos": np.cos(np.deg2rad(df["wd_100m"])),
    })
    # Ramp = change in forecast wind over the next 3 h. Computed inside a run where
    # runs are resolved; for pseudo-runs (one row per run) along valid time at fixed lead.
    lead = out["valid_time"] - out["run_time"]
    key = out["run_time"] if out["run_time"].nunique() < len(out) / 2 else lead
    ahead = out.groupby(key, sort=False)["ws_hub"].shift(-3)
    ahead_valid = out.groupby(key, sort=False)["valid_time"].shift(-3)
    aligned = (ahead_valid - out["valid_time"]) == pd.Timedelta("3h")
    out["ramp_3h"] = (ahead - out["ws_hub"]).where(aligned)
    return out


def select_latest(prepared: pd.DataFrame, issue_time, valid_time) -> pd.DataFrame:
    """For each (issue_time, valid_time) pair, the newest forecast with available_time <= issue_time.

    Returns a frame aligned row-for-row with the inputs: NWP_FEATURES plus ``age_h``
    (issue_time - run_time). Rows with no usable forecast are NaN.
    """
    left = pd.DataFrame({"issue_time": to_ns(pd.Series(np.asarray(issue_time))),
                         "valid_time": to_ns(pd.Series(np.asarray(valid_time)))})
    left["_row"] = np.arange(len(left))
    right = prepared.copy()
    for col in ("run_time", "valid_time", "available_time"):
        right[col] = to_ns(right[col])
    merged = pd.merge_asof(left.sort_values("issue_time"), right.sort_values("available_time"),
                           left_on="issue_time", right_on="available_time", by="valid_time",
                           allow_exact_matches=True)
    merged = merged.sort_values("_row").reset_index(drop=True)
    merged["age_h"] = (merged["issue_time"] - merged["run_time"]) / pd.Timedelta("1h")
    return merged[NWP_FEATURES + ["age_h"]]

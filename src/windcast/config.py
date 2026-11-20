"""YAML config loading with defaults and validation."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

DEFAULTS: dict = {
    "site": {
        "name": "kelmarsh",
        "latitude": 52.4016,
        "longitude": -0.9431,
        "elevation_m": 145.0,
        "hub_height_m": 78.5,
        "rated_kw": 2050.0,
        "n_turbines": 6,
        "cut_in_ms": 3.0,
        "cut_out_ms": 24.0,
    },
    "data": {
        "source": "synthetic",  # synthetic | kelmarsh
        "start": "2024-01-01",
        "end": "2025-01-01",
        "root": "data",
        "seed": 42,
    },
    "providers": {
        "gfs": {"openmeteo_model": "gfs_seamless", "cycle_h": 6},
        "ecmwf": {"openmeteo_model": "ecmwf_ifs025", "cycle_h": 12},
        "icon": {"openmeteo_model": "icon_seamless", "cycle_h": 6},
    },
    "weather": {
        "bias_correction": "mos",  # mos | quantile_mapping | none
        "blend": "nnls",  # nnls | inverse_mse | equal
        "max_age_h": 30.0,  # a provider older than this at issue time is treated as down
    },
    "forecast": {
        "resolution": "1h",
        "issue_freq": "3h",
        "leads": ["1h", "2h", "3h", "4h", "6h", "9h", "12h", "18h", "24h", "30h", "36h",
                  "42h", "48h", "60h", "72h", "96h", "120h", "144h", "168h"],
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "lag_steps": [1, 2, 3, 6],
        "roll_steps": [3, 6, 24],
        "stale_limit_steps": 6,
        # Day-of-year terms only help once training covers several years; with less,
        # trees extrapolate a season they have never seen. Off by default.
        "seasonal_terms": False,
    },
    # An hour is 'clean' (used to train and score) when the farm was essentially unconstrained.
    # Requiring every one of 36 turbine-samples to be perfect discards most real windy hours.
    "clean": {"min_availability": 0.95, "max_derated_frac": 0.05},
    "backtest": {
        "initial_train_days": 150,
        "test_days": 30,
        "step_days": 30,
        "stack_holdout_frac": 0.2,
        "arima": {"enabled": True, "order": [2, 0, 1], "max_lead_h": 24,
                  "max_origins_per_fold": 120, "history_steps": 400},
    },
    "model": {
        "seed": 7,
        "n_jobs": 0,  # 0 = use all cores (capped at 8)
        "lgbm": {"n_estimators": 200, "learning_rate": 0.06, "num_leaves": 15,
                 "min_child_samples": 150, "subsample": 0.8, "subsample_freq": 1,
                 "colsample_bytree": 0.6, "reg_lambda": 5.0},
        "xgb": {"n_estimators": 200, "learning_rate": 0.06, "max_depth": 5,
                "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 50,
                "reg_lambda": 5.0},
    },
    "monitoring": {
        "window_days": 14,
        "nmae_degradation_ratio": 1.25,
        "max_abs_bias": 0.05,
        "min_coverage_90": 0.80,
        "max_scada_lag_h": 3.0,
        "max_provider_lag_h": 18.0,
        "big_miss_threshold": 0.25,
        "namespace": "WindCast",
    },
    "paths": {"artifacts": "artifacts"},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict) and key != "providers":
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def validate(cfg: dict) -> None:
    qs = cfg["forecast"]["quantiles"]
    if sorted(qs) != list(qs) or not all(0 < q < 1 for q in qs) or 0.5 not in qs:
        raise ValueError("forecast.quantiles must be sorted, inside (0,1), and include 0.5")
    if not cfg["providers"]:
        raise ValueError("at least one NWP provider is required")
    if cfg["data"]["source"] not in ("synthetic", "kelmarsh"):
        raise ValueError("data.source must be 'synthetic' or 'kelmarsh'")
    res = pd.Timedelta(cfg["forecast"]["resolution"])
    for lead in cfg["forecast"]["leads"]:
        td = pd.Timedelta(lead)
        if td <= pd.Timedelta(0) or td % res != pd.Timedelta(0):
            raise ValueError(f"lead {lead} must be a positive multiple of the resolution")
    if pd.Timedelta(cfg["forecast"]["issue_freq"]) % res != pd.Timedelta(0):
        raise ValueError("forecast.issue_freq must be a multiple of the resolution")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config on top of DEFAULTS. ``providers`` replaces rather than merges."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    validate(cfg)
    return cfg


def capacity_kw(cfg: dict) -> float:
    return cfg["site"]["rated_kw"] * cfg["site"]["n_turbines"]


def leads(cfg: dict) -> list[pd.Timedelta]:
    return [pd.Timedelta(x) for x in cfg["forecast"]["leads"]]

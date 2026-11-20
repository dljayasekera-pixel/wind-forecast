"""Fit and apply the full model chain. Shared by backtesting, training and inference,
so the code that is evaluated is the code that is deployed."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from windcast.features.build import KEYS, feature_columns
from windcast.models.baselines import Climatology, persistence
from windcast.models.ensemble import StackedBlender
from windcast.models.gbm import LGBMQuantile, QuantileCalibrator, XGBPoint
from windcast.providers.stack import WeatherStack
from windcast.scada.power_curve import PowerCurve, air_density, standard_pressure_hpa

MEMBERS = ["persistence", "climatology", "nwp_pc", "xgb", "lgbm_q50"]
POINT_MODELS = ["persistence", "climatology", "arima", "nwp_pc", "xgb", "lgbm_q50", "stack"]
MIN_TRAIN_ROWS = 500


@dataclass
class Bundle:
    power_curve: PowerCurve
    weather: WeatherStack
    climatology: Climatology
    lgbm: LGBMQuantile
    xgb: XGBPoint
    stacker: StackedBlender
    calibrator: QuantileCalibrator
    features: list[str]
    quantiles: list[float]
    site: dict
    meta: dict = field(default_factory=dict)


def site_density(site: dict, temp_c) -> np.ndarray:
    return air_density(standard_pressure_hpa(site.get("elevation_m", 0.0)), temp_c)


def fit_power_curve(farm: pd.DataFrame, before: pd.Timestamp, site: dict,
                    clean: dict | None = None) -> PowerCurve:
    """Farm-level curve at forecast resolution from unconstrained operation before ``before``."""
    clean = clean or {"min_availability": 0.95, "max_derated_frac": 0.05}
    f = farm[farm.index < before]
    ok = (f["power_norm"].notna() & ~f["dq"] & (f["availability"] >= clean["min_availability"])
          & (f["derated_frac"] <= clean["max_derated_frac"]))
    f = f[ok]
    return PowerCurve(cut_out=site.get("cut_out_ms")).fit(
        f["ws_obs"], f["power_norm"], site_density(site, f["temp_obs"]))


def _member_frame(df: pd.DataFrame, clim: Climatology, lgbm: LGBMQuantile, xgb: XGBPoint,
                  features: list[str], calibrator: QuantileCalibrator | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["climatology"] = clim.predict(df)
    out["persistence"] = pd.Series(persistence(df), index=df.index).fillna(out["climatology"])
    # Physical model: blended, bias-corrected NWP through the power curve.
    out["nwp_pc"] = df["blend_pc"].fillna(out["climatology"])
    out["xgb"] = xgb.predict(df[features])
    q = lgbm.predict(df[features])
    if calibrator is not None:
        q = calibrator.transform(q, df["lead_h"])
    return pd.concat([out, q], axis=1)


def fit_bundle(train: pd.DataFrame, farm: pd.DataFrame, cfg: dict, train_end: pd.Timestamp) -> Bundle:
    """Fit everything on ``train`` (rows whose valid time is before ``train_end``)."""
    site, qs = cfg["site"], cfg["forecast"]["quantiles"]
    seed = int(cfg["model"]["seed"])
    wcfg = cfg["weather"]
    pc = fit_power_curve(farm, train_end, site, cfg["clean"])
    weather = WeatherStack(list(cfg["providers"]), pc, wcfg["bias_correction"], wcfg["blend"],
                           wcfg["max_age_h"]).fit(train)
    tr = weather.transform(train)
    features = feature_columns(tr)
    # Models learn unconstrained output; curtailed and outage periods would teach them noise.
    rows = tr[tr["tgt_clean"]]
    if len(rows) < MIN_TRAIN_ROWS:
        raise ValueError(f"only {len(rows)} clean training rows; need {MIN_TRAIN_ROWS}")
    clim = Climatology(qs).fit(rows)

    # Stacking weights come from a time-ordered hold-out at the end of training, with
    # the same embargo as the outer split. Base learners are then refitted on all rows.
    cut = rows["issue_time"].quantile(1 - cfg["backtest"]["stack_holdout_frac"])
    inner, hold = rows[rows["valid_time"] < cut], rows[rows["issue_time"] >= cut]
    jobs = int(cfg["model"].get("n_jobs", 0))
    make_lgbm = lambda: LGBMQuantile(qs, cfg["model"]["lgbm"], seed, jobs)  # noqa: E731
    make_xgb = lambda: XGBPoint(cfg["model"]["xgb"], seed, jobs)  # noqa: E731
    stacker, calibrator = StackedBlender(MEMBERS), QuantileCalibrator(qs)
    if len(inner) >= MIN_TRAIN_ROWS and len(hold) >= 150:
        lgbm_i, xgb_i = make_lgbm().fit(inner[features], inner["y"]), make_xgb().fit(inner[features], inner["y"])
        held = _member_frame(hold, clim, lgbm_i, xgb_i, features)
        calibrator.fit(held, hold["y"], hold["lead_h"])
        held[calibrator.cols] = calibrator.transform(held, hold["lead_h"])
        stacker.fit(held, hold["y"], hold["lead_h"])
    lgbm, xgb = make_lgbm().fit(rows[features], rows["y"]), make_xgb().fit(rows[features], rows["y"])
    return Bundle(pc, weather, clim, lgbm, xgb, stacker, calibrator, features, list(qs), dict(site),
                  meta={"train_end": str(train_end), "n_train_rows": int(len(rows))})


def predict_bundle(bundle: Bundle, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts for every row of ``frame``. Returns (predictions, transformed frame)."""
    tf = bundle.weather.transform(frame)
    members = _member_frame(tf, bundle.climatology, bundle.lgbm, bundle.xgb, bundle.features,
                            bundle.calibrator)
    context = [c for c in tf.columns if c.startswith("tgt_")]
    preds = pd.concat([
        tf[KEYS + ["y", *context, "fallback_tier", "n_providers", "blend_ws", "blend_pc"]],
        members, bundle.climatology.predict_quantiles(tf)], axis=1)
    preds["stack"] = bundle.stacker.predict(members, tf["lead_h"])
    # Potential power from the wind that actually blew: the anchor for root-cause analysis.
    preds["p_obs"] = bundle.power_curve.predict(tf["tgt_ws_obs"],
                                                site_density(bundle.site, tf["tgt_temp_obs"]))
    return preds, tf

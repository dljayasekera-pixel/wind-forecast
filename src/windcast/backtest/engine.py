"""Rolling-origin backtest: fit on the past, forecast the next block, repeat, then score."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from windcast.backtest import metrics as M
from windcast.backtest.splits import Fold, rolling_origin_folds, split
from windcast.features.build import build_frame
from windcast.models.baselines import arima_forecasts
from windcast.models.bundle import POINT_MODELS, Bundle, fit_bundle, predict_bundle
from windcast.providers import scorecard
from windcast.rca import attribution
from windcast.utils import get_logger, lead_bucket_label

log = get_logger(__name__)
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


@dataclass
class BacktestResult:
    preds: pd.DataFrame
    eval_frame: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    folds: list[Fold]
    last_bundle: Bundle | None = field(default=None, repr=False)


def _arima(farm: pd.DataFrame, test: pd.DataFrame, fold: Fold, cfg: dict) -> pd.DataFrame | None:
    acfg = cfg["backtest"]["arima"]
    if not acfg.get("enabled", False):
        return None
    res = pd.Timedelta(cfg["forecast"]["resolution"])
    grid = farm["power_norm"].reindex(pd.date_range(farm.index.min(), farm.index.max(), freq=res))
    leads_h = sorted(h for h in test["lead_h"].unique() if h <= acfg["max_lead_h"])
    if not leads_h:
        return None
    origins = np.sort(test["issue_time"].unique())
    if len(origins) > acfg["max_origins_per_fold"]:
        origins = origins[np.linspace(0, len(origins) - 1, acfg["max_origins_per_fold"]).astype(int)]
    steps = [int(round(pd.Timedelta(hours=h) / res)) for h in leads_h]
    fc = arima_forecasts(grid, fold.test_start, pd.DatetimeIndex(origins), steps,
                         tuple(acfg["order"]), acfg["history_steps"])
    fc["valid_time"] = fc["issue_time"] + fc["lead_step"] * res
    return fc[["issue_time", "valid_time", "arima"]]


def add_strata(preds: pd.DataFrame, site: dict) -> pd.DataFrame:
    """Reporting dimensions: season, wind regime (power-curve region), lead bucket."""
    out = preds.copy()
    out["season"] = out["valid_time"].dt.month.map(SEASONS)
    cut_in = site["cut_in_ms"]
    out["regime"] = pd.cut(out["tgt_ws_obs"], [-np.inf, cut_in + 1, 11.0, np.inf],
                           labels=["below_cut_in", "partial_load", "rated"])
    out["lead_bucket"] = pd.Categorical(lead_bucket_label(out["lead_h"]),
                                        ["0-6h", "6-24h", "24-48h", "48-96h", "96h+"], ordered=True)
    return out


def score(preds: pd.DataFrame, eval_frame: pd.DataFrame, nwp: pd.DataFrame, cfg: dict) -> dict:
    """All report tables from out-of-sample predictions."""
    qs = cfg["forecast"]["quantiles"]
    providers = list(cfg["providers"])
    clean = preds[preds["tgt_clean"]]
    issue_freq_h = pd.Timedelta(cfg["forecast"]["issue_freq"]) / pd.Timedelta("1h")
    t = {
        "metrics_overall_clean": M.metric_table(clean, POINT_MODELS),
        "metrics_overall_all": M.metric_table(preds, POINT_MODELS),
        "metrics_by_lead": M.metric_table(clean, POINT_MODELS, ["lead_h"]),
        "metrics_by_lead_bucket": M.metric_table(clean, POINT_MODELS, ["lead_bucket"]),
        "metrics_by_season": M.metric_table(clean, POINT_MODELS, ["season", "lead_bucket"]),
        "metrics_by_regime": M.metric_table(clean, POINT_MODELS, ["regime", "lead_bucket"]),
        "metrics_by_fold": M.metric_table(clean, POINT_MODELS, ["fold"]),
        "prob_by_lead": M.probabilistic_table(clean, qs, by=["lead_h"]),
        "prob_overall": M.probabilistic_table(clean, qs),
        "reliability": M.reliability_table(clean, qs),
    }
    dm_rows = []
    for lead_h, g in clean.groupby("lead_h"):
        h = max(1, int(round(lead_h / issue_freq_h)))
        g = g.sort_values("issue_time")
        for other in ("persistence", "nwp_pc", "lgbm_q50"):
            stat, p = M.dm_test(g["stack"] - g["y"], g[other] - g["y"], h=h)
            dm_rows.append({"lead_h": lead_h, "model": "stack", "versus": other, "n": len(g),
                            "dm_stat": stat, "p_value": p})
    t["dm_tests"] = pd.DataFrame(dm_rows).round(4)
    t["provider_delivery"] = scorecard.delivery_scorecard(nwp, cfg["providers"]).reset_index()
    t["provider_accuracy"] = scorecard.accuracy_scorecard(eval_frame, providers)
    t["provider_impact"] = scorecard.provider_impact(eval_frame, providers)
    dec = attribution.decompose(preds)
    thr = cfg["monitoring"]["big_miss_threshold"]
    t["rca_summary"] = attribution.summary(dec, thr)
    t["rca_by_lead"] = attribution.variance_shares(dec).reset_index()
    t["rca_events"] = attribution.big_misses(dec, thr).head(200)
    return t


def run_backtest(farm: pd.DataFrame, nwp: pd.DataFrame, cfg: dict,
                 frame: pd.DataFrame | None = None) -> BacktestResult:
    bcfg = cfg["backtest"]
    frame = build_frame(farm, nwp, cfg) if frame is None else frame
    folds = rolling_origin_folds(frame["issue_time"].min(), frame["issue_time"].max(),
                                 bcfg["initial_train_days"], bcfg["test_days"], bcfg.get("step_days"))
    providers = list(cfg["providers"])
    keep = ["issue_time", "valid_time", "lead_h", "y", "tgt_ws_obs", "tgt_dq", "tgt_clean",
            "blend_ws", "blend_pc", "fallback_tier"]
    keep += [f"{p}_{c}" for p in providers for c in ("ws_hub", "ws_bc", "pc", "age_h")]
    all_preds, all_eval, bundle = [], [], None
    for fold in folds:
        train, test = split(frame, fold)
        if test.empty:
            continue
        bundle = fit_bundle(train, farm, cfg, fold.test_start)
        preds, tf = predict_bundle(bundle, test)
        arima = _arima(farm, test, fold, cfg)
        preds["arima"] = (preds.merge(arima, on=["issue_time", "valid_time"], how="left")["arima"].to_numpy()
                          if arima is not None else np.nan)
        preds["fold"] = fold.index
        all_preds.append(preds)
        all_eval.append(tf[keep])
        fold_mae = M.mae(*preds.loc[preds["tgt_clean"], ["y", "stack"]].to_numpy().T)
        log.info("fold done", extra={"ctx": {"fold": fold.index, "test_start": str(fold.test_start.date()),
                                             "train_rows": len(train), "test_rows": len(test),
                                             "stack_nmae": round(fold_mae, 4)}})
    preds = add_strata(pd.concat(all_preds, ignore_index=True), cfg["site"])
    eval_frame = pd.concat(all_eval, ignore_index=True)
    tables = score(preds, eval_frame, nwp, cfg)
    tables["blend_weights"] = bundle.weather.blender_.table().reset_index(names="lead_bucket")
    tables["stack_weights"] = bundle.stacker.table().reset_index(names="lead_bucket")
    tables["feature_importance"] = bundle.lgbm.importance().head(30).rename("gain_share") \
        .rename_axis("feature").reset_index().round(4)
    return BacktestResult(preds, eval_frame, tables, folds, bundle)

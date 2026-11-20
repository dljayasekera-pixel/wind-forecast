"""Provider scorecards: delivery (coverage, latency, horizon, variables) and accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.connectors.base import NWP_VARIABLES
from windcast.utils import LEAD_LABELS, lead_bucket_label


def delivery_scorecard(nwp: pd.DataFrame, providers: dict) -> pd.DataFrame:
    """Reliability of each feed, independent of forecast skill."""
    rows = []
    for name, grp in nwp.groupby("provider"):
        runs = grp.drop_duplicates("run_time")
        latency = (runs["available_time"] - runs["run_time"]) / pd.Timedelta("1h")
        span_h = (grp["run_time"].max() - grp["run_time"].min()) / pd.Timedelta("1h")
        cycle = providers.get(name, {}).get("cycle_h")
        expected = span_h / cycle + 1 if cycle and span_h > 0 else np.nan
        gaps = runs["run_time"].sort_values().diff() / pd.Timedelta("1h")
        valid_span = (grp["valid_time"].max() - grp["valid_time"].min()) / pd.Timedelta("1h") + 1
        row = {
            "provider": name,
            "n_runs": len(runs),
            "run_coverage": min(1.0, len(runs) / expected) if expected == expected else np.nan,
            "valid_hour_coverage": grp["valid_time"].nunique() / valid_span,
            "latency_mean_h": latency.mean(),
            "latency_p95_h": latency.quantile(0.95),
            "longest_gap_h": gaps.max(),
            "max_lead_h": ((grp["valid_time"] - grp["run_time"]) / pd.Timedelta("1h")).max(),
        }
        for var in NWP_VARIABLES:
            row[f"avail_{var}"] = grp[var].notna().mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("provider").round(3)


def accuracy_scorecard(eval_frame: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    """Out-of-sample wind and power accuracy per provider and lead bucket.

    ``eval_frame`` is the concatenation of transformed TEST folds, so the
    bias-corrected columns were produced by corrections fitted on earlier data.
    """
    df = eval_frame.copy()
    df["lead_bucket"] = lead_bucket_label(df["lead_h"])
    obs_ws = df["tgt_ws_obs"].where(~df["tgt_dq"])
    sources = [(p, f"{p}_ws_hub", f"{p}_ws_bc", f"{p}_pc") for p in providers]
    sources.append(("blend", None, "blend_ws", "blend_pc"))
    rows = []
    for name, raw_col, bc_col, pc_col in sources:
        for bucket in LEAD_LABELS:
            m = (df["lead_bucket"] == bucket).to_numpy()
            ok = m & df[bc_col].notna().to_numpy() & obs_ws.notna().to_numpy()
            if ok.sum() < 20:
                continue
            raw = df.loc[ok, raw_col] if raw_col else df.loc[ok, bc_col]
            err_raw = raw - obs_ws[ok]
            err_bc = df.loc[ok, bc_col] - obs_ws[ok]
            clean = ok & df["tgt_clean"].to_numpy()
            err_p = df.loc[clean, pc_col] - df.loc[clean, "y"]
            rows.append({
                "provider": name, "lead_bucket": bucket, "n": int(ok.sum()),
                "ws_bias_raw": err_raw.mean(), "ws_mae_raw": err_raw.abs().mean(),
                "ws_rmse_raw": np.sqrt((err_raw**2).mean()),
                "ws_bias_bc": err_bc.mean(), "ws_mae_bc": err_bc.abs().mean(),
                "ws_rmse_bc": np.sqrt((err_bc**2).mean()),
                "ws_corr": np.corrcoef(df.loc[ok, bc_col], obs_ws[ok])[0, 1],
                "power_nmae": err_p.abs().mean(), "power_nrmse": np.sqrt((err_p**2).mean()),
            })
    out = pd.DataFrame(rows)
    out["lead_bucket"] = pd.Categorical(out["lead_bucket"], LEAD_LABELS, ordered=True)
    return out.sort_values(["lead_bucket", "provider"]).reset_index(drop=True).round(4)


def provider_impact(eval_frame: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    """How much each provider's weather error costs in power terms, relative to the blend."""
    acc = accuracy_scorecard(eval_frame, providers)
    blend = acc[acc["provider"] == "blend"].set_index("lead_bucket")["power_nmae"]
    acc = acc[acc["provider"] != "blend"].copy()
    acc["nmae_vs_blend_pct"] = 100 * (acc["power_nmae"] / acc["lead_bucket"].map(blend).astype(float) - 1)
    return acc[["provider", "lead_bucket", "power_nmae", "nmae_vs_blend_pct"]].round(3)

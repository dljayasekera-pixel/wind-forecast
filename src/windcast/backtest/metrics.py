"""Deterministic and probabilistic forecast metrics.

The target is power / installed capacity, so MAE and RMSE are already the
normalised errors (nMAE, nRMSE) used in the industry; tables report them as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(f) - np.asarray(y))))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(f) - np.asarray(y)) ** 2)))


def bias(y, f) -> float:
    """Mean error, forecast minus actual. Positive = over-forecast."""
    return float(np.mean(np.asarray(f) - np.asarray(y)))


def skill(score: float, reference: float) -> float:
    """1 - score/reference. Positive means better than the reference forecast."""
    return float(1.0 - score / reference) if reference > 0 else float("nan")


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def crps_from_quantiles(y, q_preds: np.ndarray, quantiles: list[float]) -> float:
    """CRPS approximated as twice the pinball loss integrated over quantile level.

    CRPS = 2 * integral_0^1 pinball_tau dtau. The integral is taken by the trapezoid rule
    over the available quantile levels, with the loss held flat out to tau = 0 and 1.
    Coarse quantile grids make this an approximation; compare like with like.
    """
    y, Q, taus = np.asarray(y, float), np.asarray(q_preds, float), np.asarray(quantiles, float)
    d = y[:, None] - Q
    loss = np.maximum(taus * d, (taus - 1) * d)
    grid = np.concatenate([[0.0], taus, [1.0]])
    loss = np.column_stack([loss[:, 0], loss, loss[:, -1]])
    return float(2.0 * np.mean(np.trapezoid(loss, grid, axis=1)))


def dm_test(e1, e2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy, Harvey-Leybourne-Newbold corrected.

    ``e1``/``e2`` are errors of two forecasts on the same targets, ``h`` the forecast
    horizon in sampling steps (sets the autocovariance truncation). A negative
    statistic means forecast 1 has the smaller loss. Returns (statistic, two-sided p).
    """
    d = np.abs(np.asarray(e1, float)) ** power - np.abs(np.asarray(e2, float)) ** power
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    h = int(min(max(1, h), n - 1))
    centred = d - d.mean()
    gamma = [float(np.sum(centred[k:] * centred[: n - k]) / n) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:  # truncated estimator can go negative; fall back to the lag-0 variance
        var = gamma[0] / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var)
    stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def metric_table(df: pd.DataFrame, models: list[str], by: list[str] | None = None,
                 reference: str = "persistence") -> pd.DataFrame:
    """MAE, RMSE, bias and skill per model, each scored against the reference on identical rows."""
    by = by or []
    groups = df.groupby(by, observed=True) if by else [((), df)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for model in models:
            if model not in g:
                continue
            ok = g[model].notna() & g["y"].notna() & g[reference].notna()
            if ok.sum() < 5:
                continue
            y, f, ref = g.loc[ok, "y"], g.loc[ok, model], g.loc[ok, reference]
            rows.append({**dict(zip(by, key, strict=True)), "model": model, "n": int(ok.sum()),
                         "mae": mae(y, f), "rmse": rmse(y, f), "bias": bias(y, f),
                         "skill_mae": skill(mae(y, f), mae(y, ref)),
                         "skill_rmse": skill(rmse(y, f), rmse(y, ref))})
    return pd.DataFrame(rows).round(4)


def probabilistic_table(df: pd.DataFrame, quantiles: list[float], prefix: str = "lgbm",
                        ref_prefix: str = "clim", by: list[str] | None = None) -> pd.DataFrame:
    """CRPS, CRPS skill vs climatology, and coverage / width of the central interval."""
    by = by or []
    cols = [f"{prefix}_q{int(round(q * 100)):02d}" for q in quantiles]
    ref_cols = [f"{ref_prefix}_q{int(round(q * 100)):02d}" for q in quantiles]
    lo, hi = cols[0], cols[-1]
    nominal = quantiles[-1] - quantiles[0]
    groups = df.groupby(by, observed=True) if by else [((), df)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        g = g.dropna(subset=["y", *cols, *ref_cols])
        if len(g) < 5:
            continue
        crps = crps_from_quantiles(g["y"], g[cols].to_numpy(), quantiles)
        crps_ref = crps_from_quantiles(g["y"], g[ref_cols].to_numpy(), quantiles)
        rows.append({**dict(zip(by, key, strict=True)), "n": len(g), "crps": crps,
                     "crps_climatology": crps_ref, "crpss": skill(crps, crps_ref),
                     "nominal_coverage": nominal,
                     "coverage": float(((g["y"] >= g[lo]) & (g["y"] <= g[hi])).mean()),
                     "mean_width": float((g[hi] - g[lo]).mean())})
    return pd.DataFrame(rows).round(4)


def reliability_table(df: pd.DataFrame, quantiles: list[float], prefix: str = "lgbm",
                      tol: float = 0.002) -> pd.DataFrame:
    """Observed frequency of y below each quantile forecast (calibration curve).

    Wind power has point masses at 0 and at rated power, where forecast and outcome
    coincide. Ties within ``tol`` count half, as in a randomised PIT; without that,
    every calm hour would be scored as 'below' all quantiles and distort the curve.
    """
    rows = []
    for q in quantiles:
        col = f"{prefix}_q{int(round(q * 100)):02d}"
        g = df.dropna(subset=["y", col])
        rows.append({"nominal": q, "observed": float(((g["y"] < g[col] - tol)
                                       + 0.5 * ((g["y"] - g[col]).abs() <= tol)).mean()), "n": len(g),
                     "pinball": pinball(g["y"], g[col], q)})
    return pd.DataFrame(rows).round(4)

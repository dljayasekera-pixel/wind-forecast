"""Turbine-level SCADA quality control.

Flags per turbine per 10 minutes:
    flag_missing    no power or wind-speed sample
    flag_range      physically impossible value
    flag_stuck      anemometer frozen (identical readings in a row)
    flag_outage     logged stop, or zero power in wind well above cut-in
    flag_underperf  power far below the reference curve (curtailment, derate, icing)
    flag_derated    logged partial performance, or flag_underperf
    flag_icing      flag_underperf in near-freezing air
    qc_ok           the sample is trustworthy as a measurement
    operating_ok    qc_ok and the turbine was running unconstrained
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.scada.power_curve import PowerCurve, air_density, standard_pressure_hpa

STUCK_RUN = 5  # identical consecutive wind-speed samples that mark a frozen sensor


OUTAGE_WORDS = ("forced", "outage", "maintenance", "standby", "shutdown", "electrical", "repair")


def map_iec_category(category: str) -> str:
    """Collapse an IEC 61400-26 style category to ok / derated / outage.

    Only categories that mean the turbine could not or was not allowed to produce
    are outages. 'Out of environmental specification' (wind below cut-in or above
    cut-out) is weather, which the forecast must predict, so it counts as ok, as do
    informational events with no category.
    """
    text = str(category).lower()
    if "partial" in text or "derat" in text or "curtail" in text:
        return "derated"
    if "environmental" in text or "full performance" in text:
        return "ok"
    if any(word in text for word in OUTAGE_WORDS):
        return "outage"
    return "ok"


def complete_grid(scada: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """Reindex every turbine onto one regular grid so gaps become explicit rows."""
    times = pd.date_range(scada["time"].min(), scada["time"].max(), freq=freq)
    turbines = np.sort(scada["turbine_id"].unique())
    grid = pd.MultiIndex.from_product([turbines, times], names=["turbine_id", "time"])
    return scada.set_index(["turbine_id", "time"]).reindex(grid).reset_index()


def apply_status(scada: pd.DataFrame, status: pd.DataFrame) -> pd.Series:
    """Operating state per SCADA row from the event log. Outage beats derated beats ok."""
    state = np.zeros(len(scada), dtype=np.int8)  # 0 ok, 1 derated, 2 outage
    if len(status):
        level = status["iec_category"].map(map_iec_category).map({"ok": 0, "derated": 1, "outage": 2})
        for tid, rows in scada.groupby("turbine_id").indices.items():
            times = scada["time"].to_numpy()[rows]
            ev = status[(status["turbine_id"] == tid) & (level > 0)]
            lo = np.searchsorted(times, ev["start"].to_numpy(), side="left")
            hi = np.searchsorted(times, ev["end"].to_numpy(), side="left")
            for a, b, lvl in zip(lo, hi, level.loc[ev.index].to_numpy(), strict=True):
                if b > a:
                    state[rows[a:b]] = np.maximum(state[rows[a:b]], lvl)
    return pd.Series(np.array(["ok", "derated", "outage"])[state], index=scada.index)


def _stuck(ws: pd.Series, turbine: pd.Series) -> pd.Series:
    same = ws.groupby(turbine).diff().eq(0)
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum")
    return (run_len >= STUCK_RUN) & (ws > 0.5)


def clean_scada(scada: pd.DataFrame, status: pd.DataFrame, site: dict) -> tuple[pd.DataFrame, PowerCurve]:
    """Flag every sample and return the robust turbine-level reference curve used to do it."""
    rated, cut_in = float(site["rated_kw"]), float(site["cut_in_ms"])
    df = complete_grid(scada)
    df["status"] = apply_status(df, status)
    ws, power = df["wind_speed"], df["power_kw"]

    df["flag_missing"] = ws.isna() | power.isna()
    df["flag_range"] = (power < -0.05 * rated) | (power > 1.1 * rated) | (ws < 0) | (ws > 45)
    df["flag_stuck"] = _stuck(ws, df["turbine_id"])
    df["qc_ok"] = ~(df["flag_missing"] | df["flag_range"] | df["flag_stuck"])
    df["flag_outage"] = (df["status"] == "outage") | (
        df["qc_ok"] & (power <= 0.01 * rated) & (ws >= cut_in + 2.0))

    # Density-normalised wind speed: SCADA has temperature but no pressure.
    temp = df["ambient_temp"].fillna(df["ambient_temp"].median()).fillna(10.0)
    rho = air_density(standard_pressure_hpa(site.get("elevation_m", 0.0)), temp)
    p_norm = power / rated
    base = df["qc_ok"] & ~df["flag_outage"]
    curve = PowerCurve(stat="median", cut_out=site.get("cut_out_ms")).fit(
        ws[base], p_norm[base], rho[base.to_numpy()])

    expected = curve.predict(ws, rho)
    tolerance = np.maximum(0.12, 3.0 * curve.sigma_at(ws))
    below = (base & ((p_norm - expected) < -tolerance) & (ws >= cut_in + 1.0)).fillna(False)
    # Real curtailment and icing last; an isolated low sample (gust, yaw, wake) does not.
    grp = below.groupby(df["turbine_id"])
    prev = grp.shift(1, fill_value=False).astype(bool)
    nxt = grp.shift(-1, fill_value=False).astype(bool)
    df["flag_underperf"] = below & (prev | nxt)
    df["flag_derated"] = (df["status"] == "derated") | df["flag_underperf"]
    df["flag_icing"] = df["flag_underperf"] & (df["ambient_temp"] < 2.0)
    df["operating_ok"] = df["qc_ok"] & ~df["flag_outage"] & ~df["flag_derated"]
    return df, curve


def qc_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Share of samples per flag and turbine, for the data-quality report."""
    flags = [c for c in flagged.columns if c.startswith("flag_")] + ["qc_ok", "operating_ok"]
    return flagged.groupby("turbine_id")[flags].mean().round(4)

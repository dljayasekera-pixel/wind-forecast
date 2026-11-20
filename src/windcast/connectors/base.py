"""Canonical schemas shared by every connector.

A new feed (plant historian, commercial NWP vendor, ISO API) only has to emit
these frames; nothing downstream changes.

SCADA   one row per turbine per 10 minutes, ``time`` = period START, UTC.
STATUS  one row per logged event with an IEC 61400-26 style category.
NWP     one row per provider / run / valid time. ``available_time`` is when the
        value could first have been used operationally. Features are always
        selected with ``available_time <= issue_time``, which is what keeps
        backtests free of look-ahead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from windcast.utils import to_ns

SCADA_COLUMNS = ["time", "turbine_id", "power_kw", "wind_speed", "wind_dir", "ambient_temp"]
STATUS_COLUMNS = ["turbine_id", "start", "end", "iec_category"]
NWP_COLUMNS = ["provider", "run_time", "valid_time", "available_time",
               "ws_10m", "ws_100m", "wd_100m", "temp_2m", "pressure_hpa"]
NWP_VARIABLES = ["ws_10m", "ws_100m", "wd_100m", "temp_2m", "pressure_hpa"]


class SchemaError(ValueError):
    pass


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} frame is missing columns {missing}")


def validate_scada(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, SCADA_COLUMNS, "SCADA")
    out = df[SCADA_COLUMNS].copy()
    out["time"] = to_ns(out["time"])
    out["turbine_id"] = out["turbine_id"].astype(int)
    if out.duplicated(["time", "turbine_id"]).any():
        out = out.drop_duplicates(["time", "turbine_id"], keep="first")
    return out.sort_values(["turbine_id", "time"]).reset_index(drop=True)


def validate_status(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, STATUS_COLUMNS, "status")
    out = df[STATUS_COLUMNS].copy()
    out["start"] = to_ns(out["start"])
    out["end"] = to_ns(out["end"])
    out["turbine_id"] = out["turbine_id"].astype(int)
    return out.sort_values(["turbine_id", "start"]).reset_index(drop=True)


def validate_nwp(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, NWP_COLUMNS, "NWP")
    out = df[NWP_COLUMNS].copy()
    for col in ("run_time", "valid_time", "available_time"):
        out[col] = to_ns(out[col])
    if (out["available_time"] < out["run_time"]).any():
        raise SchemaError("available_time earlier than run_time")
    out = out.drop_duplicates(["provider", "run_time", "valid_time"], keep="last")
    return out.sort_values(["provider", "run_time", "valid_time"]).reset_index(drop=True)


class NWPConnector(ABC):
    """Interface for a weather-forecast source."""

    provider: str

    @abstractmethod
    def fetch(self, start: str, end: str) -> pd.DataFrame:
        """Return a frame with NWP_COLUMNS covering valid times in [start, end)."""

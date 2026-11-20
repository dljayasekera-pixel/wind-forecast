"""SQL layer: create the schema, load frames, run the named analytical queries."""
from __future__ import annotations

import math
import re
import sqlite3
from importlib import resources
from pathlib import Path

import pandas as pd

from windcast.config import capacity_kw

_NAME = re.compile(r"^-- name:\s*(\w+)\s*$", re.MULTILINE)


def _sql(name: str) -> str:
    return resources.files("windcast.sql").joinpath(name).read_text(encoding="utf-8")


def named_queries() -> dict[str, str]:
    """Parse queries.sql into {name: sql}."""
    text = _sql("queries.sql")
    parts = _NAME.split(text)[1:]
    return {parts[i]: parts[i + 1].strip() for i in range(0, len(parts), 2)}


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    # Some SQLite builds ship without the math extension.
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None)
    conn.executescript(_sql("schema.sql"))
    return conn


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d %H:%M:%S")


def _replace(conn: sqlite3.Connection, table: str, df: pd.DataFrame, where: str, params: tuple) -> int:
    """Idempotent load: delete the slice being reloaded, then insert."""
    conn.execute(f"DELETE FROM {table} WHERE {where}", params)  # noqa: S608 - table/where are internal constants
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=50_000)
    conn.commit()
    return len(df)


def load_site(conn, cfg: dict) -> None:
    s = cfg["site"]
    conn.execute("INSERT OR REPLACE INTO sites VALUES (?,?,?,?,?,?)",
                 (s["name"], s["latitude"], s["longitude"], s["hub_height_m"], s["n_turbines"], capacity_kw(cfg)))
    conn.commit()


def load_scada(conn, flagged: pd.DataFrame, site: str) -> int:
    df = pd.DataFrame({
        "site_id": site, "turbine_id": flagged["turbine_id"], "ts": _ts(flagged["time"]),
        "power_kw": flagged["power_kw"], "wind_speed": flagged["wind_speed"],
        "wind_dir": flagged["wind_dir"], "ambient_temp": flagged["ambient_temp"],
        "status": flagged["status"], "qc_ok": flagged["qc_ok"].astype(int),
        "operating_ok": flagged["operating_ok"].astype(int)})
    return _replace(conn, "scada_10min", df, "site_id = ?", (site,))


def load_farm(conn, farm: pd.DataFrame, site: str) -> int:
    df = farm.reset_index()
    df = pd.DataFrame({
        "site_id": site, "ts": _ts(df["time"]), "power_kw": df["power_kw"], "power_norm": df["power_norm"],
        "ws_obs": df["ws_obs"], "temp_obs": df["temp_obs"], "availability": df["availability"],
        "derated_frac": df["derated_frac"], "dq": df["dq"].astype(int)})
    return _replace(conn, "farm_hourly", df, "site_id = ?", (site,))


def load_nwp(conn, nwp: pd.DataFrame, site: str) -> int:
    df = nwp.copy()
    df["lead_h"] = (df["valid_time"] - df["run_time"]) / pd.Timedelta("1h")
    for col in ("run_time", "valid_time", "available_time"):
        df[col] = _ts(df[col])
    df.insert(0, "site_id", site)
    cols = ["site_id", "provider", "run_time", "valid_time", "available_time", "lead_h",
            "ws_10m", "ws_100m", "wd_100m", "temp_2m", "pressure_hpa"]
    return _replace(conn, "nwp_forecasts", df[cols], "site_id = ?", (site,))


def load_forecasts(conn, preds: pd.DataFrame, site: str, quantile_cols: list[str],
                   point_models: tuple[str, ...] = ("persistence", "nwp_pc", "stack"),
                   model_version: str = "backtest") -> int:
    """Wide prediction frame to the long forecasts table."""
    keys = preds[["issue_time", "valid_time", "lead_h"]].copy()
    keys["issue_time"], keys["valid_time"] = _ts(keys["issue_time"]), _ts(keys["valid_time"])
    parts = []
    for model in point_models:
        if model in preds:
            parts.append(keys.assign(model=model, kind="point", value=preds[model].to_numpy()))
    for col in quantile_cols:
        model, kind = col.rsplit("_", 1)
        parts.append(keys.assign(model=model, kind=kind, value=preds[col].to_numpy()))
    df = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    df["site_id"], df["model_version"] = site, model_version
    df = df.drop_duplicates(["site_id", "model", "kind", "issue_time", "valid_time"])
    return _replace(conn, "forecasts", df, "site_id = ? AND model_version = ?", (site, model_version))


def query(conn, name: str, **params) -> pd.DataFrame:
    """Run a named query from queries.sql."""
    queries = named_queries()
    if name not in queries:
        raise KeyError(f"unknown query '{name}'; available: {sorted(queries)}")
    return pd.read_sql_query(queries[name], conn, params=params)

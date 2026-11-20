"""Kelmarsh wind farm open SCADA data (Cubico, CC-BY-4.0) from Zenodo.

Six Senvion MM92 turbines, 10-minute SCADA and event logs, 2016 to end 2024.
Record: https://zenodo.org/records/16807551

File layout inside each yearly zip:
    Turbine_Data_Kelmarsh_<n>_<from>_-_<to>_<id>.csv   10-minute signals
    Status_Kelmarsh_<n>_<from>_-_<to>_<id>.csv         event log
Both start with a block of '#' comment lines before the header row.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from windcast.connectors.base import validate_scada, validate_status
from windcast.utils import get_logger

ZENODO_RECORD = "16807551"
ZENODO_API = "https://zenodo.org/api/records/{record}"
log = get_logger(__name__)

_DATA_HEADER = "# Date and time"
_STATUS_HEADER = "Timestamp start"
_TURBINE_RE = re.compile(r"Kelmarsh_(\d+)_")


def list_files(record: str = ZENODO_RECORD, session=None) -> list[dict]:
    """File entries (key, checksum, download link) of the Zenodo record."""
    http = session or requests
    resp = http.get(ZENODO_API.format(record=record), timeout=60)
    resp.raise_for_status()
    return resp.json()["files"]


def download_year(year: int, raw_dir: str | Path, record: str = ZENODO_RECORD, session=None) -> Path:
    """Download one year's SCADA zip, verifying the md5 published by Zenodo. Idempotent."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches = [f for f in list_files(record, session) if f["key"].startswith(f"Kelmarsh_SCADA_{year}")]
    if not matches:
        raise FileNotFoundError(f"no Kelmarsh SCADA archive for {year} in Zenodo record {record}")
    entry = matches[0]
    target = raw_dir / entry["key"]
    expected = entry.get("checksum", "").removeprefix("md5:")
    if target.exists() and (not expected or _md5(target) == expected):
        return target
    http = session or requests
    url = entry["links"]["self"]
    log.info("downloading", extra={"ctx": {"url": url, "target": str(target)}})
    with http.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    if expected and _md5(target) != expected:
        target.unlink()
        raise OSError(f"checksum mismatch for {entry['key']}")
    return target


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_row(open_stream, header_prefix: str) -> int:
    """Index of the header line, read from the first lines only (the files are large)."""
    with io.TextIOWrapper(open_stream(), encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith(header_prefix):
                return i
            if i >= 100:
                break
    raise ValueError(f"header starting with {header_prefix!r} not found")


def _read_columns(open_stream, header_prefix: str, pick) -> pd.DataFrame:
    """Stream a CSV and load only the columns chosen by ``pick(all_columns)``.

    Kelmarsh turbine files have about 300 columns and 50,000 rows per year. Reading
    them whole (as text, then all columns) runs out of memory on a typical laptop;
    reading just the handful of columns needed keeps each file to a few MB.
    """
    skip = _header_row(open_stream, header_prefix)
    with open_stream() as fh:
        columns = pd.read_csv(fh, skiprows=skip, nrows=0, encoding="utf-8",
                              encoding_errors="replace").columns.tolist()
    wanted = [c for c in pick(columns) if c is not None]
    with open_stream() as fh:
        return pd.read_csv(fh, skiprows=skip, usecols=wanted, encoding="utf-8",
                           encoding_errors="replace", dtype=str)


def _find(columns, *needles: str) -> str | None:
    """First column equal to a needle, else first column containing it (case-insensitive)."""
    for needle in needles:
        for col in columns:
            if col.strip().lower() == needle.lower():
                return col
    for needle in needles:
        for col in columns:
            if needle.lower() in col.lower():
                return col
    return None


def _num(raw: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(np.nan, index=raw.index)
    return pd.to_numeric(raw[col], errors="coerce")


def read_turbine_data(open_stream, turbine_id: int) -> pd.DataFrame:
    """One turbine's 10-minute CSV to the canonical SCADA frame."""
    names = {}

    def pick(cols):
        names["power"] = _find(cols, "Power (kW)")
        names["ws"] = _find(cols, "Wind speed (m/s)")
        names["wd"] = _find(cols, "Wind direction (°)", "Wind direction")
        names["temp"] = _find(cols, "Nacelle ambient temperature (°C)", "Ambient temperature")
        if names["power"] is None or names["ws"] is None:
            raise ValueError("turbine file has no 'Power (kW)' or 'Wind speed (m/s)' column")
        return [_DATA_HEADER, *names.values()]

    raw = _read_columns(open_stream, _DATA_HEADER, pick)
    out = pd.DataFrame({
        "time": pd.to_datetime(raw[_DATA_HEADER], errors="coerce"),
        "turbine_id": turbine_id,
        "power_kw": _num(raw, names["power"]),
        "wind_speed": _num(raw, names["ws"]),
        "wind_dir": _num(raw, names["wd"]),
        "ambient_temp": _num(raw, names["temp"]),
    })
    return out.dropna(subset=["time"])


def read_status(open_stream, turbine_id: int) -> pd.DataFrame:
    """One turbine's event log to the canonical status frame."""
    names = {}

    def pick(cols):
        names["end"] = _find(cols, "Timestamp end")
        names["cat"] = _find(cols, "IEC category") or _find(cols, "Status")
        return [_STATUS_HEADER, names["end"], names["cat"]]

    raw = _read_columns(open_stream, _STATUS_HEADER, pick)
    start = pd.to_datetime(raw[_STATUS_HEADER], errors="coerce")
    end = pd.to_datetime(raw[names["end"]], errors="coerce") if names["end"] else pd.Series(pd.NaT, index=raw.index)
    category = raw[names["cat"]] if names["cat"] else pd.Series("Unknown", index=raw.index)
    out = pd.DataFrame({"turbine_id": turbine_id, "start": start,
                        "end": end.fillna(start + pd.Timedelta("10min")),
                        "iec_category": category.astype("string").fillna("Unknown")})
    return out.dropna(subset=["start"])


def _from_text(text: str):
    blob = text.encode("utf-8")
    return lambda: io.BytesIO(blob)


def parse_turbine_data(text: str, turbine_id: int) -> pd.DataFrame:
    """Same as read_turbine_data, from an in-memory string (small files, tests)."""
    return read_turbine_data(_from_text(text), turbine_id)


def parse_status(text: str, turbine_id: int) -> pd.DataFrame:
    return read_status(_from_text(text), turbine_id)


def load_archives(paths: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one or more yearly zips into (scada, status), one file at a time."""
    scada, status = [], []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                base = Path(name).name
                match = _TURBINE_RE.search(base)
                if not match or not base.endswith(".csv"):
                    continue
                tid = int(match.group(1))
                open_stream = lambda n=name: zf.open(n)  # noqa: E731 - reopened per pass
                if base.startswith("Turbine_Data_"):
                    log.info("parsing", extra={"ctx": {"file": base}})
                    scada.append(read_turbine_data(open_stream, tid))
                elif base.startswith("Status_"):
                    status.append(read_status(open_stream, tid))
    if not scada:
        raise FileNotFoundError("no Turbine_Data_Kelmarsh_*.csv found in the given archives")
    status_df = pd.concat(status, ignore_index=True) if status else pd.DataFrame(
        columns=["turbine_id", "start", "end", "iec_category"])
    return validate_scada(pd.concat(scada, ignore_index=True)), validate_status(status_df)


def fetch(cfg: dict, raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and parse every year touched by the configured date range."""
    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    years = range(start.year, (end - pd.Timedelta("1s")).year + 1)
    paths = [download_year(y, raw_dir) for y in years]
    scada, status = load_archives(paths)
    scada = scada[(scada["time"] >= start) & (scada["time"] < end)].reset_index(drop=True)
    status = status[(status["end"] >= start) & (status["start"] < end)].reset_index(drop=True)
    return scada, status

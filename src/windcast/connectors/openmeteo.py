"""Open-Meteo Previous Runs API: archived forecasts from several NWP models.

https://open-meteo.com/en/docs/previous-runs-api  (free for non-commercial use)

For each valid hour the API returns the value that was predicted 24, 48, ...
168 hours earlier (``<variable>_previous_day<N>``). That is mapped onto the
canonical schema as a pseudo-run with ``run_time = available_time =
valid_time - 24h * N``. Day 0 (the latest run) is deliberately not requested:
its issue time is ambiguous, so using it as a forecast would leak.

Consequence: with this source the freshest NWP usable for a forecast issued at
t0 was produced at least 24 hours before the valid time. Horizons under 24 h
therefore lean on SCADA lags. A run-resolved source (NOAA GFS/HRRR GRIB on
S3, ECMWF open data, a commercial vendor) removes that limit and plugs in
through the same NWPConnector interface.

The connector is not wind-specific: pass ``variables`` (Open-Meteo name to
column name) to fetch any hourly variable set, for example temperature and dew
point for load forecasting. With the default wind variables the output is the
canonical NWP frame; with custom variables it has the same time columns
(provider, run_time, valid_time, available_time) plus the requested columns.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from windcast.connectors.base import NWPConnector, validate_nwp
from windcast.utils import get_logger

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = {
    "wind_speed_10m": "ws_10m",
    "wind_speed_100m": "ws_100m",
    "wind_direction_100m": "wd_100m",
    "temperature_2m": "temp_2m",
    "surface_pressure": "pressure_hpa",
}
DAYS = range(1, 8)
log = get_logger(__name__)


TIME_COLUMNS = ["provider", "run_time", "valid_time", "available_time"]


def validate_forecast(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Time checks shared by every forecast frame, whatever its variables."""
    out = df[TIME_COLUMNS + columns].copy()
    for col in ("run_time", "valid_time", "available_time"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    if (out["available_time"] < out["run_time"]).any():
        raise ValueError("available_time earlier than run_time")
    out = out.drop_duplicates(["provider", "run_time", "valid_time"], keep="last")
    return out.sort_values(["provider", "run_time", "valid_time"]).reset_index(drop=True)


def parse_response(payload: dict, provider: str, model: str | None = None,
                   variables: dict[str, str] | None = None) -> pd.DataFrame:
    """Open-Meteo JSON to a forecast frame (the canonical NWP frame for the default variables)."""
    variables = variables or VARIABLES
    hourly = payload["hourly"]
    valid = pd.to_datetime(hourly["time"])
    frames = []
    for day in DAYS:
        data = {}
        for om_name, canon in variables.items():
            key = f"{om_name}_previous_day{day}"
            # With several models in one request the API suffixes keys with the model id.
            values = hourly.get(key, hourly.get(f"{key}_{model}" if model else key))
            data[canon] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce") \
                if values is not None else float("nan")
        frame = pd.DataFrame(data, index=range(len(valid)))
        frame["valid_time"] = valid
        frame["run_time"] = valid - pd.Timedelta(hours=24 * day)
        frame["available_time"] = frame["run_time"]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["provider"] = provider
    if variables is VARIABLES:
        out = out.dropna(subset=["ws_100m", "ws_10m"], how="all")
        return validate_nwp(out)
    out = out.dropna(subset=list(variables.values()), how="all")
    return validate_forecast(out, list(variables.values()))


class OpenMeteoPreviousRuns(NWPConnector):
    """One Open-Meteo model exposed as one provider.

    ``cache_tag`` distinguishes cache files when one provider is fetched for
    several locations (for example one point per load zone).
    """

    def __init__(self, provider: str, model: str, latitude: float, longitude: float,
                 cache_dir: str | Path | None = None, session=None,
                 max_retries: int = 4, pause_s: float = 1.0,
                 variables: dict[str, str] | None = None, cache_tag: str | None = None):
        self.provider, self.model = provider, model
        self.variables = variables or VARIABLES
        self.cache_tag = cache_tag
        self.latitude, self.longitude = latitude, longitude
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()
        self.max_retries, self.pause_s = max_retries, pause_s

    def _params(self, start: pd.Timestamp, end: pd.Timestamp) -> dict:
        hourly = [f"{v}_previous_day{d}" for v in self.variables for d in DAYS]
        return {"latitude": self.latitude, "longitude": self.longitude,
                "hourly": ",".join(hourly), "models": self.model,
                "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
                "wind_speed_unit": "ms", "timezone": "GMT"}

    def _get(self, start: pd.Timestamp, end: pd.Timestamp) -> dict:
        cache = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{self.cache_tag}_" if self.cache_tag else ""
            cache = self.cache_dir / f"{self.provider}_{tag}{start:%Y%m%d}_{end:%Y%m%d}.json"
            if cache.exists():
                return json.loads(cache.read_text())
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(BASE_URL, params=self._params(start, end), timeout=60)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                if cache:
                    cache.write_text(json.dumps(payload))
                return payload
            except (requests.RequestException, ValueError) as err:
                last_error = err
                wait = self.pause_s * 2**attempt
                log.warning("open-meteo request failed, retrying",
                            extra={"ctx": {"provider": self.provider, "attempt": attempt + 1,
                                           "wait_s": wait, "error": str(err)}})
                time.sleep(wait)
        raise RuntimeError(f"Open-Meteo fetch failed for {self.provider}") from last_error

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        """Valid times in [start, end), requested one calendar month at a time."""
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for month_start in pd.date_range(t0.normalize().replace(day=1), t1, freq="MS"):
            lo = max(month_start, t0)
            hi = min(month_start + pd.offsets.MonthEnd(1), t1 - pd.Timedelta("1s")).normalize()
            if lo > hi:
                continue
            frames.append(parse_response(self._get(lo, hi), self.provider, self.model, self.variables))
        out = pd.concat(frames, ignore_index=True)
        out = out[(out["valid_time"] >= t0) & (out["valid_time"] < t1)]
        if self.variables is VARIABLES:
            return validate_nwp(out)
        return validate_forecast(out, list(self.variables.values()))


def fetch_all(cfg: dict, cache_dir: str | Path | None = None, session=None) -> pd.DataFrame:
    """Fetch every configured provider. A failing provider is logged and skipped, not fatal."""
    frames = []
    for name, pcfg in cfg["providers"].items():
        conn = OpenMeteoPreviousRuns(name, pcfg["openmeteo_model"], cfg["site"]["latitude"],
                                     cfg["site"]["longitude"], cache_dir, session)
        try:
            frames.append(conn.fetch(cfg["data"]["start"], cfg["data"]["end"]))
        except Exception:  # noqa: BLE001 - one provider down must not stop ingestion
            log.exception("provider fetch failed", extra={"ctx": {"provider": name}})
    if not frames:
        raise RuntimeError("no NWP provider could be fetched")
    return pd.concat(frames, ignore_index=True)

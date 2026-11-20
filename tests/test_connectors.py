from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windcast.config import load_config
from windcast.connectors import kelmarsh, openmeteo, synthetic
from windcast.connectors.base import NWP_COLUMNS, SCADA_COLUMNS, SchemaError, validate_nwp

COMMENTS = "\n".join(f"# header line {i}" for i in range(9))
DATA_CSV = COMMENTS + """
# Date and time,Wind speed (m/s),Wind direction (°),Nacelle ambient temperature (°C),Power (kW),Data Availability
2024-01-01 00:00:00,7.10,220.5,5.0,650.2,1
2024-01-01 00:10:00,7.40,221.0,5.1,702.9,1
2024-01-01 00:20:00,NaN,NaN,5.1,NaN,0
"""
STATUS_CSV = COMMENTS + """
Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category
2024-01-01 00:20:00,2024-01-01 03:00:00,02:40:00,Stop,123,Gearbox fault,,Unscheduled,Forced Outage
2024-01-02 00:00:00,,,Warning,7,Curtailed,,Grid,Partial Performance
"""


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Turbine_Data_Kelmarsh_3_2024-01-01_-_2025-01-01_230.csv", DATA_CSV)
        zf.writestr("Status_Kelmarsh_3_2024-01-01_-_2025-01-01_230.csv", STATUS_CSV)
        zf.writestr("readme.txt", "ignored")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload=None, content=b"", status=200):
        self._payload, self.content, self.status_code = payload, content, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_synthetic_matches_canonical_schema(data):
    assert list(data["scada"].columns) == SCADA_COLUMNS
    assert list(data["nwp"].columns) == NWP_COLUMNS
    assert set(data["nwp"]["provider"]) == {"gfs", "ecmwf", "icon"}
    assert (data["nwp"]["available_time"] > data["nwp"]["run_time"]).all()
    assert not data["scada"].duplicated(["time", "turbine_id"]).any()


def test_synthetic_is_reproducible(cfg, data):
    again = synthetic.generate(cfg)
    pd.testing.assert_frame_equal(again["scada"], data["scada"])
    pd.testing.assert_frame_equal(again["nwp"], data["nwp"])


def test_validate_nwp_rejects_time_travel():
    bad = pd.DataFrame({"provider": ["x"], "run_time": ["2024-01-02"], "valid_time": ["2024-01-03"],
                        "available_time": ["2024-01-01"], "ws_10m": [1.0], "ws_100m": [2.0],
                        "wd_100m": [0.0], "temp_2m": [5.0], "pressure_hpa": [1000.0]})
    with pytest.raises(SchemaError):
        validate_nwp(bad)


def test_kelmarsh_parsers_and_archive(tmp_path):
    path = tmp_path / "Kelmarsh_SCADA_2024_9999.zip"
    path.write_bytes(_zip_bytes())
    scada, status = kelmarsh.load_archives([path])
    assert list(scada.columns) == SCADA_COLUMNS
    assert scada["turbine_id"].unique().tolist() == [3]
    assert scada["power_kw"].iloc[0] == pytest.approx(650.2)
    assert scada["ambient_temp"].iloc[1] == pytest.approx(5.1)
    assert status["iec_category"].tolist() == ["Forced Outage", "Partial Performance"]
    # A missing end time is filled so the event still covers at least one sample.
    assert status["end"].iloc[1] == pd.Timestamp("2024-01-02 00:10:00")


def test_kelmarsh_download_verifies_checksum(tmp_path):
    blob = _zip_bytes()
    files = [{"key": "Kelmarsh_SCADA_2024_9999.zip", "checksum": "md5:" + hashlib.md5(blob).hexdigest(),
              "links": {"self": "https://example.invalid/file"}}]

    class Session:
        calls = 0

        def get(self, url, **kwargs):
            Session.calls += 1
            return FakeResponse({"files": files}) if "api/records" in url else FakeResponse(content=blob)

    path = kelmarsh.download_year(2024, tmp_path, session=Session())
    assert path.read_bytes() == blob
    before = Session.calls
    kelmarsh.download_year(2024, tmp_path, session=Session())  # cached: only the listing call
    assert Session.calls == before + 1

    files[0]["checksum"] = "md5:" + "0" * 32
    path.unlink()
    with pytest.raises(OSError, match="checksum"):
        kelmarsh.download_year(2024, tmp_path, session=Session())
    with pytest.raises(FileNotFoundError):
        kelmarsh.download_year(2019, tmp_path, session=Session())


def _om_payload(start="2024-03-01", hours=48) -> dict:
    times = pd.date_range(start, periods=hours, freq="h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for var in openmeteo.VARIABLES:
        for day in openmeteo.DAYS:
            hourly[f"{var}_previous_day{day}"] = [float(day)] * hours
    hourly["wind_speed_100m_previous_day7"] = [None] * hours
    hourly["wind_speed_10m_previous_day7"] = [None] * hours
    return {"hourly": hourly}


def test_openmeteo_parse_maps_days_to_pseudo_runs():
    out = openmeteo.parse_response(_om_payload(), "gfs")
    assert list(out.columns) == NWP_COLUMNS
    lead_h = (out["valid_time"] - out["run_time"]) / pd.Timedelta("1h")
    assert sorted(lead_h.unique()) == [24.0 * d for d in range(1, 7)]  # day 7 was all-null and is dropped
    assert (out["available_time"] == out["run_time"]).all()
    assert (out.loc[lead_h == 48, "ws_100m"] == 2.0).all()


def test_openmeteo_fetch_retries_then_caches(tmp_path):
    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(status=503)
            assert params["models"] == "gfs_seamless" and "wind_speed_100m_previous_day1" in params["hourly"]
            return FakeResponse(_om_payload(params["start_date"], 48))

    session = Session()
    conn = openmeteo.OpenMeteoPreviousRuns("gfs", "gfs_seamless", 52.4, -0.94, tmp_path, session, pause_s=0)
    out = conn.fetch("2024-03-01", "2024-03-03")
    assert session.calls == 2 and out["valid_time"].min() == pd.Timestamp("2024-03-01")
    assert out["valid_time"].max() < pd.Timestamp("2024-03-03")
    conn.fetch("2024-03-01", "2024-03-03")
    assert session.calls == 2  # second fetch served from the on-disk cache


def test_fetch_all_survives_one_provider_failing(tmp_path):
    cfg = load_config(overrides={"data": {"start": "2024-03-01", "end": "2024-03-03"}})

    class Session:
        def get(self, url, params=None, timeout=None):
            if params["models"] == "icon_seamless":
                return FakeResponse(status=404)
            return FakeResponse(_om_payload(params["start_date"], 48))

    out = openmeteo.fetch_all(cfg, None, Session())
    assert set(out["provider"]) == {"gfs", "ecmwf"}


def test_lambda_ingest_writes_partitioned_objects(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parents[1] / "infra"))
    import lambda_ingest

    class S3:
        def __init__(self):
            self.keys = []

        def put_object(self, Bucket, Key, Body):
            assert Bucket == "raw-bucket" and len(Body) > 0
            self.keys.append(Key)

    class Session:
        def get(self, url, params=None, timeout=None):
            if params["models"] == "icon_seamless":
                return FakeResponse(status=404)
            hours = (pd.Timestamp(params["end_date"]) - pd.Timestamp(params["start_date"])).days * 24 + 24
            return FakeResponse(_om_payload(params["start_date"], hours))

    monkeypatch.setenv("RAW_BUCKET", "raw-bucket")
    monkeypatch.setenv("CONFIG_PATH", str(Path(__file__).parents[1] / "configs" / "kelmarsh.yaml"))
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: None)
    s3 = S3()
    result = lambda_ingest.handler({}, None, s3_client=s3, session=Session())
    assert result["failed_providers"] == ["icon"] and not result["all_failed"]
    assert result["written"] == len(s3.keys) > 0
    assert all(k.startswith(("nwp/provider=gfs/date=", "nwp/provider=ecmwf/date=")) for k in s3.keys)
    assert np.unique(s3.keys).size == len(s3.keys)

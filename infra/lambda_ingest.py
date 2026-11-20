"""AWS Lambda: near-real-time NWP ingestion.

Triggered hourly by EventBridge. Pulls the last few days of archived forecasts for
every configured provider and writes one Parquet object per provider and day:

    s3://<RAW_BUCKET>/nwp/provider=<name>/date=<YYYY-MM-DD>/part.parquet

Writes are overwrite-by-key, so a retry or a late re-run is harmless. A provider
that fails is reported in the result (the state machine alarms on it) but does not
stop the others.

Environment: RAW_BUCKET, CONFIG_PATH (YAML packaged with the function), LOOKBACK_DAYS.
"""
from __future__ import annotations

import io
import os
from datetime import UTC, datetime, timedelta

from windcast.config import load_config
from windcast.connectors.openmeteo import OpenMeteoPreviousRuns
from windcast.utils import get_logger

log = get_logger("windcast.lambda.ingest")


def handler(event: dict, context, s3_client=None, session=None) -> dict:
    import boto3

    cfg = load_config(os.environ.get("CONFIG_PATH", "configs/kelmarsh.yaml"))
    bucket = os.environ["RAW_BUCKET"]
    lookback = int(os.environ.get("LOOKBACK_DAYS", "2"))
    s3 = s3_client or boto3.client("s3")
    end = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=8)  # forecasts reach 7 days ahead
    start = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=lookback)

    written, failed = [], []
    for name, pcfg in cfg["providers"].items():
        try:
            conn = OpenMeteoPreviousRuns(name, pcfg["openmeteo_model"], cfg["site"]["latitude"],
                                         cfg["site"]["longitude"], session=session)
            frame = conn.fetch(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
            for day, part in frame.groupby(frame["valid_time"].dt.strftime("%Y-%m-%d")):
                buf = io.BytesIO()
                part.to_parquet(buf, index=False)
                key = f"nwp/provider={name}/date={day}/part.parquet"
                s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
                written.append(key)
        except Exception:  # noqa: BLE001 - isolate provider failures
            log.exception("provider ingest failed", extra={"ctx": {"provider": name}})
            failed.append(name)
    result = {"written": len(written), "failed_providers": failed, "all_failed": len(failed) == len(cfg["providers"])}
    log.info("ingest finished", extra={"ctx": result})
    return result

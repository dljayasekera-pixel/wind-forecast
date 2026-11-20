"""Small shared helpers: time handling, lead buckets, structured logging."""
from __future__ import annotations

import hashlib
import json
import logging
import sys

import numpy as np
import pandas as pd

# Lead-time buckets used for bias correction, blending weights and reporting.
LEAD_EDGES_H = [6, 24, 48, 96]
LEAD_LABELS = ["0-6h", "6-24h", "24-48h", "48-96h", "96h+"]


def to_ns(x):
    """Coerce to tz-naive UTC datetime64[ns].

    Every timestamp in this package is tz-naive UTC. ``merge_asof`` also needs
    both keys to share one dtype, so everything is forced to nanoseconds.
    """
    out = pd.to_datetime(x, utc=True)
    if isinstance(out, pd.Timestamp):
        return out.tz_localize(None)
    if isinstance(out, pd.Series):
        return out.dt.tz_localize(None).astype("datetime64[ns]")
    return out.tz_localize(None).astype("datetime64[ns]")


def lead_bucket(lead_h) -> np.ndarray:
    """Integer bucket index for a lead time in hours."""
    return np.digitize(np.asarray(lead_h, dtype=float), LEAD_EDGES_H, right=True)


def lead_bucket_label(lead_h) -> np.ndarray:
    return np.asarray(LEAD_LABELS)[lead_bucket(lead_h)]


def config_hash(cfg: dict) -> str:
    """Stable short hash of a config dict, used to version artifacts."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "windcast") -> logging.Logger:
    """JSON-lines logger. CloudWatch Logs Insights can query the fields directly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

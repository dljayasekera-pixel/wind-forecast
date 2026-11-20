"""Publish metrics to Amazon CloudWatch. Alarms on these metrics are defined in infra/main.tf."""
from __future__ import annotations

import math

from windcast.utils import get_logger

log = get_logger(__name__)


class CloudWatchEmitter:
    """Thin wrapper over ``put_metric_data``.

    ``client`` is injected so tests can pass a stub. With ``dry_run=True`` (the
    default) nothing leaves the machine; metrics are written to the log instead.
    """

    def __init__(self, namespace: str = "WindCast", site: str = "unknown", client=None, dry_run: bool = True):
        self.namespace, self.site, self.dry_run = namespace, site, dry_run
        self.client = client
        if client is None and not dry_run:
            import boto3  # imported lazily: optional dependency

            self.client = boto3.client("cloudwatch")

    def _datum(self, name: str, value: float) -> dict:
        unit = "Count" if name in {"n", "alerts"} else "None"
        return {"MetricName": name, "Value": float(value), "Unit": unit,
                "Dimensions": [{"Name": "Site", "Value": self.site}]}

    def put(self, metrics: dict) -> int:
        """Send every finite numeric metric. Returns the number sent."""
        data = [self._datum(k, v) for k, v in metrics.items()
                if isinstance(v, (int, float)) and math.isfinite(v)]
        if self.dry_run or self.client is None:
            log.info("cloudwatch dry run", extra={"ctx": {"namespace": self.namespace, "n_metrics": len(data)}})
            return len(data)
        for i in range(0, len(data), 20):  # API limit per call
            self.client.put_metric_data(Namespace=self.namespace, MetricData=data[i:i + 20])
        return len(data)

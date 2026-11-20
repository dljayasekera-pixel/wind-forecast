"""File-based model registry: versioned, immutable bundles plus a 'latest' pointer.

Layout (identical locally and under an S3 prefix):
    models/<version>/bundle.joblib
    models/<version>/metadata.json
    models/latest.json                -> {"version": "..."}

Promotion is a pointer change, so rollback is one write. A candidate can be run in
shadow by loading its version explicitly while 'latest' keeps serving.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib

from windcast import __version__
from windcast.utils import config_hash


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=False)
        return out.stdout.strip() or None
    except OSError:
        return None


def save(bundle, cfg: dict, root: str | Path, metrics: dict | None = None, promote: bool = True) -> str:
    """Write a new version and (optionally) point 'latest' at it. Returns the version id."""
    root = Path(root)
    version = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{config_hash(cfg)}"
    target = root / version
    target.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, target / "bundle.joblib")
    import lightgbm
    import xgboost

    metadata = {
        "version": version, "package_version": __version__, "git_sha": _git_sha(),
        "config_hash": config_hash(cfg), "config": cfg, "python": platform.python_version(),
        "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
        "bundle_meta": bundle.meta, "features": bundle.features, "metrics": metrics or {},
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if promote:
        promote_version(root, version)
    return version


def promote_version(root: str | Path, version: str) -> None:
    root = Path(root)
    if not (root / version / "bundle.joblib").exists():
        raise FileNotFoundError(f"model version {version} not found in {root}")
    (root / "latest.json").write_text(json.dumps({"version": version}))


def load(root: str | Path, version: str | None = None):
    """Load a bundle and its metadata; defaults to the promoted version."""
    root = Path(root)
    if version is None:
        pointer = root / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(f"no promoted model in {root}; run 'windcast train' first")
        version = json.loads(pointer.read_text())["version"]
    bundle = joblib.load(root / version / "bundle.joblib")
    metadata = json.loads((root / version / "metadata.json").read_text())
    return bundle, metadata


def upload_to_s3(root: str | Path, version: str, bucket: str, prefix: str = "models", client=None) -> list[str]:
    """Copy one version and the pointer to S3. Returns the keys written."""
    if client is None:
        import boto3

        client = boto3.client("s3")
    root, keys = Path(root), []
    for path in [*sorted((root / version).iterdir()), root / "latest.json"]:
        key = f"{prefix}/{path.relative_to(root).as_posix()}"
        client.upload_file(str(path), bucket, key)
        keys.append(key)
    return keys

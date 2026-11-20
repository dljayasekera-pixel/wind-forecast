"""Command line entry point: ``windcast <step> --config configs/synthetic.yaml``."""
from __future__ import annotations

import argparse
import json
import sys

from windcast import pipeline
from windcast.config import load_config
from windcast.utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="windcast", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("ingest", "prepare", "backtest", "run-all"):
        sub.add_parser(name).add_argument("--config", default=None)
    p = sub.add_parser("train")
    p.add_argument("--config", default=None)
    p.add_argument("--no-promote", action="store_true", help="register without moving the 'latest' pointer")
    p = sub.add_parser("predict")
    p.add_argument("--config", default=None)
    p.add_argument("--issue-time", default=None, help="UTC, e.g. 2024-11-15T06:00 (default: latest SCADA)")
    p.add_argument("--version", default=None, help="model version (default: promoted)")
    p = sub.add_parser("monitor")
    p.add_argument("--config", default=None)
    p.add_argument("--now", default=None)
    p.add_argument("--emit", action="store_true", help="publish to CloudWatch (needs AWS credentials)")
    p = sub.add_parser("load-sql")
    p.add_argument("--config", default=None)
    p.add_argument("--nwp-days", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("windcast.cli")
    try:
        cfg = load_config(args.config)
        if args.step == "run-all":
            out = pipeline.run_all(cfg)
        elif args.step == "train":
            out = pipeline.train(cfg, promote=not args.no_promote)
        elif args.step == "predict":
            out = pipeline.predict(cfg, args.issue_time, args.version).to_dict(orient="records")
        elif args.step == "monitor":
            out = pipeline.monitor(cfg, args.now, args.emit)
        elif args.step == "load-sql":
            out = pipeline.load_sql(cfg, args.nwp_days)
        else:
            out = pipeline.STEPS[args.step](cfg)
    except Exception:  # noqa: BLE001 - top-level handler: log with traceback, non-zero exit for the orchestrator
        log.exception("step failed", extra={"ctx": {"step": args.step}})
        return 1
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

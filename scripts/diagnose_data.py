"""Data-quality diagnostic: how many hours count as clean, and is that set representative?

    python scripts/diagnose_data.py --config configs/kelmarsh.yaml

Run after `windcast prepare`. A clean set that is small, or much calmer than the
full record, makes every accuracy figure look better (or worse) than it really is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from windcast.config import load_config
from windcast.scada.qc import map_iec_category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kelmarsh.yaml")
    cfg = load_config(parser.parse_args().config)
    root = Path(cfg["data"]["root"])
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 60)

    status = pd.read_parquet(root / "raw" / "status.parquet")
    status["hours"] = (status["end"] - status["start"]) / pd.Timedelta("1h")
    status["mapped_to"] = status["iec_category"].map(map_iec_category)
    print("\n== 1. Status log: categories and how they are treated")
    print(status.groupby(["iec_category", "mapped_to"], dropna=False)
          .agg(events=("hours", "size"), turbine_hours=("hours", "sum")).round(1)
          .sort_values("turbine_hours", ascending=False).head(25).to_string())

    flagged = pd.read_parquet(root / "curated" / "scada_flagged.parquet")
    flags = [c for c in flagged.columns if c.startswith("flag_")] + ["qc_ok", "operating_ok"]
    print("\n== 2. Share of 10-minute turbine samples per flag")
    print(flagged[flags].mean().round(4).to_string())

    farm = pd.read_parquet(root / "curated" / "farm.parquet")
    clean_cfg = cfg["clean"]
    good = farm["power_norm"].notna() & ~farm["dq"]

    def clean(min_avail: float, max_der: float) -> pd.Series:
        return good & (farm["availability"] >= min_avail) & (farm["derated_frac"] <= max_der)

    print("\n== 3. Share of hours that count as clean under different thresholds")
    rows = []
    for avail, der in [(0.999, 0.001), (clean_cfg["min_availability"], clean_cfg["max_derated_frac"]),
                       (0.9, 0.1)]:
        mask = clean(avail, der)
        rows.append({"min_availability": avail, "max_derated_frac": der, "clean_share": mask.mean(),
                     "mean_power_clean": farm.loc[mask, "power_norm"].mean(),
                     "mean_wind_clean": farm.loc[mask, "ws_obs"].mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"All valid hours: mean power {farm.loc[good, 'power_norm'].mean():.3f}, "
          f"mean wind {farm.loc[good, 'ws_obs'].mean():.2f} m/s")

    mask = clean(clean_cfg["min_availability"], clean_cfg["max_derated_frac"])
    bins = pd.cut(farm["ws_obs"], [0, 3, 5, 7, 9, 11, 13, np.inf])
    print("\n== 4. Clean share by wind speed (current thresholds): should be roughly flat")
    print(mask[good].groupby(bins[good], observed=True).agg(["mean", "size"]).round(3).to_string())

    print("\n== 5. Clean share by month")
    print(mask[good].groupby(farm.index[good].month).mean().round(3).to_string())


if __name__ == "__main__":
    main()

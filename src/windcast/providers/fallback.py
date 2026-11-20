"""Provider fallback: decide per row which providers are usable.

Tier 0  every provider delivered a fresh run
Tier 1  at least one provider is missing or stale; blend over the rest
Tier 2  no usable NWP; downstream models rely on SCADA lags and climatology
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.features.nwp import NWP_FEATURES


def apply_fallback(frame: pd.DataFrame, providers: list[str], max_age_h: float) -> pd.DataFrame:
    """Blank out stale providers and record the fallback tier. Returns a modified copy."""
    out = frame.copy()
    usable = np.zeros(len(out), dtype=int)
    for name in providers:
        age = out[f"{name}_age_h"]
        fresh = out[f"{name}_ws_hub"].notna() & (age <= max_age_h)
        cols = [f"{name}_{c}" for c in NWP_FEATURES + ["age_h"]]
        out.loc[~fresh, cols] = np.nan
        usable += fresh.to_numpy().astype(int)
    out["n_providers"] = usable
    out["fallback_tier"] = np.select([usable == len(providers), usable > 0], [0, 1], default=2)
    return out

"""WeatherStack: the fold-fitted weather-to-power chain.

fit()       on training rows only: per-provider bias correction against observed
            wind, then blending weights across providers.
transform() fallback -> bias-correct -> blend -> power-curve conversion, plus
            ensemble-spread features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.providers.bias_correction import make_corrector
from windcast.providers.blend import ProviderBlender
from windcast.providers.fallback import apply_fallback
from windcast.scada.power_curve import PowerCurve


class WeatherStack:
    def __init__(self, providers: list[str], power_curve: PowerCurve,
                 bias_correction: str = "mos", blend: str = "nnls", max_age_h: float = 30.0):
        self.providers = list(providers)
        self.power_curve = power_curve
        self.bias_correction, self.blend_method, self.max_age_h = bias_correction, blend, max_age_h
        self.correctors_ = {p: make_corrector(bias_correction) for p in self.providers}
        self.blender_ = ProviderBlender(self.providers, blend)

    def _corrected(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            np.where(df[f"{p}_ws_hub"].notna(),
                     self.correctors_[p].transform(df[f"{p}_ws_hub"], df["lead_h"]), np.nan)
            for p in self.providers])

    def fit(self, train: pd.DataFrame) -> WeatherStack:
        df = apply_fallback(train, self.providers, self.max_age_h)
        obs = df["tgt_ws_obs"].where(~df["tgt_dq"])
        for p in self.providers:
            self.correctors_[p].fit(df[f"{p}_ws_hub"], obs, df["lead_h"])
        self.blender_.fit(self._corrected(df), obs, df["lead_h"])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = apply_fallback(frame, self.providers, self.max_age_h)
        lead = df["lead_h"].to_numpy()
        pc = self.power_curve
        ws_bc = self._corrected(df)
        new = {}
        pcs = []
        for i, p in enumerate(self.providers):
            new[f"{p}_ws_bc"] = ws_bc[:, i]
            new[f"{p}_pc"] = pc.predict(ws_bc[:, i], df[f"{p}_rho"].to_numpy())
            pcs.append(new[f"{p}_pc"])
        pcs = np.column_stack(pcs)
        with _quiet():
            rho_mean = np.nanmean(np.column_stack([df[f"{p}_rho"] for p in self.providers]), axis=1)
        new["blend_ws"] = self.blender_.transform(ws_bc, lead)
        new["blend_pc"] = pc.predict(new["blend_ws"], rho_mean)
        with np.errstate(invalid="ignore"), _quiet():
            new["nwp_spread_ws"] = np.nanstd(ws_bc, axis=1)
            new["nwp_range_pc"] = np.nanmax(pcs, axis=1) - np.nanmin(pcs, axis=1)
            new["nwp_mean_pc"] = np.nanmean(pcs, axis=1)

        # Current NWP error: forecast for 'now' through the same chain vs. observed power.
        zero = np.zeros(len(df))
        now_bc = np.column_stack([
            np.where(df[f"{p}_ws_hub_now"].notna(),
                     self.correctors_[p].transform(df[f"{p}_ws_hub_now"], zero), np.nan)
            for p in self.providers])
        with np.errstate(invalid="ignore"), _quiet():
            rho_now = np.nanmean(np.column_stack([df[f"{p}_rho_now"] for p in self.providers]), axis=1)
        new["blend_pc_now"] = pc.predict(self.blender_.transform(now_bc, zero), rho_now)
        new["nwp_err_now"] = df["obs_power_t0"].to_numpy() - new["blend_pc_now"]
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


class _quiet:
    """Silence numpy's all-NaN-slice warnings; those rows are the tier-2 fallback case."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

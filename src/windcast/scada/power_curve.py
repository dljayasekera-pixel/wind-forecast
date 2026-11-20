"""Empirical power curve (IEC 61400-12-1 method of bins) and air-density correction."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RHO0 = 1.225  # kg/m3, ISO standard atmosphere at sea level
R_DRY = 287.05  # J/(kg K)


def standard_pressure_hpa(elevation_m: float) -> float:
    """Barometric formula; used when a site has no pressure measurement."""
    return 1013.25 * (1 - 2.25577e-5 * elevation_m) ** 5.25588


def air_density(pressure_hpa, temp_c):
    """Dry-air density in kg/m3."""
    return np.asarray(pressure_hpa, dtype=float) * 100.0 / (R_DRY * (np.asarray(temp_c, dtype=float) + 273.15))


def density_correct(ws, rho):
    """Equivalent wind speed at standard density (IEC 61400-12-1 for pitch-regulated turbines)."""
    return np.asarray(ws, dtype=float) * (np.asarray(rho, dtype=float) / RHO0) ** (1.0 / 3.0)


def shear_exponent(ws_low, ws_high, z_low: float, z_high: float):
    """Power-law shear exponent from two heights, clipped to a physical range."""
    ws_low, ws_high = np.asarray(ws_low, dtype=float), np.asarray(ws_high, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(ws_high / ws_low) / np.log(z_high / z_low)
    alpha = np.where((ws_low > 0.5) & np.isfinite(alpha), alpha, 1.0 / 7.0)
    return np.clip(alpha, 0.0, 0.6)


def extrapolate_to_hub(ws_ref, alpha, z_ref: float, z_hub: float):
    return np.asarray(ws_ref, dtype=float) * (z_hub / z_ref) ** np.asarray(alpha, dtype=float)


@dataclass
class PowerCurve:
    """Binned, monotone power curve in normalised units (power / rated)."""

    bin_width: float = 0.5
    ws_max: float = 30.0
    cut_out: float | None = 24.0
    min_count: int = 5
    stat: str = "mean"  # 'median' is more robust and is used for QC
    centers_: np.ndarray = field(default=None, repr=False)
    curve_: np.ndarray = field(default=None, repr=False)
    sigma_: np.ndarray = field(default=None, repr=False)

    def fit(self, ws, power_norm, rho=None) -> PowerCurve:
        ws = np.asarray(ws, dtype=float)
        power = np.asarray(power_norm, dtype=float)
        if rho is not None:
            ws = density_correct(ws, rho)
        ok = np.isfinite(ws) & np.isfinite(power)
        ws, power = ws[ok], power[ok]
        edges = np.arange(0.0, self.ws_max + self.bin_width, self.bin_width)
        centers = (edges[:-1] + edges[1:]) / 2
        which = np.digitize(ws, edges) - 1
        curve = np.full(len(centers), np.nan)
        sigma = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            vals = power[which == b]
            if len(vals) >= self.min_count:
                curve[b] = np.median(vals) if self.stat == "median" else np.mean(vals)
                sigma[b] = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        filled = np.isfinite(curve)
        if filled.sum() < 3:
            raise ValueError("not enough data to fit a power curve")
        curve = np.interp(centers, centers[filled], curve[filled])
        sigma = np.interp(centers, centers[filled], sigma[filled])
        # Monotone non-decreasing up to the peak, flat afterwards.
        peak = int(np.argmax(curve))
        curve[: peak + 1] = np.maximum.accumulate(curve[: peak + 1])
        curve[peak:] = curve[peak]
        self.centers_, self.curve_, self.sigma_ = centers, np.clip(curve, 0.0, 1.05), sigma
        return self

    def predict(self, ws, rho=None) -> np.ndarray:
        ws = np.asarray(ws, dtype=float)
        if rho is not None:
            ws = density_correct(ws, rho)
        out = np.interp(ws, self.centers_, self.curve_, left=0.0)
        if self.cut_out is not None:
            out = np.where(ws >= self.cut_out, 0.0, out)
        return np.where(np.isfinite(ws), out, np.nan)

    def sigma_at(self, ws) -> np.ndarray:
        return np.interp(np.asarray(ws, dtype=float), self.centers_, self.sigma_)

    def to_dict(self) -> dict:
        return {"centers": self.centers_.tolist(), "curve": self.curve_.tolist(),
                "sigma": self.sigma_.tolist(), "bin_width": self.bin_width, "cut_out": self.cut_out}

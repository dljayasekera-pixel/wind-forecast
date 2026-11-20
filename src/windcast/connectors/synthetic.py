"""Synthetic wind farm and NWP providers with the same schema as the real feeds.

Purpose: CI and offline development. The generator plants known faults
(outages, curtailment, icing, stuck anemometers, comms gaps, late and missing
NWP runs) so that QC, fallback and root-cause logic can be tested against
ground truth. It is not a substitute for real data when quoting accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import norm

from windcast.connectors.base import validate_nwp, validate_scada, validate_status

# Error characteristics per synthetic provider. Real providers are configured by
# name in the YAML; any name without a profile here gets DEFAULT_PROFILE.
PROVIDER_PROFILES = {
    "gfs": dict(latency_h=4.5, max_lead_h=240, mult_bias=0.07, add_bias=0.3,
                s0=0.9, s1=0.60, miss=0.02, outage_days=0),
    "ecmwf": dict(latency_h=7.0, max_lead_h=240, mult_bias=-0.03, add_bias=0.0,
                  s0=0.7, s1=0.45, miss=0.01, outage_days=0),
    "icon": dict(latency_h=3.5, max_lead_h=180, mult_bias=0.0, add_bias=-0.4,
                 s0=0.8, s1=0.65, miss=0.03, outage_days=4),
}
DEFAULT_PROFILE = dict(latency_h=5.0, max_lead_h=240, mult_bias=0.0, add_bias=0.0,
                       s0=0.9, s1=0.6, miss=0.02, outage_days=0)


def true_power_curve(ws, cut_in=3.0, rated_ws=12.5, cut_out=24.0):
    """Idealised normalised power curve (cubic between cut-in and rated)."""
    ws = np.asarray(ws, dtype=float)
    frac = (ws**3 - cut_in**3) / (rated_ws**3 - cut_in**3)
    power = np.clip(frac, 0.0, 1.0)
    return np.where((ws < cut_in) | (ws >= cut_out), 0.0, power)


def _ar1(shape, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary unit-variance AR(1) noise along the last axis."""
    eps = rng.standard_normal(shape)
    eps[..., 1:] *= np.sqrt(1.0 - rho**2)
    return lfilter([1.0], [1.0, -rho], eps, axis=-1)


def make_truth(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Hourly 'true' hub-height weather with synoptic, mesoscale, seasonal and diurnal structure."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(idx)
    syn = _ar1(n, np.exp(-1 / 40), rng)
    meso = _ar1(n, np.exp(-1 / 5), rng)
    z = (0.85 * syn + 0.5 * meso) / np.hypot(0.85, 0.5)
    u = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)
    doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()
    scale = 8.3 * (1 + 0.18 * np.cos(2 * np.pi * (doy - 15) / 365)) \
        * (1 + 0.05 * np.cos(2 * np.pi * (hour - 14) / 24))
    ws_hub = scale * (-np.log1p(-u)) ** (1 / 2.1)  # Weibull marginal, k = 2.1
    alpha = np.clip(0.18 + 0.08 * np.cos(2 * np.pi * (hour - 3) / 24)
                    + 0.03 * _ar1(n, 0.9, rng), 0.05, 0.40)
    temp = 10 + 7 * np.cos(2 * np.pi * (doy - 200) / 365) \
        + 3 * np.cos(2 * np.pi * (hour - 15) / 24) + 2.5 * _ar1(n, np.exp(-1 / 30), rng)
    pressure = 998 + 9 * _ar1(n, np.exp(-1 / 60), rng)
    wd = (230 + 70 * _ar1(n, np.exp(-1 / 50), rng) + 15 * _ar1(n, np.exp(-1 / 6), rng)) % 360
    return pd.DataFrame({"ws_hub": ws_hub, "alpha": alpha, "temp_c": temp,
                         "pressure_hpa": pressure, "wd": wd}, index=idx.rename("time"))


def make_scada(truth: pd.DataFrame, site: dict, seed: int = 42):
    """10-minute turbine SCADA, a status log, and the list of planted events."""
    rng = np.random.default_rng(seed + 1)
    n_t, rated = int(site["n_turbines"]), float(site["rated_kw"])
    idx = pd.date_range(truth.index[0], truth.index[-1] + pd.Timedelta("50min"), freq="10min")
    base = truth.reindex(idx).interpolate("time").ffill()
    n = len(idx)
    days = n / 144

    factors = np.linspace(0.95, 1.03, n_t)[rng.permutation(n_t)][:, None]
    ws = np.clip(base["ws_hub"].to_numpy()[None, :] * factors
                 * (1 + 0.09 * _ar1((n_t, n), 0.7, rng)), 0, None)
    temp = base["temp_c"].to_numpy()
    rho = base["pressure_hpa"].to_numpy() * 100 / (287.05 * (temp + 273.15))
    ws_eq = ws * (rho[None, :] / 1.225) ** (1 / 3)
    frac = true_power_curve(ws_eq, site["cut_in_ms"], 12.5, site["cut_out_ms"])
    power = frac * rated + 12 * rng.standard_normal((n_t, n))
    power = np.where(frac <= 0, -5 + 2 * rng.standard_normal((n_t, n)), power)
    ws_meas = np.clip(ws + 0.15 * rng.standard_normal((n_t, n)), 0, None)
    wd = (base["wd"].to_numpy()[None, :] + 4 * rng.standard_normal((n_t, n))) % 360
    temp_meas = temp[None, :] + 0.3 * rng.standard_normal((n_t, n))

    status_rows, events = [], []

    def _span(dur_steps):
        i0 = int(rng.integers(0, max(1, n - dur_steps)))
        return i0, min(n, i0 + dur_steps)

    # Turbine outages, always logged.
    for k in range(n_t):
        for _ in range(rng.poisson(days / 45)):
            dur = int(np.clip(rng.lognormal(np.log(8), 1.0), 1, 120) * 6)
            i0, i1 = _span(dur)
            power[k, i0:i1] = -5 + 2 * rng.standard_normal(i1 - i0)
            cat = "Forced Outage" if rng.random() < 0.7 else "Scheduled Maintenance"
            status_rows.append((k + 1, idx[i0], idx[i1 - 1] + pd.Timedelta("10min"), cat))
            events.append(("outage", k + 1, idx[i0], idx[i1 - 1] + pd.Timedelta("10min")))

    # Farm-wide curtailment. Only ~70% of events reach the status log, so the
    # statistical under-performance flag has real work to do.
    for _ in range(rng.poisson(days / 12)):
        i0, i1 = _span(int(rng.integers(12, 60)))
        cap = rng.uniform(0.3, 0.7) * rated
        power[:, i0:i1] = np.minimum(power[:, i0:i1], cap)
        logged = rng.random() < 0.7
        end = idx[i1 - 1] + pd.Timedelta("10min")
        if logged:
            status_rows += [(k + 1, idx[i0], end, "Partial Performance") for k in range(n_t)]
        events.append(("curtailment" if logged else "curtailment_unlogged", 0, idx[i0], end))

    # Icing: unlogged derate on roughly half of the cold days.
    day_id = (np.arange(n) // 144)
    icy_days = rng.random(day_id.max() + 1) < 0.5
    ice = (temp < 0.5) & icy_days[day_id]
    power[:, ice] = np.where(power[:, ice] > 0, power[:, ice] * 0.6, power[:, ice])
    if ice.any():
        events.append(("icing", 0, idx[ice][0], idx[ice][-1]))

    # Stuck anemometers.
    for _ in range(max(1, int(days / 120))):
        k = int(rng.integers(0, n_t))
        i0, i1 = _span(int(rng.integers(36, 144)))
        ws_meas[k, i0:i1] = ws_meas[k, i0]
        events.append(("stuck_sensor", k + 1, idx[i0], idx[i1 - 1] + pd.Timedelta("10min")))

    power = np.clip(power, -10, rated * 1.02)
    scada = pd.DataFrame({
        "time": np.tile(idx.to_numpy(), n_t),
        "turbine_id": np.repeat(np.arange(1, n_t + 1), n),
        "power_kw": power.ravel(), "wind_speed": ws_meas.ravel(),
        "wind_dir": wd.ravel(), "ambient_temp": temp_meas.ravel(),
    })

    # Missing data: random drops plus farm-wide communication gaps.
    keep = rng.random(len(scada)) > 0.008
    for _ in range(max(1, int(days / 180))):
        i0, i1 = _span(int(rng.integers(18, 72)))
        keep &= ~scada["time"].between(idx[i0], idx[i1 - 1]).to_numpy()
        events.append(("comms_gap", 0, idx[i0], idx[i1 - 1] + pd.Timedelta("10min")))
    scada = scada[keep]

    status = pd.DataFrame(status_rows, columns=["turbine_id", "start", "end", "iec_category"])
    events = pd.DataFrame(events, columns=["kind", "turbine_id", "start", "end"])
    return validate_scada(scada), validate_status(status), events


def make_nwp(truth: pd.DataFrame, providers: dict, site: dict, seed: int = 42,
             return_events: bool = False):
    """Forecast runs for each provider: error grows with lead and is partly shared between providers.

    Also plants one window in which no provider delivers anything (an ingestion
    outage on our side), which is what drives the tier-2 fallback.
    """
    rng = np.random.default_rng(seed + 2)
    hub = float(site["hub_height_m"])
    runs6 = pd.date_range(truth.index[0], truth.index[-1], freq="6h")
    leads = np.arange(0, 241)
    n_truth = len(truth)
    common = _ar1((len(runs6), len(leads)), 0.92, rng)
    t = {c: truth[c].to_numpy() for c in truth.columns}
    feed_down = np.zeros(len(runs6), dtype=bool)
    events = []
    if len(runs6) > 240:  # only for spans long enough to spare three days
        f0 = int(rng.integers(len(runs6) // 3, 2 * len(runs6) // 3))
        feed_down[f0:f0 + 12] = True
        events.append(("nwp_feed_outage", 0, runs6[f0], runs6[f0 + 12]))
    frames = []
    for name, pcfg in providers.items():
        prof = {**DEFAULT_PROFILE, **PROVIDER_PROFILES.get(name, {})}
        step = max(1, int(pcfg.get("cycle_h", 6)) // 6)
        sel = np.arange(0, len(runs6), step)
        runs = runs6[sel]
        own = _ar1((len(runs), len(leads)), 0.92, rng)
        eps = 0.6 * common[sel] + 0.8 * own
        run_hours = (sel * 6)[:, None]
        vi = run_hours + leads[None, :]
        ok = (vi < n_truth) & (leads[None, :] <= prof["max_lead_h"])
        vic = np.clip(vi, 0, n_truth - 1)
        ws_true = t["ws_hub"][vic]
        sigma = (prof["s0"] + prof["s1"] * np.sqrt(leads / 24))[None, :] * (0.55 + 0.06 * ws_true)
        ws_hub = np.clip(ws_true * (1 + prof["mult_bias"]) + prof["add_bias"] + sigma * eps, 0, None)
        alpha = np.clip(t["alpha"][vic] + 0.03 * rng.standard_normal(vi.shape), 0.02, 0.5)
        lead_scale = np.sqrt(leads / 24)[None, :]
        wd = (t["wd"][vic] + (5 + 6 * lead_scale) * _ar1(vi.shape, 0.9, rng)) % 360
        temp = t["temp_c"][vic] + (0.5 + 0.4 * lead_scale) * _ar1(vi.shape, 0.9, rng)
        pres = t["pressure_hpa"][vic] + (0.4 + 0.6 * lead_scale) * _ar1(vi.shape, 0.9, rng)

        # Delivery problems: randomly missing runs, one multi-day outage, variable latency.
        delivered = rng.random(len(runs)) > prof["miss"]
        if prof["outage_days"]:
            r0 = int(rng.integers(len(runs) // 3, 2 * len(runs) // 3))
            delivered[r0:r0 + int(prof["outage_days"] * 24 / (6 * step))] = False
        ok &= (delivered & ~feed_down[sel])[:, None]
        latency_h = prof["latency_h"] + rng.gamma(2.0, 0.25, len(runs))
        avail = runs + pd.to_timedelta(np.round(latency_h * 60), unit="min")

        r_idx, l_idx = np.nonzero(ok)
        frames.append(pd.DataFrame({
            "provider": name,
            "run_time": runs.to_numpy()[r_idx],
            "valid_time": truth.index.to_numpy()[vi[r_idx, l_idx]],
            "available_time": avail.to_numpy()[r_idx],
            "ws_10m": (ws_hub * (10 / hub) ** alpha)[r_idx, l_idx],
            "ws_100m": (ws_hub * (100 / hub) ** alpha)[r_idx, l_idx],
            "wd_100m": wd[r_idx, l_idx],
            "temp_2m": temp[r_idx, l_idx],
            "pressure_hpa": pres[r_idx, l_idx],
        }))
    nwp = validate_nwp(pd.concat(frames, ignore_index=True))
    if return_events:
        return nwp, pd.DataFrame(events, columns=["kind", "turbine_id", "start", "end"])
    return nwp


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    """Everything a pipeline run needs, from one seed."""
    seed = int(cfg["data"]["seed"])
    truth = make_truth(cfg["data"]["start"], cfg["data"]["end"], seed)
    scada, status, events = make_scada(truth, cfg["site"], seed)
    nwp, nwp_events = make_nwp(truth, cfg["providers"], cfg["site"], seed, return_events=True)
    events = pd.concat([events, nwp_events], ignore_index=True)
    return {"scada": scada, "status": status, "nwp": nwp, "events": events,
            "truth": truth.reset_index()}

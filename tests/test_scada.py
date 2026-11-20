from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from windcast.connectors.synthetic import true_power_curve
from windcast.scada.aggregate import to_farm
from windcast.scada.power_curve import (
    PowerCurve,
    air_density,
    density_correct,
    extrapolate_to_hub,
    shear_exponent,
    standard_pressure_hpa,
)
from windcast.scada.qc import clean_scada, map_iec_category

SITE = {"rated_kw": 2000.0, "n_turbines": 2, "cut_in_ms": 3.0, "cut_out_ms": 24.0, "elevation_m": 0.0}


def _handmade(days: int = 20, seed: int = 0):
    """Two turbines following the ideal curve exactly, at standard density."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=days * 144, freq="10min")
    frames = []
    for tid in (1, 2):
        ws = np.clip(8 + 4 * np.sin(np.arange(len(times)) / 50 + tid) + rng.normal(0, 0.8, len(times)), 0, None)
        frames.append(pd.DataFrame({"time": times, "turbine_id": tid, "wind_speed": ws,
                                    "power_kw": true_power_curve(ws) * 2000.0 + rng.normal(0, 5, len(times)),
                                    "wind_dir": 200.0, "ambient_temp": 15.0}))
    status = pd.DataFrame(columns=["turbine_id", "start", "end", "iec_category"])
    return pd.concat(frames, ignore_index=True), status, times


def test_density_and_shear_physics():
    assert air_density(1013.25, 15.0) == pytest.approx(1.225, abs=0.002)
    assert standard_pressure_hpa(0) == pytest.approx(1013.25)
    assert standard_pressure_hpa(1000) == pytest.approx(898.7, abs=1.0)
    # Denser air carries more energy: equivalent wind speed rises.
    assert density_correct(10.0, 1.30) > 10.0 > density_correct(10.0, 1.10)
    alpha = shear_exponent(5.0, 5.0 * 10**0.2, 10.0, 100.0)
    assert alpha == pytest.approx(0.2)
    assert extrapolate_to_hub(8.0, 0.2, 100.0, 100.0) == pytest.approx(8.0)
    assert extrapolate_to_hub(8.0, 0.2, 100.0, 80.0) < 8.0
    assert shear_exponent(0.1, 3.0, 10.0, 100.0) == pytest.approx(1 / 7)  # calm near the ground: default


def test_power_curve_recovers_truth_and_is_monotone():
    rng = np.random.default_rng(1)
    ws = rng.uniform(0, 26, 20_000)
    power = true_power_curve(ws) + rng.normal(0, 0.01, len(ws))
    pc = PowerCurve().fit(ws, power)
    grid = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
    assert np.abs(pc.predict(grid) - true_power_curve(grid)).max() < 0.03
    below_cutout = pc.centers_ < 24
    assert (np.diff(pc.curve_[below_cutout]) >= -1e-12).all()
    assert pc.predict(2.0) < 0.01 and pc.predict(25.0) == 0.0 and np.isnan(pc.predict(np.nan))
    # Same wind, thinner air, less power.
    assert pc.predict(8.0, rho=1.10) < pc.predict(8.0, rho=1.225)


def test_power_curve_needs_data():
    with pytest.raises(ValueError):
        PowerCurve().fit([5.0, 6.0], [0.1, 0.2])


def test_iec_category_mapping():
    assert map_iec_category("Full Performance") == "ok"
    assert map_iec_category("Partial Performance") == "derated"
    assert map_iec_category("Forced Outage") == "outage"
    assert map_iec_category("Scheduled Maintenance") == "outage"
    assert map_iec_category("Technical Standby") == "outage"
    assert map_iec_category("Requested Shutdown") == "outage"
    assert map_iec_category("Out of Environmental Specification") == "ok"  # calm or storm: weather
    assert map_iec_category("") == "ok" and map_iec_category("Unknown") == "ok"


def test_qc_flags_each_planted_fault():
    scada, status, times = _handmade()
    t1 = scada["turbine_id"] == 1
    idx = scada.index[t1]
    jitter = np.random.default_rng(1).normal(0, 0.05, 60)
    scada.loc[idx[100:160], "wind_speed"] = 10.0 + jitter
    scada.loc[idx[100:160], "power_kw"] = 0.35 * true_power_curve(10.0) * 2000  # unlogged derate
    scada.loc[idx[300:340], "wind_speed"] = 9.0 + jitter[:40]
    scada.loc[idx[300:340], "power_kw"] = 0.0                                   # unlogged stop
    scada.loc[idx[500:530], "wind_speed"] = 7.77                                # frozen anemometer
    scada.loc[idx[700], "power_kw"] = 9999.0                                    # impossible value
    scada = scada.drop(idx[900:912])                                            # telemetry gap
    status = pd.DataFrame({"turbine_id": [2], "start": [times[1000]], "end": [times[1100]],
                           "iec_category": ["Forced Outage"]})

    flagged, _ = clean_scada(scada, status, SITE)
    one = flagged[flagged["turbine_id"] == 1].reset_index(drop=True)
    two = flagged[flagged["turbine_id"] == 2].reset_index(drop=True)

    assert one.loc[100:159, "flag_underperf"].mean() > 0.95
    assert one.loc[100:159, "flag_derated"].mean() > 0.95 and not one.loc[100:159, "operating_ok"].any()
    assert one.loc[300:339, "flag_outage"].all()
    assert one.loc[505:529, "flag_stuck"].all()
    assert one.loc[700, "flag_range"] and not one.loc[700, "qc_ok"]
    assert one.loc[900:911, "flag_missing"].all()
    assert two.loc[1000:1099, "flag_outage"].all() and not two.loc[1100, "flag_outage"]
    # Healthy samples are overwhelmingly left alone.
    healthy = one.loc[1200:]
    assert healthy["operating_ok"].mean() > 0.97


def test_farm_series_is_stamped_at_period_end():
    scada, status, times = _handmade(days=3)
    scada["power_kw"] = 0.0
    hour = (scada["time"] >= "2024-01-02 05:00") & (scada["time"] < "2024-01-02 06:00")
    scada.loc[hour, "power_kw"] = 1000.0
    scada["wind_speed"] = np.where(hour, 9.0, 1.0) + np.random.default_rng(0).normal(0, 0.01, len(scada))
    flagged, _ = clean_scada(scada, status, SITE)
    farm = to_farm(flagged, SITE, "1h")
    # Samples 05:00..05:50 cover 05:00-06:00, so the hourly value belongs to the 06:00 stamp.
    assert farm.loc["2024-01-02 06:00", "power_norm"] == pytest.approx(0.5)
    assert farm.loc["2024-01-02 05:00", "power_norm"] == pytest.approx(0.0)
    assert farm.loc["2024-01-02 06:00", "power_kw"] == pytest.approx(2000.0)


def test_farm_handles_outage_and_missing_turbines():
    scada, status, times = _handmade(days=3)
    status = pd.DataFrame({"turbine_id": [2], "start": [pd.Timestamp("2024-01-02 00:00")],
                           "end": [pd.Timestamp("2024-01-02 12:00")], "iec_category": ["Forced Outage"]})
    gap = (scada["time"] >= "2024-01-03 00:00") & (scada["time"] < "2024-01-03 06:00")
    scada = scada[~gap]
    flagged, _ = clean_scada(scada, status, SITE)
    farm = to_farm(flagged, SITE, "1h")
    assert farm.loc["2024-01-02 06:00", "availability"] == pytest.approx(0.5)
    assert farm.loc["2024-01-01 12:00", "availability"] == pytest.approx(1.0)
    assert np.isnan(farm.loc["2024-01-03 03:00", "power_norm"]) and farm.loc["2024-01-03 03:00", "dq"]
    assert not farm.loc["2024-01-01 12:00", "dq"]


def test_synthetic_qc_finds_logged_outages(data, flagged):
    df = flagged[0]
    outages = data["events"][data["events"]["kind"] == "outage"]
    assert len(outages) > 0
    hit = []
    for ev in outages.itertuples():
        rows = df[(df["turbine_id"] == ev.turbine_id) & (df["time"] >= ev.start) & (df["time"] < ev.end)]
        rows = rows[~rows["flag_missing"]]
        if len(rows):
            hit.append(rows["flag_outage"].mean())
    assert np.mean(hit) > 0.99

"""Feature tests. The important ones prove there is no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd

from windcast.config import load_config
from windcast.features.build import build_frame, feature_columns, obs_features
from windcast.features.nwp import prepare_nwp, select_latest


def _nwp(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["run_time", "valid_time", "available_time", "ws_100m"])
    for col in ("run_time", "valid_time", "available_time"):
        df[col] = pd.to_datetime(df[col])
    return df.assign(provider="x", ws_10m=df["ws_100m"] * 0.7, wd_100m=180.0, temp_2m=10.0, pressure_hpa=1000.0)


def test_select_latest_uses_only_runs_available_at_issue_time():
    nwp = _nwp([
        ("2024-01-01 00:00", "2024-01-02 00:00", "2024-01-01 05:00", 5.0),
        ("2024-01-01 06:00", "2024-01-02 00:00", "2024-01-01 11:00", 6.0),
        ("2024-01-01 12:00", "2024-01-02 00:00", "2024-01-01 17:00", 7.0),
    ])
    prepared = prepare_nwp(nwp, 100.0)
    valid = ["2024-01-02 00:00"] * 4
    issue = ["2024-01-01 04:59", "2024-01-01 05:00", "2024-01-01 12:00", "2024-01-01 17:00"]
    out = select_latest(prepared, pd.to_datetime(issue), pd.to_datetime(valid))
    # Before anything is published: nothing. The 12:00 run exists at 12:00 but is not available until 17:00.
    assert np.isnan(out["ws_hub"].iloc[0])
    assert out["ws_hub"].tolist()[1:] == [5.0, 6.0, 7.0]
    assert out["age_h"].tolist()[1:] == [5.0, 6.0, 5.0]


def test_select_latest_preserves_input_order():
    nwp = _nwp([("2024-01-01 00:00", f"2024-01-01 {h:02d}:00", "2024-01-01 01:00", float(h)) for h in range(2, 10)])
    prepared = prepare_nwp(nwp, 100.0)
    valid = pd.to_datetime(["2024-01-01 07:00", "2024-01-01 03:00", "2024-01-01 05:00"])
    issue = pd.to_datetime(["2024-01-01 06:00", "2024-01-01 02:00", "2024-01-01 04:00"])
    assert select_latest(prepared, issue, valid)["ws_hub"].tolist() == [7.0, 3.0, 5.0]


def test_ramp_is_within_run_and_needs_a_three_hour_step():
    nwp = _nwp([("2024-01-01 00:00", f"2024-01-01 {h:02d}:00", "2024-01-01 01:00", float(h)) for h in range(0, 8)])
    prepared = prepare_nwp(nwp, 100.0)
    assert prepared["ramp_3h"].iloc[0] == 3.0
    assert prepared["ramp_3h"].iloc[-3:].isna().all()


def test_frame_keys_and_targets_line_up(cfg, farm, frame):
    row = frame.dropna(subset=["y"]).iloc[1234]
    assert row["valid_time"] - row["issue_time"] == pd.Timedelta(hours=row["lead_h"])
    assert row["y"] == farm.loc[row["valid_time"], "power_norm"]
    assert row["obs_power_t0"] == farm.loc[row["issue_time"], "power_norm"]
    assert set(frame["lead_h"].unique()) == {1.0, 3.0, 6.0, 12.0, 24.0, 48.0}


def test_no_feature_changes_when_the_future_changes(cfg, data, farm):
    """Corrupt every observation and every NWP run after a cutoff; features issued before it must not move."""
    cutoff = pd.Timestamp("2024-03-01 00:00")
    issue_times = pd.date_range("2024-02-20", cutoff, freq="6h")
    base = build_frame(farm, data["nwp"], cfg, issue_times=issue_times)

    farm_bad = farm.copy()
    after = farm_bad.index > cutoff
    farm_bad.loc[after, ["power_norm", "ws_obs", "availability", "derated_frac"]] = 0.123
    nwp_bad = data["nwp"].copy()
    late = nwp_bad["available_time"] > cutoff
    nwp_bad.loc[late, ["ws_10m", "ws_100m", "temp_2m"]] = 99.0
    changed = build_frame(farm_bad, nwp_bad, cfg, issue_times=issue_times)

    feats = feature_columns(base)
    pd.testing.assert_frame_equal(base[feats], changed[feats])
    # The targets, by contrast, do change: the corruption was real.
    assert not base["y"].equals(changed["y"])


def test_feature_columns_exclude_target_side_information(frame):
    feats = feature_columns(frame)
    assert "lead_h" in feats and "obs_power_t0" in feats and "gfs_ws_hub" in feats
    assert not [c for c in feats if c.startswith("tgt_") or c in ("y", "issue_time", "valid_time")]
    assert not [c for c in feats if "doy" in c]  # seasonal terms are off by default


def test_obs_features_handle_gaps(cfg):
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    farm = pd.DataFrame({"power_norm": np.linspace(0, 1, 48), "ws_obs": 8.0, "availability": 1.0,
                         "derated_frac": 0.0}, index=idx)
    farm.loc[idx[20:30], "power_norm"] = np.nan
    out = obs_features(farm, cfg)
    assert out["obs_power_t0"].iloc[22] == farm["power_norm"].iloc[19]  # carried forward
    assert out["obs_stale_steps"].iloc[22] == 3
    assert np.isnan(out["obs_power_t0"].iloc[27])  # beyond stale_limit_steps: give up
    assert out["obs_power_lag1"].iloc[10] == farm["power_norm"].iloc[9]


def test_sub_hourly_leads_land_exactly_on_the_grid(cfg, data, flagged):
    from windcast.scada.aggregate import to_farm

    cfg10 = load_config(overrides={"forecast": {"resolution": "10min", "issue_freq": "1h",
                                                "leads": ["10min", "20min", "40min", "60min"],
                                                "roll_steps": [3, 6]}})
    farm10 = to_farm(flagged[0], cfg10["site"], "10min")
    issue = pd.date_range("2024-02-01", periods=24, freq="h")
    frame = build_frame(farm10, data["nwp"], cfg10, issue_times=issue)
    assert (frame["valid_time"].dt.nanosecond == 0).all() and (frame["valid_time"].dt.second == 0).all()
    assert frame["y"].notna().mean() > 0.9
    assert frame["gfs_ws_hub"].notna().mean() > 0.9

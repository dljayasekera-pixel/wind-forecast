"""Shared fixtures: a small synthetic site, built once per test session."""
from __future__ import annotations

import pytest

from windcast.config import load_config
from windcast.connectors import synthetic
from windcast.features.build import build_frame
from windcast.scada.aggregate import to_farm
from windcast.scada.qc import clean_scada

SMALL = {
    "data": {"source": "synthetic", "start": "2024-01-01", "end": "2024-05-01", "seed": 11},
    "forecast": {"issue_freq": "3h", "leads": ["1h", "3h", "6h", "12h", "24h", "48h"]},
    "backtest": {"initial_train_days": 60, "test_days": 20,
                 "arima": {"enabled": True, "order": [2, 0, 1], "max_lead_h": 6,
                           "max_origins_per_fold": 12, "history_steps": 200}},
    "model": {"n_jobs": 2,
              "lgbm": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 60,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7},
              "xgb": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 4, "min_child_weight": 30}},
}


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(overrides=SMALL)


@pytest.fixture(scope="session")
def data(cfg) -> dict:
    return synthetic.generate(cfg)


@pytest.fixture(scope="session")
def flagged(cfg, data):
    return clean_scada(data["scada"], data["status"], cfg["site"])


@pytest.fixture(scope="session")
def farm(cfg, flagged):
    return to_farm(flagged[0], cfg["site"], cfg["forecast"]["resolution"])


@pytest.fixture(scope="session")
def frame(cfg, data, farm):
    return build_frame(farm, data["nwp"], cfg)

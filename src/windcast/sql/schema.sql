-- windcast relational schema.
-- Written for SQLite so the repo runs with no server. It ports to PostgreSQL /
-- TimescaleDB by changing TEXT timestamps to TIMESTAMPTZ and INTEGER flags to BOOLEAN.
-- All timestamps are UTC, 'YYYY-MM-DD HH:MM:SS'. Farm-level rows are stamped at period END.

CREATE TABLE IF NOT EXISTS sites (
    site_id       TEXT PRIMARY KEY,
    latitude      REAL NOT NULL,
    longitude     REAL NOT NULL,
    hub_height_m  REAL NOT NULL,
    n_turbines    INTEGER NOT NULL,
    capacity_kw   REAL NOT NULL
);

-- Raw 10-minute turbine telemetry with QC flags.
CREATE TABLE IF NOT EXISTS scada_10min (
    site_id       TEXT NOT NULL REFERENCES sites(site_id),
    turbine_id    INTEGER NOT NULL,
    ts            TEXT NOT NULL,
    power_kw      REAL,
    wind_speed    REAL,
    wind_dir      REAL,
    ambient_temp  REAL,
    status        TEXT,
    qc_ok         INTEGER NOT NULL,
    operating_ok  INTEGER NOT NULL,
    PRIMARY KEY (site_id, turbine_id, ts)
);

-- Farm aggregate at forecast resolution: the forecasting target.
CREATE TABLE IF NOT EXISTS farm_hourly (
    site_id       TEXT NOT NULL REFERENCES sites(site_id),
    ts            TEXT NOT NULL,
    power_kw      REAL,
    power_norm    REAL,
    ws_obs        REAL,
    temp_obs      REAL,
    availability  REAL,
    derated_frac  REAL,
    dq            INTEGER NOT NULL,
    PRIMARY KEY (site_id, ts)
);

-- One row per provider / run / valid hour. available_time drives as-of queries.
CREATE TABLE IF NOT EXISTS nwp_forecasts (
    site_id         TEXT NOT NULL REFERENCES sites(site_id),
    provider        TEXT NOT NULL,
    run_time        TEXT NOT NULL,
    valid_time      TEXT NOT NULL,
    available_time  TEXT NOT NULL,
    lead_h          REAL NOT NULL,
    ws_10m          REAL,
    ws_100m         REAL,
    wd_100m         REAL,
    temp_2m         REAL,
    pressure_hpa    REAL,
    PRIMARY KEY (site_id, provider, run_time, valid_time)
);
CREATE INDEX IF NOT EXISTS ix_nwp_valid ON nwp_forecasts (site_id, valid_time, provider, available_time);

-- Power forecasts in long form. kind = 'point' or a quantile label such as 'q05'.
CREATE TABLE IF NOT EXISTS forecasts (
    site_id        TEXT NOT NULL REFERENCES sites(site_id),
    model          TEXT NOT NULL,
    kind           TEXT NOT NULL,
    issue_time     TEXT NOT NULL,
    valid_time     TEXT NOT NULL,
    lead_h         REAL NOT NULL,
    value          REAL,
    model_version  TEXT,
    PRIMARY KEY (site_id, model, kind, issue_time, valid_time)
);
CREATE INDEX IF NOT EXISTS ix_fc_valid ON forecasts (site_id, valid_time, model, kind);

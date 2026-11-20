# windcast: utility-scale wind power forecasting

[![ci](https://github.com/dljayasekera-pixel/wind-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/dljayasekera-pixel/wind-forecast/actions/workflows/ci.yml)

Probabilistic power forecasts for a wind farm from 10 minutes to 7 days ahead, built from turbine SCADA
and several NWP providers, with the provider benchmarking, backtesting, root-cause analysis and
production plumbing that an operational forecast needs.

**Site:** Kelmarsh wind farm, UK. Six Senvion MM92 turbines, 12.3 MW.
**Data:** all public.

| Source | What | Licence / access |
|---|---|---|
| [Kelmarsh wind farm data](https://zenodo.org/records/16807551) (Cubico, Zenodo) | 10-minute SCADA and event logs per turbine, 2016 to end 2024 | CC-BY-4.0 |
| [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | Archived forecasts from GFS, ECMWF IFS and ICON, by lead day, from January 2024 | Free for non-commercial use |
| Built-in generator | Synthetic farm and providers with planted faults, same schema | For CI and offline work |

## Quick start

```bash
make install                                   # pinned versions from requirements.lock
make test                                      # 64 tests, about 20 s
windcast run-all --config configs/synthetic.yaml   # offline, about 2 minutes on 2 cores
make app                                       # Streamlit dashboard over the backtest
```

### Windows (PowerShell)

`make` is not available in PowerShell, so `tasks.ps1` provides the same steps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tasks.ps1 install
.\tasks.ps1 test
.\tasks.ps1 run                                       # synthetic data
.\tasks.ps1 run -Config configs/kelmarsh.yaml         # real data
.\tasks.ps1 app -Artifacts artifacts/kelmarsh
```

Real data (needs internet; downloads the 2024 SCADA archive, a few hundred MB, from Zenodo and calls the Open-Meteo API):

```bash
windcast run-all --config configs/kelmarsh.yaml
make app ARTIFACTS=artifacts/kelmarsh
```

Intra-hour product (10-minute resolution, 10 to 60 minutes ahead): `--config configs/intrahour.yaml`.

Individual steps: `windcast ingest | prepare | backtest | train | predict | monitor | load-sql`.

## Results on real data

Kelmarsh wind farm, 2024: real turbine SCADA and real archived forecasts from three NWP models.
Rolling-origin backtest with 7 monthly folds, testing 30 May to 26 Dec 2024 (each fold trains on
everything before it). 23,252 forecasts on hours when the farm was unconstrained, meaning at least
95% availability and at most 5% derated. Errors are fractions of installed capacity (12.3 MW), so
0.099 means 9.9%.

| Model | nMAE | nRMSE | Bias | RMSE skill vs persistence |
|---|---|---|---|---|
| Climatology (hour of day) | 0.226 | 0.275 | +0.036 | 0.03 |
| Persistence | 0.198 | 0.283 | -0.018 | 0 |
| ARIMA(2,0,1), leads up to 24 h only | 0.135 | 0.187 | -0.005 | 0.12 |
| XGBoost | 0.101 | 0.140 | +0.001 | 0.51 |
| Blended NWP through power curve | 0.099 | 0.145 | -0.031 | 0.49 |
| Stacked ensemble | 0.099 | 0.144 | -0.020 | 0.49 |
| LightGBM median | 0.098 | 0.144 | -0.024 | 0.49 |

nMAE by lead time:

| Model | 0-6 h | 6-24 h | 24-48 h | 48-96 h | 96 h+ |
|---|---|---|---|---|---|
| Persistence | 0.101 | 0.190 | 0.230 | 0.254 | 0.267 |
| Blended NWP through power curve | 0.078 | 0.077 | 0.088 | 0.113 | 0.166 |
| LightGBM median | 0.075 | 0.076 | 0.086 | 0.112 | 0.169 |
| Stacked ensemble | **0.072** | 0.077 | 0.087 | 0.115 | 0.174 |

Probabilistic: CRPS 0.068, a 52% improvement on climatology. The 5-95% interval covers 87.6% of
outcomes against a nominal 90%, so the intervals are slightly too narrow.

What the results show:

- **Machine learning earns its keep in the first six hours.** That is where recent SCADA output adds
  information the weather forecast lacks; the stacked ensemble cuts error by 28% against persistence
  and 7% against the physical model. From 6 hours on, the boosted models and the physical chain
  (bias-corrected, blended NWP through a fitted power curve) are within a few tenths of a percent of
  each other. With one site and under a year of training data, the physical chain is a strong
  baseline that the trees can only match, not beat.
- **Ensembling does not help at long range.** Beyond 48 hours the stacked ensemble is slightly worse
  than its best member. Its weights are fitted on a short hold-out and do not transfer well when
  skill is low. A per-lead choice of model would do better.
- **Long-range forecasts under-forecast power, and the loss function explains why.** At 96 h+ the
  physical chain and the LightGBM median both run about 7.5% of capacity low, while XGBoost is
  unbiased. As weather skill fades, bias correction pulls wind forecasts towards the mean. Because
  the power curve is convex below rated speed, the power of an average wind is less than the average
  power. The LightGBM median is lower than the mean by design, because power is skewed. XGBoost is
  trained on power with squared error, so it targets the mean directly.
- **Statistical baselines are weak for wind.** ARIMA barely beats persistence on MAE at 0-6 hours.
  It is in the table as a reference point.

**A data-quality bug found on the real data.** The first real run scored only 6% of hours, and they
were mostly calm (climatology over-forecast them by 24% of capacity). Three causes: status-log
categories the code did not recognise, including "out of environmental specification" (wind out of
range, which is weather), were treated as outages; an hour was rejected if any one of its 36
turbine samples was flagged, which removes windy hours most; and single noisy samples were flagged as
curtailment. After the fix, 72% of hours count as clean, and climatology's bias on them fell from
+0.24 to +0.04, so the clean set is close to the full record. `scripts/diagnose_data.py` checks this. The synthetic data could not reveal the
problem, because its fault flags are clean by construction.

For an offline check without downloads, `configs/synthetic.yaml` runs the same pipeline on generated
data with planted faults. Its accuracy says the code works, not how well it forecasts.

## How it works

```
 Zenodo (SCADA, logs)        Open-Meteo / NWP vendors
          |                            |
   connectors/kelmarsh        connectors/openmeteo        <- one class per feed, common schema
          |                            |
          v                            v
   scada/qc  -> flags        features/nwp -> as-of selection (available_time <= issue_time)
   scada/aggregate -> farm series stamped at period END
          \                            /
           +---- features/build ------+        one row per (issue time, lead)
                        |
        providers/stack (fitted per fold)      fallback -> bias correction -> blend -> power curve
                        |
        models/bundle                          persistence, climatology, ARIMA | LightGBM quantiles,
                        |                      XGBoost | calibration | stacked ensemble
        backtest/engine                        rolling origin, embargo, metrics, DM tests
                        |
   rca/attribution   monitoring/health   registry   storage (SQL)   app (Streamlit)
```

### Design decisions

**No look-ahead, by construction.** Every NWP row carries `available_time`. Features use the newest
run with `available_time <= issue_time`, so a run initialised at 12:00 but published at 17:00 is
invisible to a forecast issued at 14:00. The farm series is stamped at period end, so the value at
`t` is known at `t`. Training rows keep only labels whose valid time precedes the test block, which
is an embargo as long as the longest lead. `tests/test_features.py` corrupts all data after a cutoff
and asserts that no earlier feature changes.

**Everything learned from data is fitted inside the fold.** Bias correction, blending weights, the
power curve, calibration and stacking weights are refitted on each training window. The same
`fit_bundle` / `predict_bundle` code runs in the backtest, in training and at inference.

**SCADA quality control uses the event log and statistics.** Logged events give outage and derate
states. Because logs are incomplete, samples far below a robust reference power curve are flagged as
under-performance (unlogged curtailment, icing). Frozen anemometers, impossible values and gaps are
flagged separately. Models train on unconstrained hours only.

**Weather-to-power conversion.** NWP wind at 10 m and 100 m gives a shear exponent, which
extrapolates wind to hub height. Air density from NWP pressure and temperature converts it to an
equivalent wind speed at standard density (IEC 61400-12-1), which goes through a binned, monotone
farm power curve.

**Provider handling.** Each provider is bias-corrected against on-site wind (MOS regression per lead
bucket, or quantile mapping). Blending weights come from non-negative least squares per lead bucket,
which accounts for correlated errors. A provider that is missing or older than `max_age_h` drops
out and the remaining weights renormalise (tier 1). With no NWP at all the models run on SCADA lags
and climatology (tier 2). The tier is stored with every forecast.

**Probabilistic output.** One LightGBM model per quantile on the pinball loss, pooled over leads with
lead time as a feature. Quantiles are sorted to remove crossing, then shifted by a conformal-style
calibrator fitted on a time-ordered hold-out. Raw boosted quantiles tend to be too narrow in the
tails; with calibration the 5-95% interval covers 87.6% on Kelmarsh against a nominal 90%.

**Root-cause analysis.** Each error splits exactly into three terms:

```
actual - forecast = (actual - p_obs)   plant:   operations if turbines were down, else conversion
                  + (p_obs  - p_nwp)   weather: what the NWP wind error cost
                  + (p_nwp  - forecast) model:  what the model did with the weather it was given
```

`p_obs` is power from the measured wind through the power curve; `p_nwp` the same from forecast
wind. Large misses are labelled by dominant term, or as data quality or provider outage when they
cannot be decomposed. On Kelmarsh, of the misses above 25% of capacity, 89% are weather, 7% model,
4% data quality and under 1% operations. Weather causes 85% of all absolute error.

## Provider benchmarking

`artifacts/<run>/backtest/tables/` holds two scorecards, both out of sample.

*Delivery* (`provider_delivery.csv`): run coverage, mean and 95th-percentile publication latency,
longest gap, maximum horizon and per-variable availability.

*Accuracy* (`provider_accuracy.csv`, `provider_impact.csv`): wind-speed bias, MAE and RMSE before
and after correction by lead bucket, and the power nMAE that would result from using each provider
alone, relative to the blend. Kelmarsh, 24-48 h leads:

| Provider | Wind bias raw (m/s) | Wind RMSE raw | Wind RMSE corrected | Wind correlation | Power nMAE alone |
|---|---|---|---|---|---|
| ECMWF IFS | +1.05 | 1.79 | 1.28 | 0.87 | 0.094 |
| GFS | +1.05 | 2.05 | 1.34 | 0.85 | 0.103 |
| ICON | +1.36 | 2.20 | 1.35 | 0.85 | 0.106 |
| Blend | +0.00 | 1.18 | 1.18 | 0.89 | 0.088 |

- Every provider reads 1.0-1.6 m/s above the turbine anemometers at every lead. That is mostly the
  reference, not the models: nacelle anemometers sit behind the rotor and read low. Bias correction
  removes it and cuts wind RMSE by about a third.
- ECMWF is the best single provider up to 96 hours. Beyond that, the three are within half a percent
  of capacity of each other (ICON has about a third fewer forecasts in that bucket).
- The blend beats the best single provider at every lead time, by 0.4 to 0.8 percent of capacity.
- Skill fades with horizon: correlation with measured wind falls from 0.92 at 0-6 h to 0.59 beyond
  96 h.

A commercial vendor forecast plugs in the same way: emit the canonical NWP frame (or, for a vendor
*power* forecast, add it as a column and list it as a stack member) and it appears in every table.

## Production workflow

| Concern | Implementation |
|---|---|
| Orchestration | `infra/statemachine.asl.json`: Lambda ingest, ECS prepare and predict, Lambda monitor, SageMaker retrain branch, SNS on failure, retries with backoff |
| Compute | One container image (`Dockerfile`) for every step |
| Storage | S3 raw / curated / artifacts zones, versioned; Parquet partitioned by provider and date |
| Monitoring | `windcast monitor`: trailing nMAE, bias, skill, interval coverage, fallback share, feed freshness; published to CloudWatch; alarms in `infra/main.tf` |
| Alerts | Feed problems are CRITICAL; accuracy drift is WARN and sets `retrain_recommended` |
| Degradation | Stale SCADA or missing NWP lowers quality and raises the fallback tier; it does not fail the run |
| Registry | Immutable versions with config, git SHA, library versions and backtest metrics; `latest` pointer for promotion and rollback; `--no-promote` for shadow runs |
| Reproducibility | Seeds, deterministic LightGBM, pinned `requirements.lock`, config hash in every version id; a test asserts two backtests give identical results |
| CI | `.github/workflows/ci.yml`: ruff, unit tests, end-to-end tests, image build |
| Branching | Trunk-based: short-lived feature branches, pull request with CI required, tagged releases build the deployable image |

The Terraform and state machine are **sketches**: they show the service mapping and have not been
applied to an AWS account.

## SQL layer

`src/windcast/sql/schema.sql` defines `sites`, `scada_10min`, `farm_hourly`, `nwp_forecasts` and
`forecasts`. `queries.sql` holds nine named queries, run with `storage.query(conn, name, ...)`:

| Query | Technique |
|---|---|
| `provider_error_by_lead` | join forecasts to observations, bucketed aggregate |
| `latest_forecast_as_of` | `ROW_NUMBER()` window for as-of selection |
| `provider_latency_and_gaps` | `LAG()` over runs |
| `daily_energy` | time-bucketed aggregate |
| `ramp_events` | `LAG()` over the farm series |
| `forecast_skill_by_lead` | CTEs and a self-join against persistence |
| `rolling_7d_mae` | window frame `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` |
| `provider_disagreement` | self-join across providers |
| `interval_coverage` | pivot by conditional aggregation |

SQLite is used so the repo runs without a server; the header of each file lists what changes for
PostgreSQL. A test checks that the SQL skill query and the Python metrics agree.

## Requirement coverage

| Requirement | Evidence |
|---|---|
| pandas, NumPy, scikit-learn stack, statsmodels, LightGBM, XGBoost | `features/`, `models/gbm.py`, `models/baselines.py` (SARIMAX) |
| Forecasting and time-series models, developed and deployed | `models/bundle.py`, `pipeline.py` (`train`, `predict`), `registry.py` |
| Feature engineering | `features/build.py`, `features/nwp.py`, `providers/stack.py`: shear, density, power-curve transform, ramps, lags, rolling statistics, provider spread, current NWP error, Fourier terms |
| Regression and time-series methods | MOS regression, ARIMA, persistence, climatology |
| Gradient boosting | LightGBM quantile models, XGBoost point model |
| Ensemble forecasting | Provider blend (`providers/blend.py`), stacked ensemble (`models/ensemble.py`) |
| Probabilistic and quantile forecasting | Pinball-loss models, calibration, CRPS, reliability and coverage tables |
| Validation and backtesting | `backtest/`: rolling origin with embargo; metrics by lead, season, wind regime and fold; Diebold-Mariano tests |
| Weather APIs, NWP, SCADA, databases, external feeds | `connectors/`, `storage.py`, `infra/lambda_ingest.py` |
| Weather-provider evaluation and its impact on accuracy | `providers/scorecard.py` |
| SQL and data analysis | `sql/`, `storage.py` |
| R&D to production | Same code path for backtest and inference; CLI steps; registry; fallback; monitoring |
| AWS (S3, Lambda, ECS, Step Functions, CloudWatch, SageMaker, EC2) | `infra/`, `monitoring/cloudwatch.py`, `registry.upload_to_s3`; EC2 discussed in `infra/main.tf` |
| Git, testing, reproducibility | 64 tests, CI workflow, lockfile, seeds, config hashing |
| Utility-scale wind, turbine SCADA, power curves | `scada/` |
| Multiple NWP providers | GFS, ECMWF IFS, ICON through one interface |
| Benchmarking against vendors | Provider impact table; vendor power forecasts slot in as stack members |
| Intra-hour, day-ahead, multi-day horizons | `configs/intrahour.yaml` (10-60 min), default config (1 h to 7 days) |
| Streamlit and Plotly | `app/streamlit_app.py` |
| Root-cause analysis of misses | `rca/attribution.py` |
| Logging, alerts, error handling, recovery | JSON logs, `monitoring/health.py`, retries in connectors and state machine, fallback tiers |

## Known limitations

- **Clean-hour filter.** Accuracy is scored on unconstrained hours (72% of the record). Hours with
  outages or curtailment are excluded, as is standard for forecasting unconstrained output, but the
  status log may not record every curtailment.
- **NWP freshness with Open-Meteo.** The Previous Runs API returns forecasts by lead *day*, so the
  newest usable forecast was made at least 24 hours before the valid time. Leads under 24 h therefore
  lean on SCADA. A run-resolved source (NOAA GFS/HRRR GRIB on S3, ECMWF open data, a vendor feed)
  removes this and needs only a new `NWPConnector`.
- **Nacelle wind as truth.** Provider wind is scored against nacelle anemometers, which sit behind
  the rotor. A met mast, lidar or reanalysis (the Zenodo record includes ERA5 for the site) would be
  a cleaner reference.
- **One year of data.** 2024 is the only year both public sources cover, so there is no
  year-over-year validation and day-of-year features are switched off.
- **Availability is not forecast.** Models predict unconstrained output. A production system would
  multiply by scheduled availability from the maintenance plan.
- **The monitoring baseline is not seasonal.** The nMAE alert compares a trailing window with the
  whole-backtest average, so it fires in windy months. Compare against the same season in production.
- **Infrastructure is a sketch.** Not applied; IAM and networking are variables.

## Repository layout

```
src/windcast/
  connectors/   base (schemas), kelmarsh, openmeteo, synthetic
  scada/        qc, power_curve, aggregate
  features/     nwp (as-of selection), build (modelling frame)
  providers/    bias_correction, blend, fallback, stack, scorecard
  models/       baselines, gbm, ensemble, bundle
  backtest/     splits, metrics, engine
  rca/          attribution
  monitoring/   health, cloudwatch
  sql/          schema.sql, queries.sql
  storage.py  registry.py  pipeline.py  cli.py  config.py  utils.py
app/            streamlit_app.py
infra/          main.tf, statemachine.asl.json, lambda_ingest.py
configs/        synthetic.yaml, kelmarsh.yaml, intrahour.yaml
tests/          64 tests; end-to-end ones are marked slow
```

## Data citation

Plumley, C. *Kelmarsh wind farm data*. Zenodo. https://zenodo.org/records/16807551 (CC-BY-4.0).
Weather data by [Open-Meteo.com](https://open-meteo.com/).

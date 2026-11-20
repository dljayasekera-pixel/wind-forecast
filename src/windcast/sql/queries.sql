-- Analytical queries. Each block starts with '-- name: <id>' and is loaded by windcast.storage.
-- SQLite dialect. PostgreSQL equivalents: julianday() differences -> EXTRACT(EPOCH FROM a - b) / 3600,
-- date(ts) -> date_trunc('day', ts), datetime(x, '+48 hours') -> x + INTERVAL '48 hours'.

-- name: provider_error_by_lead
-- Wind-speed accuracy of each provider against on-site observations, bucketed by lead time.
SELECT n.provider,
       CASE WHEN n.lead_h <= 6 THEN '0-6h' WHEN n.lead_h <= 24 THEN '6-24h'
            WHEN n.lead_h <= 48 THEN '24-48h' WHEN n.lead_h <= 96 THEN '48-96h' ELSE '96h+' END AS lead_bucket,
       COUNT(*)                                                     AS n,
       ROUND(AVG(n.ws_100m - f.ws_obs), 3)                          AS ws_bias,
       ROUND(AVG(ABS(n.ws_100m - f.ws_obs)), 3)                     AS ws_mae,
       ROUND(SQRT(AVG((n.ws_100m - f.ws_obs) * (n.ws_100m - f.ws_obs))), 3) AS ws_rmse
FROM nwp_forecasts n
JOIN farm_hourly f ON f.site_id = n.site_id AND f.ts = n.valid_time
WHERE n.site_id = :site AND f.dq = 0 AND f.ws_obs IS NOT NULL
GROUP BY n.provider, lead_bucket
ORDER BY MIN(n.lead_h), n.provider;

-- name: latest_forecast_as_of
-- What each provider was saying about the next 48 hours, using only runs available at :as_of.
WITH ranked AS (
    SELECT n.*,
           ROW_NUMBER() OVER (PARTITION BY provider, valid_time ORDER BY available_time DESC) AS rn
    FROM nwp_forecasts n
    WHERE site_id = :site AND available_time <= :as_of
      AND valid_time > :as_of AND valid_time <= datetime(:as_of, '+48 hours')
)
SELECT provider, valid_time, run_time,
       ROUND((julianday(:as_of) - julianday(run_time)) * 24, 1) AS run_age_h,
       ROUND(ws_100m, 2) AS ws_100m
FROM ranked
WHERE rn = 1
ORDER BY valid_time, provider;

-- name: provider_latency_and_gaps
-- Delivery reliability: publication delay and the longest gap between consecutive runs.
WITH runs AS (
    SELECT DISTINCT provider, run_time, available_time FROM nwp_forecasts WHERE site_id = :site
), gaps AS (
    SELECT provider, run_time,
           (julianday(available_time) - julianday(run_time)) * 24 AS latency_h,
           (julianday(run_time) - julianday(LAG(run_time) OVER (PARTITION BY provider ORDER BY run_time))) * 24 AS gap_h
    FROM runs
)
SELECT provider,
       COUNT(*)                  AS n_runs,
       ROUND(AVG(latency_h), 2)  AS latency_mean_h,
       ROUND(MAX(latency_h), 2)  AS latency_max_h,
       ROUND(MAX(gap_h), 1)      AS longest_gap_h
FROM gaps
GROUP BY provider
ORDER BY provider;

-- name: daily_energy
-- Daily energy, capacity factor and availability (time-bucketed aggregate).
SELECT date(ts, '-1 minute')               AS day,
       ROUND(SUM(power_kw) / 1000.0, 1)    AS energy_mwh,
       ROUND(AVG(power_norm), 3)           AS capacity_factor,
       ROUND(AVG(availability), 3)         AS availability,
       SUM(dq)                             AS bad_hours
FROM farm_hourly
WHERE site_id = :site
GROUP BY day
ORDER BY day;

-- name: ramp_events
-- Hours where farm output moved by more than :threshold of capacity within three hours.
WITH d AS (
    SELECT ts, power_norm,
           power_norm - LAG(power_norm, 3) OVER (ORDER BY ts) AS delta_3h
    FROM farm_hourly
    WHERE site_id = :site AND dq = 0
)
SELECT ts, ROUND(power_norm, 3) AS power_norm, ROUND(delta_3h, 3) AS delta_3h,
       CASE WHEN delta_3h > 0 THEN 'up' ELSE 'down' END AS direction
FROM d
WHERE ABS(delta_3h) >= :threshold
ORDER BY ABS(delta_3h) DESC
LIMIT 50;

-- name: forecast_skill_by_lead
-- nMAE, nRMSE, bias and RMSE skill versus persistence per model and lead, on unconstrained hours.
WITH err AS (
    SELECT fc.model, fc.lead_h, fc.value - a.power_norm AS e
    FROM forecasts fc
    JOIN farm_hourly a ON a.site_id = fc.site_id AND a.ts = fc.valid_time
    WHERE fc.site_id = :site AND fc.kind = 'point'
      AND a.dq = 0 AND a.availability >= 0.95 AND a.derated_frac <= 0.05
), m AS (
    SELECT model, lead_h, COUNT(*) AS n, AVG(ABS(e)) AS mae, SQRT(AVG(e * e)) AS rmse, AVG(e) AS bias
    FROM err GROUP BY model, lead_h
)
SELECT m.model, m.lead_h, m.n, ROUND(m.mae, 4) AS nmae, ROUND(m.rmse, 4) AS nrmse, ROUND(m.bias, 4) AS bias,
       ROUND(1 - m.rmse / p.rmse, 4) AS skill_vs_persistence
FROM m
JOIN m AS p ON p.lead_h = m.lead_h AND p.model = 'persistence'
ORDER BY m.lead_h, m.model;

-- name: rolling_7d_mae
-- Day-ahead (24 h lead) error per model as a trailing seven-day average: the drift chart.
WITH daily AS (
    SELECT fc.model, date(fc.valid_time, '-1 minute') AS day, AVG(ABS(fc.value - a.power_norm)) AS mae
    FROM forecasts fc
    JOIN farm_hourly a ON a.site_id = fc.site_id AND a.ts = fc.valid_time
    WHERE fc.site_id = :site AND fc.kind = 'point' AND fc.lead_h = 24 AND a.dq = 0
    GROUP BY fc.model, day
)
SELECT model, day, ROUND(mae, 4) AS mae,
       ROUND(AVG(mae) OVER (PARTITION BY model ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 4) AS mae_7d
FROM daily
ORDER BY model, day;

-- name: provider_disagreement
-- Largest disagreements between two providers for the same run and valid hour, with what happened.
SELECT a.valid_time, a.lead_h,
       ROUND(a.ws_100m, 2) AS ws_a, ROUND(b.ws_100m, 2) AS ws_b,
       ROUND(ABS(a.ws_100m - b.ws_100m), 2) AS spread,
       ROUND(f.ws_obs, 2) AS ws_obs
FROM nwp_forecasts a
JOIN nwp_forecasts b
  ON b.site_id = a.site_id AND b.valid_time = a.valid_time AND b.run_time = a.run_time AND b.provider = :provider_b
LEFT JOIN farm_hourly f ON f.site_id = a.site_id AND f.ts = a.valid_time
WHERE a.site_id = :site AND a.provider = :provider_a AND a.lead_h = :lead_h
ORDER BY spread DESC
LIMIT 20;

-- name: interval_coverage
-- Empirical coverage and width of the 5-95% interval by lead (pivot via conditional aggregation).
WITH q AS (
    SELECT site_id, issue_time, valid_time, lead_h,
           MAX(CASE WHEN kind = 'q05' THEN value END) AS lo,
           MAX(CASE WHEN kind = 'q95' THEN value END) AS hi
    FROM forecasts
    WHERE site_id = :site AND model = 'lgbm'
    GROUP BY site_id, issue_time, valid_time, lead_h
)
SELECT q.lead_h, COUNT(*) AS n,
       ROUND(AVG(CASE WHEN a.power_norm BETWEEN q.lo AND q.hi THEN 1.0 ELSE 0.0 END), 3) AS coverage,
       ROUND(AVG(q.hi - q.lo), 3) AS mean_width
FROM q
JOIN farm_hourly a ON a.site_id = q.site_id AND a.ts = q.valid_time
WHERE a.dq = 0 AND q.lo IS NOT NULL AND q.hi IS NOT NULL
GROUP BY q.lead_h
ORDER BY q.lead_h;

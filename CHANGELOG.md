# Changelog

## 0.2.0 (2026-11-20)

- The Open-Meteo connector can fetch any hourly variable set (`variables=`), not only wind,
  and keeps cache files apart per location (`cache_tag=`). Used by the
  [load-forecast](https://github.com/dljayasekera-pixel/load-forecast) project for zone temperature
  and humidity forecasts. Wind behaviour and existing cache file names are unchanged.

## 0.1.0 (2026-11-20)

- First release: SCADA and multi-provider NWP wind power forecasting, backtested on Kelmarsh 2024.

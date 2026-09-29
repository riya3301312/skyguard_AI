# SkyGuard AI - Weather-vs-Sensor Analysis

_Generated from `artifacts/` (2026-09-29T07:12:36+00:00)._

## Objective

Distinguish genuine regional weather (neighbours change together, T/P/RH physically consistent) from isolated sensor/data faults - including a faulty station *during* a regional event - without hiding true faults.

## Method

1. **Regional evidence score** (0-1): for every strongly deviating parameter, the share of valid neighbours showing a similar change within +/-1 observation interval (temporal alignment), weighted by neighbour coverage (3+ neighbours = full weight). Neighbours that deviate from the rest of the neighbourhood (e.g. a drifting neighbour) are excluded first (robust median).
2. **Station exception**: while the regional score is high, a parameter that deviates strongly but is *not* shared, an inconsistent temperature/humidity relationship, or a deterministic-QC violation marks the station as an exception.
3. **Weather status** per observation: NORMAL, REGIONAL_WEATHER, REGIONAL_WEATHER_WITH_STATION_EXCEPTION, SENSOR_ANOMALY, UNCERTAIN.
4. **Decision rules** (validated): DRIFT/STEP alerts require sustained, one-signed, station-specific deviation; an alert whose evidence is regional and has no station exception is not raised as a sensor fault (logged as INFO). A station exception or deterministic-QC fault is never suppressed.
5. Parameters selected on validation: `{"confirm_persistent": true, "k6": 1.5, "c3": 0.5, "max_shared": 0.5, "regional_suppress": true, "regional_min": 0.5}`, threshold 0.23.

## Validation findings that motivated the change (validation split only)

* Before: 25 of 108 regional-event station instances alerted (23.1%); alerted patterns {'DRIFT': 19, 'STEP_OFFSET': 14, 'MULTIVARIATE_INCONSISTENCY': 4, 'UNKNOWN_SUSPECT': 2}; alerted parameters {'temperature_c': 28, 'relative_humidity_pct': 9, 'pressure_hpa': 2}; alert rate by neighbour count {'1.0': 0.167, '2.0': 0.048, '3.0': 0.067}.
* Before: validation false-positive patterns {'DRIFT': 259, 'STEP_OFFSET': 208, 'COMPOSITE_FAULT': 74, 'MULTIVARIATE_INCONSISTENCY': 56, 'FLATLINE': 26, 'UNKNOWN_SUSPECT': 10, 'NEGATIVE_SPIKE': 9, 'SPIKE': 2, 'SPATIAL_INCONSISTENCY': 1}.

* After: 6 of 108 instances alerted (5.6%); false-positive patterns {'STEP_OFFSET': 194, 'DRIFT': 127, 'COMPOSITE_FAULT': 117, 'MULTIVARIATE_INCONSISTENCY': 94, 'FLATLINE': 71, 'NEGATIVE_SPIKE': 11, 'UNKNOWN_SUSPECT': 9, 'SPATIAL_INCONSISTENCY': 9, 'SPIKE': 7}.

## Final-test results (v2, paired on identical data)

| Variant | Precision | Recall | F1 | FA / station-day | Event recall | Regional events alerted |
|---|---:|---:|---:|---:|---:|---:|
| Without decision rules | 0.701 | 0.592 | 0.642 | 0.464 | 0.950 | 0.433 |
| With decision rules (deployed) | 0.820 | 0.457 | 0.587 | 0.184 | 0.901 | 0.134 |

* Regional-event station instances: 157; with any alert: 21; recognised as REGIONAL_WEATHER at least once: 132.
* Regional-event rows: 1196; rows given a regional status: 622 (recall of the regional status on injected events 52.0%).
* Rows with REGIONAL_WEATHER status: 3180, of which inside injected events 622. Precision of the REGIONAL_WEATHER status against injected events is a lower bound: genuine regional weather in the real observations is unlabelled.
* Status counts: {'NORMAL': 87061, 'SENSOR_ANOMALY': 4102, 'REGIONAL_WEATHER': 3180, 'REGIONAL_WEATHER_WITH_STATION_EXCEPTION': 24}.
* Station-exception rows: 24 (10 are injected anomalies).

Station-exception examples (v2 final test):

| Station | Time | Pattern | Parameter | Probability | Exception reason | Ground truth |
|---|---:|---:|---:|---:|---:|---:|
| VOBG | 2024-09-02 12:00:00+00:00 | SPATIAL_INCONSISTENCY | temperature_c | 0.981 | isolated deviation in temperature_c | SPATIAL_INCONSISTENCY |
| VOBG | 2024-10-05 08:00:00+00:00 | SPIKE | pressure_hpa | 0.358 | temperature/humidity relationship inconsistent | NORMAL |
| VOBG | 2024-10-08 16:00:00+00:00 | STEP_OFFSET | temperature_c | 0.993 | isolated deviation in temperature_c | DRIFT |
| VOCB | 2024-09-23 02:00:00+00:00 | STEP_OFFSET | temperature_c | 0.569 | isolated deviation in temperature_c | NORMAL |
| VOCB | 2024-12-12 07:00:00+00:00 | MULTIVARIATE_INCONSISTENCY | relative_humidity_pct | 0.569 | temperature/humidity relationship inconsistent | STEP_OFFSET |

## SIH demo evidence

* **F_regional_weather_event** (PASS): regional weather -> regional evidence and no sensor-fault alert. Evidence: `{"cluster_observations": 28, "sensor_fault_alerts": 0, "alerts": [], "weather_status_counts": {"NORMAL": 22, "REGIONAL_WEATHER": 6}, "regional_events_logged_as_info": 3, "max_probability": 0.5027932960893855}`

* **F2_regional_weather_with_station_exception** (PASS): regional weather + isolated sensor fault -> isolated station still alerted. Evidence: `{"station": "VOTR", "rows": [{"hour": 22, "is_anomaly": false, "fault_pattern": "NORMAL", "parameter": "", "anomaly_probability": 0.1567622950819672, "weather_status": "NORMAL"}, {"hour": 23, "is_anomaly": true, "fault_pattern": "COMPOSITE_FAULT", "parameter": "pressure_hpa", "anomaly_probability": 0.2611683848797251, "weather_status": "REGIONAL_WEATHER_WITH_STATION_EXCEPTION"}]}`

* **G_isolated_station_event** (PASS): isolated change -> spatial-inconsistency sensor alert, while the regional event raises none. Evidence: `{"alerts": [{"hour": 29, "fault_pattern": "SPATIAL_INCONSISTENCY", "weather_status": "SENSOR_ANOMALY", "anomaly_probability": 0.41304347826086957, "severity": "LOW"}, {"hour": 30, "fault_pattern": "SPATIAL_INCONSISTENCY", "weather_status": "SENSOR_ANOMALY", "anomaly_probability": 0.4383561643835616, "severity": "LOW"}, {"hour": 32, "fault_pattern": "SPATIAL_INCONSISTENCY", "weather_status": "SENSOR_ANOMALY", "anomaly_probability": 0.2611683848797251, "severity": "LOW"}], "sensor_fault_alerts_during_regional_event": 0, "raw_detector_max_probability_isolated": 0.4383561643835616, "raw_detector_m`

## Explanation evidence

Every alert explanation carries a `weather_vs_sensor` block (status, regional score, stations supporting / considered, temporal alignment, spatial consistency, station exception + reason) and a plain sentence such as "Regional weather present (4 of 4 neighbours share the change), but this station shows an exception: isolated deviation in pressure_hpa. Flagged separately from the regional event." The dashboard shows it in the alert detail.

## Limitations

* Injected regional events are synthetic overlays on real data; genuine regional weather in the source is unlabelled, so REGIONAL_WEATHER precision is a lower bound.
* Stations with fewer than two neighbours cannot use regional evidence; their alerts rely on temporal and multivariate evidence.
* A station leading its neighbours by one interval is resolved when neighbour data arrive (provisional -> revised scoring in the live engine).
* 13.4% of v2 regional events still raised at least one alert.

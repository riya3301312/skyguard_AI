# SkyGuard AI — Project Status & Progress Report

**Status date:** 2026-09-29 · **Branch:** `feat/skyguard-sih-prototype` · **Latest commit:** `8ad71df` · **Working tree:** clean (not pushed)

**Overall status: production-oriented SIH prototype — VERIFIED** (Docker: environment-unverified · ESP32: hardware-unverified)

SkyGuard AI is a real-time, explainable anomaly-detection layer for Automatic Weather Station **temperature, pressure and
relative humidity**, built for remote Indian stations. It checks each observation, gives each alert a calibrated probability,
a fault pattern, a severity and a plain-language explanation, and keeps a sensor-health score. It tells regional weather
apart from single-station sensor faults, suggests corrected values without ever overwriting raw data, and works with stations
that go offline and upload later. A live dashboard and a tiny C edge-check library for ESP32-class loggers are included.

All figures below come from files in `artifacts/`. The detailed reports are in `docs/reports/`.

---

## 1. Status at a glance

| Area | Status | Evidence |
|---|---|---|
| Data pipeline (real Indian observations, provenance) | ✅ Done | `data/raw/iem_metar/PROVENANCE.json` |
| Anomaly injection (12 fault classes, seeded) | ✅ Done | `injection/injector.py`, unit tests |
| Leakage-safe splits + frozen final benchmark | ✅ Done | `data/benchmark/*/MANIFEST.json` (SHA-256) |
| Hybrid detector + 4 baselines + ablation | ✅ Done | `artifacts/final_evaluation/` |
| Weather-vs-sensor decision layer | ✅ Done | `ml/decision.py`, weather-vs-sensor report |
| Explanations, severity, health, correction, maintenance | ✅ Done | `ml/postprocess.py` |
| Real-time engine + offline buffering / reconnect | ✅ Done | scenarios E and I |
| FastAPI backend + security basics | ✅ Done | integration tests |
| React dashboard (live, no page refresh) | ✅ Verified in browser | deployment report |
| Edge C library | ✅ Cross-compiled + host-tested | `artifacts/edge/edge_metrics.json` |
| Test suite | ✅ 56 passed, 1 skipped (Docker) | `artifacts/regression/pytest_junit.xml` |
| SIH scenarios A–J (+F2) | ✅ **11 / 11 pass** | `artifacts/scenarios/scenario_results.json` |
| Docker build/run | ⚠️ **Not verified** (no Docker on this machine) | `artifacts/docker_verification/` |
| Physical ESP32 + energy measurement | ⚠️ Not done (no hardware) | — |

---

## 2. Progress timeline

| Phase | What happened | Commit |
|---|---|---|
| **Pass 1: build** | Starting from an empty repo, built the whole system: data, injector, benchmark (v1 frozen), models, API, dashboard, edge, tests and reports. | `7e19754` |
| | Result: 9 of 10 scenarios passed. **A (clean station)** failed with one false DRIFT alert. 41% of regional weather events were wrongly alerted. Docker was not verified. | |
| **Freeze v2** | Before changing any code, froze a new final test set (seed 9002) so the fixes could be judged on unseen data. | `c1bcbfc` |
| **Pass 2: hardening** | Added persistence-aware DRIFT/STEP confirmation, a regional-evidence score, station-exception detection and a robust neighbour median. Rules and thresholds were chosen on validation only; v2 was evaluated once. | `8ad71df` |
| | Result: **11 / 11 scenarios pass**. Regional false alerts fell 43.3% → 13.4% (paired, same data). | |

---

## 3. Key results (v2 frozen final test — evaluated once)

The test set has 94,367 hourly observations from 22 Indian stations, including 4 stations never seen in training (VOBG, VOTV,
VIJP, VOPB). Seed and injection profile are unseen, and the SHA-256 was verified at evaluation.

| Metric | SkyGuard Hybrid |
|---|---:|
| Precision | **0.820** |
| Recall (row-level) | 0.457 |
| F1 | 0.587 |
| PR-AUC / ROC-AUC | **0.688 / 0.903** |
| False alerts per station-day | **0.184** |
| Event-level recall | **90.1%** |
| Regional weather events wrongly alerted | **13.4%** |
| Fault-pattern macro-F1 | 0.562 |
| Calibration: Brier / ECE | 0.040 / 0.009 |
| Held-out stations F1 (vs seen stations) | 0.577 (vs 0.589) |

> Row-level recall counts individual anomalous observations that were flagged. Event-level recall counts whether an injected
> event was detected at least once. That is why one is 0.46 and the other 90%.

### Baseline comparison (same test data)

| Approach | F1 | PR-AUC |
|---|---:|---:|
| Fixed thresholds | 0.205 | 0.173 |
| Robust statistical (z-score) | 0.215 | 0.249 |
| Isolation Forest | 0.280 | 0.393 |
| Temporal autoencoder (MLP) | 0.242 | 0.273 |
| **SkyGuard Hybrid** | **0.587** | **0.688** |

### What the pass-2 decision rules changed (same model, same v2 data)

| | Without the new rules | **With the new rules (deployed)** |
|---|---:|---:|
| Regional events wrongly alerted | 43.3% | **13.4%** |
| False alerts per station-day | 0.464 | **0.184** |
| Precision | 0.701 | **0.820** |
| Event recall | 95.0% | 90.1% |
| Row-level recall | 0.592 | 0.457 |

Drift and step alerts now wait for sustained evidence. That costs some row-level recall and buys far fewer false alarms.

### Other measurements

- **Correction accuracy** (mean absolute error of the recommended value vs the true value): temperature 2.03 °C, pressure 1.43 hPa, humidity 8.25 %.
- **Real-time latency** (full engine, one observation at a time, this PC): p50 340 ms, p95 538 ms, p99 632 ms; ~2.9 observations/s. Batched uploads are much faster per observation.
- **Edge C library** (cross-compiled for the ESP32-C3 instruction set): 1,134 bytes flash, 28 B state per station, 96 B worst-case stack, ~250 ns per observation on the host PC (not on an ESP32).

---

## 4. SIH scenario matrix

| # | Scenario | Result |
|---|---|---|
| A | Clean station → no false alert | ✅ PASS |
| B | Temperature spike (55.2 °C) → alert, pattern, explanation, estimate | ✅ PASS |
| C | Flatline → FLATLINE pattern | ✅ PASS |
| D | Gradual pressure drift → DRIFT/STEP alerts, maintenance risk HIGH | ✅ PASS |
| E | Communication failure → missing-data alert | ✅ PASS |
| F | Regional weather event → no sensor-fault alert | ✅ PASS |
| F2 | Regional weather + one station's own fault → that station still flagged | ✅ PASS |
| G | Isolated station change → spatial-inconsistency sensor alert | ✅ PASS |
| H | Held-out stations → reasonable generalisation | ✅ PASS |
| I | Remote station offline → buffer → reconnect → sync, no data loss | ✅ PASS |
| J | Edge checks within resource budget | ✅ PASS |

---

## 5. Known limitations (open items)

1. **Docker not verified.** Docker isn't installed on this machine. Install Docker Desktop, then run `python scripts/verify_docker.py`.
2. **No physical ESP32 test** and no energy or battery measurement. The edge library is validated by cross-compilation and host execution only.
3. **Proxy data.** Observations are Indian aerodrome METAR reports from the Iowa Environmental Mesonet archive, **not** the IMD AWS network feed. Temperature resolution is 1 °C and pressure is QNH.
4. **Regional false alerts.** 13.4% on the unseen test, above the 10% target that was met on validation.
5. **Row-level recall is modest (0.457).** Drift and step alerts are deliberately delayed until confirmed.
6. **Latency** is a few hundred ms per live observation (pandas-based). That's fine for hourly data but not optimised.
7. **Prototype architecture.** SQLite in a single process; PostgreSQL and horizontal scaling are the documented next step.
8. **Scenario G criterion changed** in pass 2, from comparing raw probabilities to checking the final decision. This is disclosed in the reports.

---

## 6. Suggested next steps

1. Install Docker Desktop and run `python scripts/verify_docker.py` to close the Docker gate.
2. Flash the ESP32 sketch (`edge/firmware/esp32/`) on real hardware and measure runtime and power.
3. Obtain real IMD AWS data and evaluate it as a separate dataset with its own provenance.
4. Reduce the remaining regional false alerts, working on validation only, then freeze a new v3 test set.
5. Speed up per-observation feature computation if sub-100 ms latency is needed.

---

## 7. How to run

```bash
pip install -r requirements.txt
cd dashboard && npm ci && npm run build && cd ..
uvicorn backend.app:app --port 8000          # open http://localhost:8000, choose "sih_demo", Start replay
python scripts/run_scenarios.py --base-url http://127.0.0.1:8000
python -m pytest tests -q
```

The full reproduction pipeline (data → benchmark → train → evaluate → reports) is in `README.md`.

## 8. Where to find things

| What | Where |
|---|---|
| Final evaluation report | `docs/reports/SkyGuard_AI_Final_SIH_Evaluation_Report.md` |
| Weather-vs-sensor analysis | `docs/reports/SkyGuard_AI_Weather_vs_Sensor_Analysis.md` |
| Deployment verification | `docs/reports/SkyGuard_AI_Final_Deployment_Verification.md` |
| Metrics and plots (v2 / v1) | `artifacts/final_evaluation/`, `artifacts/final_evaluation_v1/` |
| Trained model + metadata | `artifacts/final_model/` |
| Scenario evidence | `artifacts/scenarios/scenario_results.json` |
| Regression evidence | `artifacts/regression/` |

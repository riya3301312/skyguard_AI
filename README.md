# SkyGuard AI

Real-time, explainable anomaly detection for **temperature, atmospheric pressure and relative humidity** from
Automatic Weather Stations, designed for remote and data-sparse Indian deployments.

It is a production-quality **SIH prototype** that takes AWS observations and turns them into verified alerts:

* **Checks:** deterministic QC, then temporal, multivariate and spatial evidence, feeding a calibrated anomaly probability.
* **Per alert:** an observable fault pattern, severity, explanation and sensor-health score.
* **Correction:** an optional correction *recommendation*. The raw value is always preserved.
* **Operations:** store-and-forward remote uplink, a live dashboard, and a tiny C edge-check library for ESP32-class loggers.

Measured results are in [`docs/SkyGuard_AI_Final_SIH_Evaluation_Report.md`](docs/SkyGuard_AI_Final_SIH_Evaluation_Report.md), generated
from `artifacts/`. The README deliberately repeats no numbers.

## Architecture

```text
AWS logger ─ edge C checks (edge/firmware) ─ StationClient store-and-forward buffer (edge/station_client.py)
      │  HTTPS + bearer token, batched, idempotent
      ▼
FastAPI  backend/app.py ── typed Pydantic payloads, size/rate limits, CORS
      ▼
StreamEngine  ml/stream.py   (same code as batch evaluation → training/serving parity, tested)
  deterministic QC (ml/qc.py, config/skyguard.toml)
  → temporal features (past-only, station time-of-day baseline)     ml/features.py
  → multivariate T/P/RH consistency (dew-point physics)
  → spatial features from configurable neighbours (optional)
  → MLP temporal autoencoder → gradient-boosted detector → isotonic calibration   ml/models.py
  → regional-weather-event rule → fault-pattern + parameter classifiers
  → severity · explanation · sensor health · correction · maintenance risk        ml/postprocess.py
      ▼
SQLite (raw values immutable via trigger)  ──  SSE /api/v1/stream  ──  React dashboard (dashboard/)
```

The model inputs are T, P and RH only. Latitude and longitude are used solely to choose neighbours. Nothing names a station
in the code; stations, neighbours (radius and cap), thresholds and windows all come from `config/skyguard.toml`.

## Repository layout

| Path | Contents |
|---|---|
| `config/skyguard.toml` | versioned configuration (QC rules, windows, split, seeds, severity, health) |
| `data/raw/iem_metar/` | raw observations exactly as downloaded (gzip) + `PROVENANCE.json` |
| `data/processed/` | canonical dataset + station registry (generated) |
| `data/benchmark/` | train / val / **frozen** `final_eval` benchmarks + manifests |
| `data/demo/` | deterministic SIH demo replay + its ground truth |
| `injection/` | reproducible anomaly injector |
| `ml/` | QC, features, models, stream engine, post-processing, evaluation |
| `backend/` | FastAPI app, SQLite store, replay runner |
| `dashboard/` | React + Vite dashboard |
| `edge/` | C99 edge checks, ESP32 sketch, host harness, station uplink simulator |
| `scripts/` | data, benchmark, training, evaluation, replay, scenarios, report |
| `tests/` | unit / integration / ml / e2e |
| `artifacts/` | final model, validation selection, final evaluation, scenarios, edge metrics |

## Data provenance

The real Indian observations are **IMD aerodrome METAR reports**, retrieved from the Iowa Environmental Mesonet archive
(network `IN__ASOS`). `scripts/fetch_raw_data.py` records the URL, retrieval time and SHA-256 of every file, and raw files are never modified. Some limits of this source:

* It is an AWS-like surface observation stream, **not** the IMD AWS network feed.
* Temperature resolution is 1 °C.
* Pressure is the QNH altimeter setting.
* ESSDP/INCOIS, NCAOR, Meghalaya and IMD DSP were evaluated but are not openly bulk-downloadable from this environment; the report gives the reasons.

The core demo needs no internet access: raw data, the demo stream and the trained model are all local.

## Setup

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
(On Linux/macOS use `source .venv/bin/activate` instead.)
```bash
pip install -r requirements.txt
```
```bash
cd dashboard && npm ci && npm run build && cd ..
```

## Reproduce everything (exact commands)

```bash
python scripts/fetch_raw_data.py          # only step needing internet; skipped files already in data/raw
python scripts/build_dataset.py           # raw -> data/processed/clean.csv + stations.json
python scripts/inject_anomalies.py --input data/processed/clean.csv --output data/benchmark.csv --seed 42   # stand-alone injector
python scripts/build_benchmark.py         # leakage-safe splits; FREEZES data/benchmark/final_eval (re-run = integrity check)
python scripts/train.py                   # trains baselines + ablation variants on TRAIN, selects on VALIDATION
python scripts/evaluate.py                # validation report -> artifacts/validation/
python scripts/evaluate_final.py --dataset data/benchmark/final_eval --model artifacts/final_model
python edge/build_edge.py                 # C edge library + RV32 size/stack + host timing -> artifacts/edge/
python scripts/build_demo.py              # deterministic SIH demo replay -> data/demo/
python scripts/make_report.py             # docs/SkyGuard_AI_Final_SIH_Evaluation_Report.md from artifacts
```

Order matters: the final benchmark is frozen **before** training. `train.py` refuses to run otherwise, and it records the
benchmark hash and freeze time in the model metadata. `evaluate_final.py` verifies both before evaluating.

## Run the API and dashboard

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The API serves the built dashboard. For UI development, run `npm run dev` inside `dashboard/` (Vite proxies `/api`).

## Replay and scenarios

* **Dashboard:** choose `sih_demo`, then *Start replay*. There's an 8-day warm-up, then 72 live hours.
* **CLI replay over HTTP:** `python scripts/replay.py --input data/demo/sih_demo.csv --speed 3600`
* **SIH scenarios A–J against a running API:** `python scripts/run_scenarios.py --base-url http://127.0.0.1:8000`. Evidence is written to `artifacts/scenarios/scenario_results.json`.

## API (all payloads typed; see http://localhost:8000/docs)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | service, model and DB state, detector mode |
| GET | `/api/v1/stations`, `/api/v1/stations/{id}` | status, health, maintenance, neighbours |
| GET | `/api/v1/health/stations` | sensor-health summary |
| POST | `/api/v1/observations` | one observation or `{"observations": [...]}` batch (≤ 5000) |
| GET | `/api/v1/observations/latest`, `/api/v1/observations/history` | stored raw + results |
| GET/PATCH | `/api/v1/anomalies`, `/api/v1/anomalies/{id}` | alerts; operator review state |
| GET | `/api/v1/metrics` | latency p50/p95/p99, throughput, CPU/RAM, distributions |
| POST | `/api/v1/replay/start`, `/api/v1/replay/stop`; GET `/api/v1/replay/status`, `/api/v1/replay/scenarios` | demo replay |
| POST | `/api/v1/admin/reset` | clear processed state for a reproducible demo |
| GET | `/api/v1/stream` | Server-Sent Events: live updates without page refresh |

### Semantics

* **`anomaly_probability`**: isotonic-calibrated on validation data (Brier score and reliability curve are reported).
* **`anomaly_score`**: the uncalibrated detector output. It is not a probability.
* **Deterministic QC hits** (missing, sentinel, impossible range) are reported as probability 1.0 with detector `deterministic_qc`.
* **No model available:** the API runs an explicit `qc_only_fallback` mode. It reports no probability, and the dashboard shows a red banner.
* **Delayed or out-of-order records** are accepted and scored on the history that precedes their *observation* time. Duplicates are idempotent.
* **Severity** is `score = 40·p + 20·min(|z|/10,1) + 15·min(persistence/6,1) + 10·(n_vars−1)/2 + 10·spatial_isolation + 5·qc_hard`, with levels MEDIUM ≥ 45, HIGH ≥ 65 and CRITICAL ≥ 82 (`config/skyguard.toml [severity]`).
* **Sensor health** is 100 minus weighted 24-hour rates of anomalies, missing data, flatline, drift/offset and communication gaps. It is a data-quality score, not a hardware diagnosis. Statuses: NORMAL ≥ 85, WATCH ≥ 70, DEGRADED ≥ 50, then CRITICAL.
* **Maintenance risk** is LOW, MEDIUM or HIGH with a reason, or "Insufficient history". It never predicts a time to failure.

## Security

* **Bearer token:** set `SKYGUARD_API_TOKEN` to require a token for every write. If it is unset, the service runs in local-development mode and `health` reports that no auth is required.
* **Secrets:** none are committed. See `.env.example`.
* **Input limits:** strict Pydantic validation (extra fields rejected), a 2 MB body limit, a 5000-record batch limit, and a per-client write rate limit.
* **Other controls:** configured CORS origins, no payloads or tokens in logs, and replay scenarios chosen by name only (no file paths).
* **Public deployment:** terminate HTTPS at a reverse proxy.

## Docker

```bash
cp .env.example .env
```
```bash
docker compose up --build
```

This is a single container (dashboard, API and edge library) with a named volume for the database and a health check. See the final report for whether Docker was verified in this environment.

## Tests

```bash
python -m pytest tests -q
```

The unit, integration, ml and e2e suites are all included. The e2e tests start a real uvicorn server and run the SIH scenarios over HTTP.

## Known limitations

See section 13 of the evaluation report. In brief:

* **Data:** METAR resolution and pressure type, and injected rather than real labelled faults.
* **Spatial coverage:** evidence exists only where neighbours are within the configured radius.
* **Edge:** no ESP32 hardware measurement, and no energy figures.
* **Scale:** single-process SQLite prototype.

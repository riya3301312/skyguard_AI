# SkyGuard AI - Deployment Verification

_Generated from `artifacts/` (2026-09-29T07:12:36+00:00)._

## Docker

* Status: **ENVIRONMENT-UNVERIFIED**
* Checked at: 2026-09-29T06:47:58+00:00 on `win32`
* Docker version: not installed

* No Docker engine is installed on the verification machine; build/startup/health/smoke were NOT run.

| Check | Result |
|---|---|
| build | NOT RUN |
| startup | NOT RUN |
| health | NOT RUN |
| api_smoke | NOT RUN |
| dashboard_smoke | NOT RUN |
| scenarios | NOT RUN |
| logs | NOT RUN |
| shutdown | NOT RUN |

Container definition checks that do not need a Docker engine (`tests/integration/test_docker.py::test_dockerfile_healthcheck_targets_existing_route`): the HEALTHCHECK URL maps to an existing API route; the image runs as a non-root user; compose requires `SKYGUARD_API_TOKEN` and persists the database in a named volume. The live container test is skipped when Docker is absent.

## Local (non-container) deployment verified on this machine

* Tests: 56 passed, 0 failed, 0 errors, 1 skipped (by suite: {'e2e': 2, 'integration': 12, 'ml': 7, 'unit': 36}).
* SIH scenarios against a running uvicorn server: 11/11 passed (2026-09-29T07:23:20+00:00); detector hybrid, model fv2/full/2026-09-29T06:58:05+00:00.
* Dashboard verified in a browser against the real API (live SSE updates, station view, alert explanation incl. weather-vs-sensor evidence, analytics, connectivity log).
* Security: bearer token on writes, strict validation, 2 MB body limit, 5000-record batch limit, per-client write rate limit, configured CORS, no secrets committed (`.env.example` only).

## Not verified

* Docker build / container startup / container health / container smoke tests (no Docker engine).
* Physical ESP32 hardware and energy use.
* HTTPS termination and multi-process / PostgreSQL deployment.

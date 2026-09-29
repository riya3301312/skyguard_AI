"""SkyGuard AI API.  uvicorn backend.app:app --host 0.0.0.0 --port 8000

Environment (see .env.example): SKYGUARD_API_TOKEN, SKYGUARD_CORS_ORIGINS, SKYGUARD_DB,
SKYGUARD_MODEL_DIR, SKYGUARD_STATIONS, SKYGUARD_DEMO_DIR, SKYGUARD_WRITE_RATE_PER_MIN.
"""
import os

# single-row inference: avoid OpenMP thread spin-up per call (set before sklearn loads)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
import hmac
import json
import logging
import sqlite3
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.db import DB
from backend.replay import ReplayRunner
from backend.schemas import (AnomalyOut, AnomalyUpdate, BatchIn, HealthOut, IngestResponse, ObservationIn,
                             ObservationResult, ReplayStart, StationOut)
from ml.config import ROOT, load_config
from ml.evaluation import latency_stats
from ml.pipeline import ModelBundle, load_stations
from ml.stream import StreamEngine

log = logging.getLogger("skyguard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
CFG = load_config()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class State:
    def __init__(self):
        self.db = DB(env("SKYGUARD_DB", str(ROOT / "data" / "skyguard.db")))
        self.stations = load_stations(Path(env("SKYGUARD_STATIONS", str(ROOT / "data" / "processed" / "stations.json"))))
        self.bundle, self.model_error = None, None
        try:
            self.bundle = ModelBundle.load(Path(env("SKYGUARD_MODEL_DIR", str(ROOT / "artifacts" / "final_model"))))
        except Exception as e:  # explicit, labelled fallback - never a silent substitution
            self.model_error = f"{type(e).__name__}: {e}"
            log.error("model unavailable, running qc_only_fallback: %s", self.model_error)
        self.engine = StreamEngine(self.stations, self.bundle, CFG)
        self.db.upsert_stations(self.stations)
        self.token = os.environ.get("SKYGUARD_API_TOKEN") or None
        self.latency = deque(maxlen=20000)       # engine processing latency (ms)
        self.e2e = deque(maxlen=20000)           # request-side latency incl. DB + publish (ms)
        self.done_at = deque(maxlen=100000)      # wall-clock completion times for throughput
        self.subscribers: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started = time.time()
        self.rate: dict[str, deque] = {}
        self.replay = ReplayRunner(self)

    def reset(self) -> None:
        self.db.reset()
        self.engine = StreamEngine(self.stations, self.bundle, CFG)
        self.latency.clear(), self.e2e.clear(), self.done_at.clear()

    def publish(self, event: str, data: dict) -> None:
        if not self.loop:
            return
        msg = f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
        for q in list(self.subscribers):
            self.loop.call_soon_threadsafe(_offer, q, msg)

    def ingest(self, observations: list[dict], quiet: bool = False) -> list[dict]:
        """Shared ingestion path for the API, the replay engine and the station simulator."""
        t0 = time.perf_counter()
        results = self.engine.process_many(observations)
        src = {o["station_id"]: o.get("source") for o in observations}
        for r in results:
            if r["status"] != "accepted":
                continue
            try:
                self.db.insert_result(r, src.get(r["station_id"]))
            except sqlite3.Error as e:
                for x in results:  # keep engine and DB consistent: nothing from this request counts as ingested
                    if x["status"] == "accepted":
                        self.engine.forget(x["station_id"], x["observation_time"])
                log.error("database error on insert: %s", type(e).__name__)
                raise HTTPException(503, "database unavailable; observations not stored, retry later") from e
        for rv in self.engine.pop_revisions():
            self.db.apply_revision(rv)
            if not quiet:
                self.publish("revision", {k: rv[k] for k in ("station_id", "observation_time", "was_anomaly", "is_anomaly", "fault_pattern")})
        per = (time.perf_counter() - t0) * 1000 / max(1, len(results))
        now = time.time()
        for r in results:
            if r["status"] != "accepted":
                continue
            self.latency.append(r["processing_latency_ms"])
            self.e2e.append(per)
            self.done_at.append(now)
            if not quiet:
                self.publish("observation", _compact(r))
                if r["is_anomaly"] or r.get("gap_event") or r.get("regional_suppressed"):
                    self.publish("anomaly", {"station_id": r["station_id"], "observation_time": r["observation_time"],
                                             "fault_pattern": r["fault_pattern"], "severity": r["severity"],
                                             "regional_suppressed": r.get("regional_suppressed"), "gap": bool(r.get("gap_event"))})
        return results


def _offer(q: asyncio.Queue, msg: str) -> None:
    if not q.full():
        q.put_nowait(msg)


def _compact(r: dict) -> dict:
    keys = ["station_id", "observation_time", "ingestion_time", "temperature_c", "pressure_hpa", "relative_humidity_pct",
            "is_anomaly", "anomaly_probability", "fault_pattern", "parameter", "severity", "health_score", "health_status",
            "estimated_temperature_c", "estimated_pressure_hpa", "estimated_relative_humidity_pct", "regional_evidence",
            "regional_suppressed", "delayed", "out_of_order", "processing_latency_ms", "edge_flags", "edge_health_score"]
    return {k: r.get(k) for k in keys}


S: State | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global S
    S = State()
    S.loop = asyncio.get_running_loop()
    log.info("SkyGuard API ready: %d stations, mode=%s", len(S.stations), S.engine.mode)
    yield
    S.replay.stop()


app = FastAPI(title="SkyGuard AI", version="1.0.0", lifespan=lifespan,
              description="Real-time anomaly detection for Automatic Weather Stations (T, P, RH).")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in env(
    "SKYGUARD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
                   allow_methods=["GET", "POST", "PATCH"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def limits(request: Request, call_next):
    """Upload size limit and per-client write rate limit (safe logging: no payloads, no tokens)."""
    if request.method in ("POST", "PATCH"):
        cl = request.headers.get("content-length")
        if cl is None or not cl.isdigit():
            if request.url.path.startswith("/api/v1/observations"):
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        elif int(cl) > CFG["api"]["max_body_bytes"]:
            return JSONResponse({"detail": f"payload exceeds {CFG['api']['max_body_bytes']} bytes"}, status_code=413)
        limit = int(env("SKYGUARD_WRITE_RATE_PER_MIN", "600"))
        key = request.client.host if request.client else "unknown"
        q = S.rate.setdefault(key, deque())
        now = time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        q.append(now)
    return await call_next(request)


def require_token(authorization: str | None = Header(default=None)) -> None:
    if S.token is None:
        return  # local development mode (documented); set SKYGUARD_API_TOKEN in any shared deployment
    ok = authorization is not None and authorization.startswith("Bearer ") and hmac.compare_digest(authorization[7:], S.token)
    if not ok:
        raise HTTPException(401, "missing or invalid bearer token")


# ------------------------------------------------------------------------------------------ health
@app.get("/api/v1/health", response_model=HealthOut)
def health():
    try:
        S.db.query("SELECT 1")
        db_ok = True
    except sqlite3.Error:
        db_ok = False
    md = S.bundle.metadata if S.bundle else {}
    return HealthOut(status="ok" if db_ok else "degraded", model_loaded=S.bundle is not None, detector_mode=S.engine.mode,
                     model_version=f"{md.get('feature_version')}/{md.get('selected_variant')}/{md.get('trained_at')}" if md else None,
                     database_ok=db_ok, stations=len(S.stations), auth_required_for_writes=S.token is not None)


def _active_counts() -> dict:
    rows = S.db.query("SELECT station_id, COUNT(*) n, SUM(severity='CRITICAL') c FROM anomalies a WHERE state IN ('OPEN','ACKNOWLEDGED') "
                      "AND observation_time >= (SELECT datetime(MAX(o.observation_time), '-2 hours') FROM observations o WHERE o.station_id=a.station_id) "
                      "GROUP BY station_id")
    return {r["station_id"]: (r["n"], r["c"] or 0) for r in rows}


def _station(sid: str, active: dict) -> StationOut:
    return StationOut(**S.engine.station_summary(sid), active_anomalies=active.get(sid, (0, 0))[0],
                      active_critical=active.get(sid, (0, 0))[1])


@app.get("/api/v1/stations", response_model=list[StationOut])
def stations():
    active = _active_counts()
    return [_station(s, active) for s in sorted(S.stations)]


@app.get("/api/v1/stations/{station_id}", response_model=StationOut)
def station(station_id: str):
    if station_id not in S.stations:
        raise HTTPException(404, "unknown station")
    return _station(station_id, _active_counts())


@app.get("/api/v1/health/stations")
def station_health():
    return [{k: v for k, v in S.engine.station_summary(s).items() if k in (
        "station_id", "health_score", "health_status", "health_reasons", "maintenance", "status", "label")} for s in sorted(S.stations)]


# ------------------------------------------------------------------------------------------ observations
@app.post("/api/v1/observations", response_model=IngestResponse, dependencies=[Depends(require_token)])
def post_observations(payload: BatchIn | ObservationIn):
    obs = payload.observations if isinstance(payload, BatchIn) else [payload]
    if len(obs) > CFG["api"]["max_batch"]:
        raise HTTPException(413, "batch too large")
    res = S.ingest([o.model_dump(mode="json") for o in obs])
    return IngestResponse(accepted=sum(r["status"] == "accepted" for r in res),
                          duplicates=sum(r["status"] == "duplicate" for r in res),
                          rejected=sum(r["status"] == "rejected" for r in res),
                          results=[ObservationResult(**r) for r in res])


def _obs_row(r: dict) -> dict:
    r = dict(r)
    for k in ("delayed", "out_of_order", "is_anomaly", "regional_evidence", "regional_suppressed", "replacement_recommended"):
        r[k] = bool(r[k]) if r.get(k) is not None else None
    r["edge_flags"] = json.loads(r["edge_flags"]) if r.get("edge_flags") else None
    return r


@app.get("/api/v1/observations/latest")
def latest(station_id: str | None = None):
    sql = ("SELECT o.* FROM observations o JOIN (SELECT station_id, MAX(observation_time) t FROM observations GROUP BY station_id) m "
           "ON o.station_id=m.station_id AND o.observation_time=m.t")
    rows = S.db.query(sql + (" WHERE o.station_id=?" if station_id else ""), (station_id,) if station_id else ())
    return [_obs_row(r) for r in rows]


@app.get("/api/v1/observations/history")
def history(station_id: str, start: str | None = None, end: str | None = None, limit: int = Query(500, ge=1, le=5000)):
    sql, args = "SELECT * FROM observations WHERE station_id=?", [station_id]
    if start:
        sql, args = sql + " AND observation_time>=?", args + [start]
    if end:
        sql, args = sql + " AND observation_time<=?", args + [end]
    rows = S.db.query(f"SELECT * FROM ({sql} ORDER BY observation_time DESC LIMIT ?) ORDER BY observation_time", tuple(args + [limit]))
    return [_obs_row(r) for r in rows]


# ------------------------------------------------------------------------------------------ anomalies
def _anom(r: dict) -> AnomalyOut:
    return AnomalyOut(**{**r, "regional_evidence": bool(r["regional_evidence"]),
                         "explanation": json.loads(r["explanation"]) if r["explanation"] else None})


@app.get("/api/v1/anomalies", response_model=list[AnomalyOut])
def anomalies(station_id: str | None = None, state: str | None = None, severity: str | None = None,
              include_info: bool = True, limit: int = Query(200, ge=1, le=2000)):
    sql, args = "SELECT * FROM anomalies WHERE 1=1", []
    for col, val in (("station_id", station_id), ("state", state), ("severity", severity)):
        if val:
            sql, args = sql + f" AND {col}=?", args + [val]
    if not include_info:
        sql += " AND state != 'INFO'"
    return [_anom(r) for r in S.db.query(sql + " ORDER BY observation_time DESC, id DESC LIMIT ?", tuple(args + [limit]))]


@app.get("/api/v1/anomalies/{anomaly_id}", response_model=AnomalyOut)
def anomaly(anomaly_id: int):
    rows = S.db.query("SELECT * FROM anomalies WHERE id=?", (anomaly_id,))
    if not rows:
        raise HTTPException(404, "anomaly not found")
    return _anom(rows[0])


@app.patch("/api/v1/anomalies/{anomaly_id}", response_model=AnomalyOut, dependencies=[Depends(require_token)])
def review_anomaly(anomaly_id: int, body: AnomalyUpdate):
    """Operator review (acknowledge / mark false alarm). Only the review state changes."""
    if not S.db.execute("UPDATE anomalies SET state=? WHERE id=? AND state!='INFO'", (body.state, anomaly_id)):
        raise HTTPException(404, "anomaly not found")
    return anomaly(anomaly_id)


# ------------------------------------------------------------------------------------------ metrics
@app.get("/api/v1/metrics")
def metrics():
    now = time.time()
    recent = [t for t in S.done_at if now - t <= 60]
    proc = psutil.Process()
    by = lambda col: {r["k"]: r["n"] for r in S.db.query(  # noqa: E731
        f"SELECT {col} k, COUNT(*) n FROM anomalies WHERE state NOT IN ('INFO','RETRACTED') GROUP BY {col}")}
    tot = S.db.query("SELECT COUNT(*) n, SUM(temperature_c IS NOT NULL AND pressure_hpa IS NOT NULL AND relative_humidity_pct IS NOT NULL) ok, "
                     "SUM(delayed) delayed, SUM(out_of_order) ooo FROM observations")[0]
    return {
        "observations": tot["n"], "complete_observations_pct": 100 * (tot["ok"] or 0) / max(tot["n"], 1),
        "delayed_observations": tot["delayed"] or 0, "out_of_order_observations": tot["ooo"] or 0,
        "processing_latency_ms": latency_stats(list(S.latency)) if S.latency else None,
        "end_to_end_latency_ms": latency_stats(list(S.e2e)) if S.e2e else None,
        "throughput_obs_per_s_last_60s": len(recent) / 60.0,
        "anomalies_by_station": by("station_id"), "anomalies_by_parameter": by("parameter"),
        "anomalies_by_pattern": by("fault_pattern"), "anomalies_by_severity": by("severity"),
        "review_states": {r["state"]: r["n"] for r in S.db.query("SELECT state, COUNT(*) n FROM anomalies GROUP BY state")},
        "regional_events_suppressed": S.db.query("SELECT COUNT(*) n FROM anomalies WHERE state='INFO'")[0]["n"],
        "cpu_percent": proc.cpu_percent(interval=None), "rss_mb": proc.memory_info().rss / 2**20,
        "uptime_s": now - S.started, "detector_mode": S.engine.mode,
        "health_trend": S.db.query("SELECT station_id, substr(observation_time,1,13) hour, AVG(health_score) health "
                                   "FROM observations GROUP BY station_id, hour ORDER BY hour DESC LIMIT 600"),
    }


# ------------------------------------------------------------------------------------------ replay + live stream
@app.post("/api/v1/admin/reset", dependencies=[Depends(require_token)])
def admin_reset():
    """Clear processed state for a clean, reproducible demo/scenario run (stations and model are kept)."""
    S.replay.stop()
    S.reset()
    return {"status": "reset"}


@app.get("/api/v1/replay/scenarios")
def scenarios():
    return S.replay.scenarios()


@app.get("/api/v1/replay/status")
def replay_status():
    return S.replay.status()


@app.post("/api/v1/replay/start", dependencies=[Depends(require_token)])
def replay_start(body: ReplayStart):
    if body.scenario not in S.replay.scenarios():
        raise HTTPException(404, "unknown scenario")
    return S.replay.start(body.scenario, body.speed, body.reset)


@app.post("/api/v1/replay/stop", dependencies=[Depends(require_token)])
def replay_stop():
    return S.replay.stop()


@app.get("/api/v1/stream")
async def stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=2000)
    S.subscribers.add(q)

    async def gen():
        try:
            yield "event: hello\ndata: {}\n\n"
            while not await request.is_disconnected():
                try:
                    yield await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            S.subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


DIST = ROOT / "dashboard" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="dashboard")

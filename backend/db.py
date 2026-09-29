"""SQLite storage (prototype; PostgreSQL is the scale target). Raw values are immutable (trigger)."""
import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations(
  station_id TEXT PRIMARY KEY, latitude REAL, longitude REAL, altitude_m REAL, source TEXT);
CREATE TABLE IF NOT EXISTS observations(
  id INTEGER PRIMARY KEY, station_id TEXT NOT NULL, observation_time TEXT NOT NULL, ingestion_time TEXT NOT NULL,
  temperature_c REAL, pressure_hpa REAL, relative_humidity_pct REAL, source TEXT,
  delayed INTEGER, out_of_order INTEGER, lag_minutes REAL, processing_latency_ms REAL, detector_mode TEXT,
  is_anomaly INTEGER, anomaly_probability REAL, anomaly_score REAL, fault_pattern TEXT, parameter TEXT,
  severity TEXT, severity_score REAL, regional_evidence INTEGER, regional_suppressed INTEGER,
  weather_status TEXT, regional_score REAL,
  health_score REAL, health_status TEXT,
  estimated_temperature_c REAL, estimated_pressure_hpa REAL, estimated_relative_humidity_pct REAL,
  replacement_recommended INTEGER, edge_flags TEXT, edge_health_score REAL,
  UNIQUE(station_id, observation_time));
CREATE INDEX IF NOT EXISTS obs_station_time ON observations(station_id, observation_time);
CREATE TRIGGER IF NOT EXISTS raw_immutable BEFORE UPDATE OF
  station_id, observation_time, temperature_c, pressure_hpa, relative_humidity_pct ON observations
BEGIN SELECT RAISE(ABORT, 'raw observations are immutable'); END;
CREATE TABLE IF NOT EXISTS anomalies(
  id INTEGER PRIMARY KEY, observation_id INTEGER, station_id TEXT NOT NULL, observation_time TEXT NOT NULL,
  detected_at TEXT NOT NULL, parameter TEXT, fault_pattern TEXT NOT NULL, severity TEXT,
  anomaly_probability REAL, anomaly_score REAL, raw_value REAL, estimated_value REAL,
  regional_evidence INTEGER, explanation TEXT, state TEXT NOT NULL DEFAULT 'OPEN', detector_mode TEXT);
CREATE INDEX IF NOT EXISTS anom_station ON anomalies(station_id, observation_time);
"""

OBS_COLS = ["station_id", "observation_time", "ingestion_time", "temperature_c", "pressure_hpa", "relative_humidity_pct",
            "source", "delayed", "out_of_order", "lag_minutes", "processing_latency_ms", "detector_mode", "is_anomaly",
            "anomaly_probability", "anomaly_score", "fault_pattern", "parameter", "severity", "severity_score",
            "regional_evidence", "regional_suppressed", "weather_status", "regional_score", "health_score", "health_status", "estimated_temperature_c",
            "estimated_pressure_hpa", "estimated_relative_humidity_pct", "replacement_recommended", "edge_flags",
            "edge_health_score"]


class DB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        for col in ("weather_status TEXT", "regional_score REAL"):  # migrate databases created before these columns
            try:
                self.conn.execute(f"ALTER TABLE observations ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.conn.executescript("DELETE FROM anomalies; DELETE FROM observations;")
            self.conn.commit()

    def upsert_stations(self, stations: dict) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO stations VALUES (?,?,?,?,?)",
                [(s, m.get("latitude"), m.get("longitude"), m.get("altitude_m"), m.get("source")) for s, m in stations.items()])
            self.conn.commit()

    def insert_result(self, r: dict, source: str | None) -> int | None:
        """Insert one processed observation (+ anomaly rows). Returns observation id, None if duplicate."""
        vals = {**r, "source": source, "edge_flags": json.dumps(r.get("edge_flags")) if r.get("edge_flags") is not None else None}
        with self.lock:
            cur = self.conn.execute(
                f"INSERT OR IGNORE INTO observations ({','.join(OBS_COLS)}) VALUES ({','.join('?' * len(OBS_COLS))})",
                [vals.get(c) for c in OBS_COLS])
            if cur.rowcount == 0:
                self.conn.commit()
                return None
            oid = cur.lastrowid
            if r.get("is_anomaly"):
                p = r.get("parameter") or ""
                raw = r.get(p) if p in ("temperature_c", "pressure_hpa", "relative_humidity_pct") else None
                est = r.get(f"estimated_{p}") if raw is not None else None
                self._anomaly(oid, r, r["fault_pattern"], r.get("severity"), raw, est, r.get("explanation"), "OPEN")
            if r.get("regional_suppressed"):
                self._anomaly(oid, r, "REGIONAL_WEATHER_EVENT", None, None, None, r.get("explanation"), "INFO")
            if r.get("gap_event"):
                g = r["gap_event"]
                self._anomaly(None, {**r, "observation_time": g["start"], "parameter": "all", "anomaly_probability": None,
                                     "anomaly_score": None}, "MISSING_COMMUNICATION", "MEDIUM", None, None,
                              {"status": "ANOMALY DETECTED", "likely_fault_pattern": "Likely fault pattern: missing data / communication gap",
                               "evidence": [g["explanation"]], "pattern": "MISSING_COMMUNICATION", "parameter": "all",
                               "detector_mode": "deterministic_qc", "note": "Observable data pattern, not a confirmed physical hardware diagnosis."}, "OPEN")
            self.conn.commit()
            return oid

    def apply_revision(self, rv: dict) -> None:
        """Apply a re-scored (provisional -> revised) result. Only result columns change; raw values are immutable."""
        cols = ["is_anomaly", "anomaly_probability", "anomaly_score", "fault_pattern", "parameter", "severity", "severity_score",
                "regional_evidence", "regional_suppressed", "weather_status", "regional_score", "estimated_temperature_c", "estimated_pressure_hpa",
                "estimated_relative_humidity_pct", "replacement_recommended"]
        key = (rv["station_id"], rv["observation_time"])
        with self.lock:
            cur = self.conn.execute(f"UPDATE observations SET {', '.join(c + '=?' for c in cols)} WHERE station_id=? AND observation_time=?",
                                    [rv.get(c) for c in cols] + list(key))
            if cur.rowcount == 0:
                self.conn.commit()
                return
            oid = self.conn.execute("SELECT id, ingestion_time FROM observations WHERE station_id=? AND observation_time=?", key).fetchone()
            live = "station_id=? AND observation_time=? AND observation_id IS NOT NULL AND state IN ('OPEN','ACKNOWLEDGED')"
            p = rv.get("parameter") or ""
            raw = rv["raw"].get(p) if p in rv["raw"] else None
            est = rv.get(f"estimated_{p}") if raw is not None else None
            if rv["was_anomaly"] and not rv["is_anomaly"]:
                note = {"status": "RETRACTED", "evidence": [f"Provisional alert retracted: {rv['reason']}."] +
                        ((rv.get("explanation") or {}).get("evidence") or []), "note": "Revised with complete spatial context."}
                self.conn.execute(f"UPDATE anomalies SET state='RETRACTED', explanation=? WHERE {live}", (json.dumps(note), *key))
            elif rv["is_anomaly"] and self.conn.execute(f"UPDATE anomalies SET fault_pattern=?, parameter=?, severity=?, anomaly_probability=?, "
                                                        f"anomaly_score=?, raw_value=?, estimated_value=?, regional_evidence=?, explanation=? WHERE {live}",
                                                        (rv["fault_pattern"], p, rv["severity"], rv["anomaly_probability"], rv["anomaly_score"], raw, est,
                                                         int(rv["regional_evidence"]), json.dumps(rv.get("explanation")), *key)).rowcount == 0:
                self._anomaly(oid["id"], {**rv, "ingestion_time": oid["ingestion_time"]}, rv["fault_pattern"], rv["severity"], raw, est,
                              rv.get("explanation"), "OPEN")
            if rv.get("regional_suppressed"):
                self._anomaly(oid["id"], {**rv, "ingestion_time": oid["ingestion_time"]}, "REGIONAL_WEATHER_EVENT", None, None, None,
                              rv.get("explanation"), "INFO")
            self.conn.commit()

    def _anomaly(self, oid, r, pattern, sev, raw, est, exp, state):
        self.conn.execute(
            "INSERT INTO anomalies (observation_id, station_id, observation_time, detected_at, parameter, fault_pattern, severity,"
            " anomaly_probability, anomaly_score, raw_value, estimated_value, regional_evidence, explanation, state, detector_mode)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, r["station_id"], r["observation_time"], r["ingestion_time"], r.get("parameter"), pattern, sev,
             r.get("anomaly_probability"), r.get("anomaly_score"), raw, est, int(bool(r.get("regional_evidence"))),
             json.dumps(exp) if exp is not None else None, state, r.get("detector_mode")))

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(x) for x in self.conn.execute(sql, args).fetchall()]

    def execute(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.rowcount

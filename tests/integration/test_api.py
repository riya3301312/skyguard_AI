import sqlite3

import pytest

import backend.app as appmod
from tests.integration.conftest import history


def post(c, obs, **kw):
    body = {"observations": obs} if isinstance(obs, list) else obs
    return c.post("/api/v1/observations", json=body, **kw)


def warm(c, sid="AWS_A", **kw):
    r = post(c, history(sid, **kw))
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_stations(make_client):
    with make_client() as c:
        h = c.get("/api/v1/health").json()
        assert h["model_loaded"] and h["detector_mode"] == "hybrid" and h["database_ok"] and h["stations"] == 4
        st = c.get("/api/v1/stations").json()
        assert {s["station_id"] for s in st} == {"AWS_A", "AWS_B", "AWS_C", "AWS_REMOTE"}
        assert c.get("/api/v1/stations/NOPE").status_code == 404


def test_valid_batch_upload_and_history(make_client):
    with make_client() as c:
        body = warm(c)
        assert body["accepted"] == 216 and body["duplicates"] == 0
        hist = c.get("/api/v1/observations/history", params={"station_id": "AWS_A", "limit": 50}).json()
        assert len(hist) == 50 and hist[0]["observation_time"] < hist[-1]["observation_time"]
        latest = c.get("/api/v1/observations/latest", params={"station_id": "AWS_A"}).json()
        assert latest[0]["observation_time"] == hist[-1]["observation_time"]


def test_malformed_and_missing_parameter(make_client):
    with make_client() as c:
        assert post(c, {"station_id": "AWS_A", "timestamp": "2024-05-01T00:00:00Z", "temperature_c": "hot"}).status_code == 422
        assert post(c, {"station_id": "AWS_A", "timestamp": "not a time"}).status_code == 422
        assert post(c, {"timestamp": "2024-05-01T00:00:00Z", "temperature_c": 20}).status_code == 422
        assert post(c, {"station_id": "bad id!", "timestamp": "2024-05-01T00:00:00Z"}).status_code == 422
        assert post(c, {"station_id": "AWS_A", "timestamp": "2024-05-01T00:00:00Z", "extra": 1}).status_code == 422
        warm(c)
        r = post(c, {"station_id": "AWS_A", "timestamp": "2024-05-10T00:00:00Z", "pressure_hpa": 1008.0, "relative_humidity_pct": 60.0})
        res = r.json()["results"][0]
        assert r.status_code == 200 and res["status"] == "accepted" and res["temperature_c"] is None
        allmiss = post(c, {"station_id": "AWS_A", "timestamp": "2024-05-10T01:00:00Z"}).json()["results"][0]
        assert allmiss["is_anomaly"] and allmiss["fault_pattern"] == "MISSING_COMMUNICATION"
        unknown = post(c, {"station_id": "AWS_ZZ", "timestamp": "2024-05-10T01:00:00Z", "temperature_c": 1}).json()
        assert unknown["rejected"] == 1


def test_duplicate_delayed_out_of_order_and_future(make_client):
    with make_client() as c:
        obs = history("AWS_A")
        body = post(c, obs[:100] + obs[101:]).json()
        assert body["accepted"] == len(obs) - 1
        dup = post(c, obs[:5]).json()
        assert dup["duplicates"] == 5 and dup["accepted"] == 0
        late = post(c, obs[100]).json()["results"][0]
        assert late["status"] == "accepted" and late["out_of_order"] and late["delayed"] and not late["is_anomaly"]
        fut = post(c, {"station_id": "AWS_A", "timestamp": "2099-01-01T00:00:00Z", "temperature_c": 20}).json()
        assert fut["rejected"] == 1 and "future" in fut["results"][0]["reason"]


def test_spike_detection_explanation_and_raw_preservation(make_client):
    with make_client() as c:
        for s in ("AWS_A", "AWS_B", "AWS_C"):
            warm(c, s)
        r = post(c, [{"station_id": s, "timestamp": "2024-05-10T00:00:00Z", "temperature_c": 55.2 if s == "AWS_A" else 26.0,
                      "pressure_hpa": 1008.0, "relative_humidity_pct": 62.0} for s in ("AWS_B", "AWS_C", "AWS_A")]).json()
        a = [x for x in r["results"] if x["station_id"] == "AWS_A"][0]
        assert a["is_anomaly"] and a["temperature_c"] == 55.2 and not a["regional_evidence"], a
        assert a["explanation"]["evidence"] and a["estimated_temperature_c"] is not None and a["replacement_recommended"]
        an = c.get("/api/v1/anomalies", params={"station_id": "AWS_A"}).json()[0]
        assert an["raw_value"] == 55.2 and an["estimated_value"] == a["estimated_temperature_c"]
        assert c.get(f"/api/v1/anomalies/{an['id']}").json()["id"] == an["id"]
        assert c.patch(f"/api/v1/anomalies/{an['id']}", json={"state": "FALSE_ALARM"}).json()["state"] == "FALSE_ALARM"
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            appmod.S.db.conn.execute("UPDATE observations SET temperature_c=31.8 WHERE station_id='AWS_A'")
        stored = c.get("/api/v1/observations/latest", params={"station_id": "AWS_A"}).json()[0]
        assert stored["temperature_c"] == 55.2


def test_no_spatial_neighbors_path(make_client):
    with make_client() as c:
        warm(c, "AWS_REMOTE")
        r = post(c, {"station_id": "AWS_REMOTE", "timestamp": "2024-05-10T00:00:00Z", "temperature_c": 50.0,
                     "pressure_hpa": 1008.0, "relative_humidity_pct": 62.0}).json()["results"][0]
        assert r["status"] == "accepted" and r["is_anomaly"]
        assert any("No spatial neighbours" in e for e in r["explanation"]["evidence"])
        assert c.get("/api/v1/stations/AWS_REMOTE").json()["neighbors"] == []


def test_model_unavailable_explicit_fallback(make_client):
    with make_client(model=False) as c:
        h = c.get("/api/v1/health").json()
        assert not h["model_loaded"] and h["detector_mode"] == "qc_only_fallback"
        warm(c)
        r = post(c, {"station_id": "AWS_A", "timestamp": "2024-05-10T00:00:00Z", "temperature_c": 25.0,
                     "pressure_hpa": -9999.0, "relative_humidity_pct": 60.0}).json()["results"][0]
        assert r["detector_mode"] == "qc_only_fallback" and r["anomaly_probability"] is None
        assert r["is_anomaly"] and r["fault_pattern"] == "SENTINEL_CORRUPTION"


def test_database_error_returns_503_and_retry_succeeds(make_client, monkeypatch):
    with make_client() as c:
        warm(c)
        ob = {"station_id": "AWS_A", "timestamp": "2024-05-10T00:00:00Z", "temperature_c": 25.0, "pressure_hpa": 1008.0,
              "relative_humidity_pct": 60.0}
        real = appmod.S.db.insert_result

        def boom(*a, **k):
            raise sqlite3.OperationalError("disk I/O error")
        monkeypatch.setattr(appmod.S.db, "insert_result", boom)
        assert post(c, ob).status_code == 503
        monkeypatch.setattr(appmod.S.db, "insert_result", real)
        again = post(c, ob).json()
        assert again["accepted"] == 1 and again["duplicates"] == 0  # engine state rolled back, retry is not lost


def test_security_token_body_limit_and_cors(make_client):
    with make_client(token="s3cret") as c:
        ob = {"station_id": "AWS_A", "timestamp": "2024-05-10T00:00:00Z", "temperature_c": 25.0}
        assert post(c, ob).status_code == 401
        assert post(c, ob, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert post(c, ob, headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/v1/health").json()["auth_required_for_writes"]
        big = c.post("/api/v1/observations", content=b"x" * 2_100_000,
                     headers={"Authorization": "Bearer s3cret", "Content-Type": "application/json"})
        assert big.status_code == 413
        bad = c.options("/api/v1/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
        assert bad.headers.get("access-control-allow-origin") != "http://evil.example"
        ok = c.options("/api/v1/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_metrics_and_replay_validation(make_client):
    with make_client() as c:
        warm(c)
        m = c.get("/api/v1/metrics").json()
        assert m["observations"] == 216 and m["processing_latency_ms"]["p95_ms"] > 0 and "cpu_percent" in m
        assert c.post("/api/v1/replay/start", json={"scenario": "../../etc/passwd"}).status_code == 422
        assert c.post("/api/v1/replay/start", json={"scenario": "does_not_exist"}).status_code == 404

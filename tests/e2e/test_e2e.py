"""End-to-end: real uvicorn server + HTTP clients + SSE + scenario replay through station clients."""
import json
import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

from ml.config import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "final_model" / "model.joblib").exists() or
                                not (ROOT / "data" / "demo" / "sih_demo.csv").exists(),
                                reason="needs trained model (scripts/train.py) and demo (scripts/build_demo.py)")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = free_port()
    log = open(tmp_path_factory.mktemp("log") / "server.log", "wb")
    env = dict(os.environ, SKYGUARD_DB=str(tmp_path_factory.mktemp("db") / "e2e.sqlite"), SKYGUARD_API_TOKEN="e2e-token",
               SKYGUARD_WRITE_RATE_PER_MIN="100000")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)  # file, not a pipe: a full pipe would block the server
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(base + "/api/v1/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError(f"server did not start, see {log.name}")
    yield base
    proc.terminate()
    proc.wait(timeout=20)


def test_live_sse_update_without_refresh(server):
    events = []

    def listen():
        with httpx.stream("GET", server + "/api/v1/stream", timeout=30) as r:
            for line in r.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                if "observation" in events:
                    return

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    time.sleep(1.0)
    H = {"Authorization": "Bearer e2e-token"}
    sid = httpx.get(server + "/api/v1/stations").json()[0]["station_id"]
    r = httpx.post(server + "/api/v1/observations", headers=H, json={"station_id": sid, "timestamp": "2024-09-01T00:00:00Z",
                                                                      "temperature_c": 27.0, "pressure_hpa": 1008.0, "relative_humidity_pct": 70.0})
    assert r.status_code == 200
    t.join(timeout=20)
    assert "hello" in events and "observation" in events
    assert httpx.post(server + "/api/v1/observations", json={"station_id": sid, "timestamp": "2024-09-01T01:00:00Z"}).status_code == 401


def test_sih_scenarios_end_to_end(server):
    r = subprocess.run([sys.executable, "scripts/run_scenarios.py", "--base-url", server, "--token", "e2e-token"], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((ROOT / "artifacts" / "scenarios" / "scenario_results.json").read_text())["scenarios"]
    # the E2E behaviours required by the specification
    assert res["B_temperature_spike"]["passed"], res["B_temperature_spike"]            # spike -> alert (+ explanation, estimate)
    assert res["B_temperature_spike"]["evidence"]["estimated_temperature_c"] is not None  # -> correction recommendation
    assert res["C_flatline"]["passed"], res["C_flatline"]                              # flatline -> alert
    assert res["D_gradual_drift"]["passed"], res["D_gradual_drift"]                    # drift -> alert
    assert res["E_communication_failure"]["passed"], res["E_communication_failure"]
    assert res["F_regional_weather_event"]["passed"], res["F_regional_weather_event"]  # regional -> reduced FP
    assert res["I_offline_remote_station"]["passed"], res["I_offline_remote_station"]  # outage -> buffer -> reconnect -> sync

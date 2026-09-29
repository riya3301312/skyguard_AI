"""Verify the Docker deployment and record evidence in artifacts/docker_verification/.

python scripts/verify_docker.py
If no Docker engine is available the result is ENVIRONMENT-UNVERIFIED (nothing is fabricated).
Otherwise: build -> compose up -> health -> API smoke (ingest) -> dashboard smoke -> SIH scenarios -> logs -> compose down.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "docker_verification"
TOKEN = os.environ.get("SKYGUARD_API_TOKEN", "docker-verify-token")


def run(cmd, **kw):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)
    return {"cmd": " ".join(cmd), "returncode": r.returncode, "stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": sys.platform}
    if shutil.which("docker") is None:
        res.update(status="ENVIRONMENT-UNVERIFIED", docker_version=None,
                   reason="No Docker engine is installed on the verification machine; build/startup/health/smoke were NOT run.",
                   checks={k: "NOT RUN" for k in ("build", "startup", "health", "api_smoke", "dashboard_smoke", "scenarios", "logs", "shutdown")})
        (OUT / "docker_verification.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return
    env = dict(os.environ, SKYGUARD_API_TOKEN=TOKEN, SKYGUARD_CORS_ORIGINS="http://localhost:8000")
    res["docker_version"] = run(["docker", "--version"])["stdout_tail"].strip()
    checks, steps = {}, []
    steps.append(b := run(["docker", "compose", "build"], env=env, timeout=3600))
    checks["build"] = "PASS" if b["returncode"] == 0 else "FAIL"
    steps.append(u := run(["docker", "compose", "up", "-d"], env=env, timeout=600))
    checks["startup"] = "PASS" if u["returncode"] == 0 else "FAIL"
    base = "http://127.0.0.1:8000"
    health = None
    for _ in range(90):
        try:
            health = httpx.get(base + "/api/v1/health", timeout=3).json()
            if health.get("model_loaded"):
                break
        except (httpx.HTTPError, ValueError):
            time.sleep(2)
    checks["health"] = "PASS" if health and health.get("status") == "ok" and health.get("model_loaded") else "FAIL"
    res["health_response"] = health
    try:
        H = {"Authorization": f"Bearer {TOKEN}"}
        sid = httpx.get(base + "/api/v1/stations").json()[0]["station_id"]
        r = httpx.post(base + "/api/v1/observations", headers=H, json={"station_id": sid, "timestamp": "2024-09-01T00:00:00Z",
                                                                         "temperature_c": 27, "pressure_hpa": 1008, "relative_humidity_pct": 70})
        unauth = httpx.post(base + "/api/v1/observations", json={"station_id": sid, "timestamp": "2024-09-01T01:00:00Z"}).status_code
        checks["api_smoke"] = "PASS" if r.status_code == 200 and r.json()["accepted"] == 1 and unauth == 401 else "FAIL"
        page = httpx.get(base + "/")
        checks["dashboard_smoke"] = "PASS" if page.status_code == 200 and "SkyGuard" in page.text else "FAIL"
    except httpx.HTTPError as e:
        checks["api_smoke"] = checks["dashboard_smoke"] = f"FAIL ({e})"
    sc = run([sys.executable, "scripts/run_scenarios.py", "--base-url", base, "--token", TOKEN], timeout=3600)
    steps.append(sc)
    checks["scenarios"] = "PASS" if sc["returncode"] == 0 and "FAIL " not in sc["stdout_tail"] else "FAIL"
    steps.append(restart := run(["docker", "compose", "restart"], env=env, timeout=600))
    time.sleep(10)
    try:
        n = httpx.get(base + "/api/v1/metrics", timeout=10).json()["observations"]
        checks["persistence_after_restart"] = "PASS" if restart["returncode"] == 0 and n > 0 else "FAIL"
    except httpx.HTTPError:
        checks["persistence_after_restart"] = "FAIL"
    logs = run(["docker", "compose", "logs", "--no-color"], env=env)
    bad = [line for line in logs["stdout_tail"].splitlines() if "Traceback" in line or " ERROR " in line]
    checks["logs"] = "PASS" if not bad else f"REVIEW ({len(bad)} error lines)"
    steps.append(d := run(["docker", "compose", "down"], env=env, timeout=600))
    checks["shutdown"] = "PASS" if d["returncode"] == 0 else "FAIL"
    res.update(checks=checks, steps=steps, status="PASS" if all(v == "PASS" for v in checks.values()) else "FAIL")
    (OUT / "docker_verification.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("status", "docker_version", "checks")}, indent=2))


if __name__ == "__main__":
    main()

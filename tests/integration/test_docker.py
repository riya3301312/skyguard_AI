"""Docker checks. The static check always runs; the live container check needs a Docker engine and is
skipped (explicitly, with a reason) when Docker is not installed on the machine running the tests."""
import re
import shutil
import subprocess
import time

import httpx
import pytest

from ml.config import ROOT


def test_dockerfile_healthcheck_targets_existing_route():
    df = (ROOT / "Dockerfile").read_text()
    url = re.search(r"HEALTHCHECK.*?(http://[^'\"]+)", df).group(1)
    path = "/" + url.split("/", 3)[3]
    from backend.app import app
    assert path in {r.path for r in app.routes}
    assert "USER skyguard" in df and "EXPOSE 8000" in df
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "SKYGUARD_API_TOKEN" in compose and "skyguard-db:/app/data/db" in compose


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker engine not installed on this machine (environment-unverified)")
def test_docker_healthcheck():
    subprocess.run(["docker", "build", "-t", "skyguard-ai:test", "."], cwd=ROOT, check=True, timeout=3600)
    cid = subprocess.run(["docker", "run", "-d", "-p", "18000:8000", "-e", "SKYGUARD_API_TOKEN=t", "skyguard-ai:test"],
                         capture_output=True, text=True, check=True).stdout.strip()
    try:
        for _ in range(60):
            try:
                if httpx.get("http://127.0.0.1:18000/api/v1/health", timeout=2).json()["model_loaded"]:
                    break
            except (httpx.HTTPError, ValueError):
                time.sleep(2)
        else:
            pytest.fail("container did not become healthy")
    finally:
        subprocess.run(["docker", "rm", "-f", cid], check=False)

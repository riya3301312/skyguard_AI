"""Deterministic replay of a scenario CSV to a running API over HTTP (store-and-forward station clients).

python scripts/replay.py --input data/demo/sih_demo.csv --speed 3600
speed = observation-time seconds per wall-clock second (3600 -> one hourly record per second).
Warm-up rows (phase=warmup) are uploaded immediately in batches.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.replay import _obs, load_scenario  # noqa: E402
from edge.station_client import StationClient  # noqa: E402

try:
    from edge.edge_checks import EdgeChecker
    EdgeChecker()
except (FileNotFoundError, OSError):
    EdgeChecker = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/demo/sih_demo.csv")
    ap.add_argument("--speed", type=float, default=3600.0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("SKYGUARD_API_TOKEN", ""))
    ap.add_argument("--reset", action="store_true", help="clear server state first")
    a = ap.parse_args()
    cli = httpx.Client(base_url=a.base_url, timeout=120,
                       headers={"Authorization": f"Bearer {a.token}"} if a.token else {})
    if a.reset:
        cli.post("/api/v1/admin/reset").raise_for_status()

    def send(batch):
        r = cli.post("/api/v1/observations", json={"observations": batch})
        r.raise_for_status()
        return r.json()

    df = load_scenario(ROOT / a.input)
    warm = df[df["phase"] == "warmup"]
    for sid, g in warm.groupby("station_id"):
        recs = [_obs(r) for r in g.itertuples()]
        for i in range(0, len(recs), 500):
            send(recs[i:i + 500])
    print(f"warm-up uploaded: {len(warm)} records")
    live = df[df["phase"] != "warmup"]
    clients = {s: StationClient(s, send, edge=EdgeChecker() if EdgeChecker else None) for s in live["station_id"].unique()}
    prev, lat = None, []
    for ts, tick in live.groupby("timestamp", sort=True):
        if prev is not None:
            time.sleep((ts - prev).total_seconds() / a.speed)
        prev = ts
        for r in tick.itertuples():
            c = clients[r.station_id]
            if bool(r.link_up) != c.online:
                print(f"{ts} {r.station_id} link {'UP - syncing ' + str(len(c.buffer)) + ' buffered' if r.link_up else 'DOWN - buffering'}")
                c.set_link(bool(r.link_up))
            t0 = time.perf_counter()
            c.observe(_obs(r))
            lat.append((time.perf_counter() - t0) * 1000)
            if isinstance(r.note, str) and r.note:
                print(f"{ts} {r.note}")
    for c in clients.values():
        c.set_link(True)
    s = pd.Series(lat)
    print(f"live records: {len(live)}; client round-trip ms p50={s.quantile(.5):.1f} p95={s.quantile(.95):.1f}")


if __name__ == "__main__":
    main()

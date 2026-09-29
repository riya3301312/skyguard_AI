"""Deterministic replay of a scenario CSV through per-station StationClients (store-and-forward).

Scenario files live in data/demo/<name>.csv (only names matching [a-z0-9_] are accepted; no
arbitrary paths). Columns: timestamp, station_id, temperature_c, pressure_hpa,
relative_humidity_pct, phase ('warmup'|'live'), link_up (1/0), note.
Warm-up rows are delivered immediately; live rows are paced by observation time / speed.
"""
import re
import threading
from pathlib import Path

import pandas as pd

from edge.station_client import StationClient
from ml.config import ROOT

try:
    from edge.edge_checks import EdgeChecker
    EdgeChecker()
    EDGE_AVAILABLE = True
except (FileNotFoundError, OSError):
    EDGE_AVAILABLE = False


def load_scenario(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c, d in (("phase", "live"), ("link_up", 1), ("note", "")):
        if c not in df:
            df[c] = d
    df["note"] = df["note"].fillna("")
    return df.sort_values(["timestamp", "station_id"], kind="stable").reset_index(drop=True)


class ReplayRunner:
    def __init__(self, state, demo_dir: Path | None = None):
        self.S = state
        self.dir = Path(demo_dir or ROOT / "data" / "demo")
        self.thread: threading.Thread | None = None
        self.stop_ev = threading.Event()
        self._status = {"running": False, "scenario": None}

    def scenarios(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.csv")
                      if re.fullmatch(r"[a-z0-9_]{1,64}", p.stem) and not p.stem.endswith("_truth"))  # truth = answer key

    def status(self) -> dict:
        return dict(self._status)

    def start(self, name: str, speed: float, reset: bool) -> dict:
        self.stop()
        if reset:
            self.S.reset()
        df = load_scenario(self.dir / f"{name}.csv")
        self.stop_ev.clear()
        self._status = {"running": True, "scenario": name, "speed": speed, "phase": "starting", "progress": 0.0,
                        "rows": int(len(df)), "stream_time": None, "edge": "C library via ctypes" if EDGE_AVAILABLE else "not built",
                        "links": {}, "buffered": {}, "note": ""}
        self.thread = threading.Thread(target=self._run, args=(df, speed), daemon=True)
        self.thread.start()
        return self.status()

    def stop(self) -> dict:
        self.stop_ev.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        self._status["running"] = False
        return self.status()

    def _run(self, df: pd.DataFrame, speed: float) -> None:
        S = self.S
        try:
            warm = df[df["phase"] == "warmup"]
            live = df[df["phase"] != "warmup"]
            self._status["phase"] = "warmup"
            S.publish("replay", self._status)
            n_warm = max(1, warm["station_id"].nunique())
            for k, (sid, g) in enumerate(warm.groupby("station_id")):
                recs = [_obs(r) for r in g.itertuples()]
                for i in range(0, len(recs), 200):
                    if self.stop_ev.is_set():
                        return
                    S.ingest(recs[i:i + 200], quiet=True)
                self._status["progress"] = round(0.5 * (k + 1) / n_warm, 3)
                S.publish("replay", self._status)
            edge = {}
            clients = {}
            for sid in live["station_id"].unique():
                edge[sid] = EdgeChecker() if EDGE_AVAILABLE else None
                clients[sid] = StationClient(sid, send=lambda batch: S.ingest(batch), edge=edge[sid])
            self._status["phase"] = "live"
            prev = None
            n = len(live)
            for i, (ts, tick) in enumerate(live.groupby("timestamp", sort=True)):
                if prev is not None:
                    wait = (ts - prev).total_seconds() / speed
                    if self.stop_ev.wait(wait):
                        return
                prev = ts
                for r in tick.itertuples():
                    c = clients[r.station_id]
                    up = bool(r.link_up)
                    if up != c.online:
                        S.publish("link", {"station_id": r.station_id, "online": up, "buffered": len(c.buffer),
                                           "stream_time": ts.isoformat()})
                        c.set_link(up)
                    c.observe(_obs(r))
                    if r.note:
                        self._status["note"] = r.note
                self._status.update(progress=round(0.5 + 0.5 * tick.index[-1] / max(n, 1) if n else 1.0, 3),
                                    stream_time=ts.isoformat(),
                                    links={s: c.online for s, c in clients.items()},
                                    buffered={s: len(c.buffer) for s, c in clients.items()})
                S.publish("replay", self._status)
            for c in clients.values():  # end of scenario: every station reconnects and syncs
                c.set_link(True)
            self._status.update(buffered={s: len(c.buffer) for s, c in clients.items()},
                                client_stats={s: c.stats for s, c in clients.items()})
        except Exception as e:  # surface failures instead of dying silently
            self._status["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._status["running"] = False
            self._status["phase"] = "finished" if not self.stop_ev.is_set() else "stopped"
            S.publish("replay", self._status)


def _obs(r) -> dict:
    def f(x):
        return None if pd.isna(x) else float(x)
    return {"station_id": r.station_id, "timestamp": r.timestamp.isoformat(), "temperature_c": f(r.temperature_c),
            "pressure_hpa": f(r.pressure_hpa), "relative_humidity_pct": f(r.relative_humidity_pct), "source": "replay"}

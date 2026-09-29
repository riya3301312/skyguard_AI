"""Remote-station uplink: store-and-forward buffer (offline -> buffer -> reconnect -> upload).

`send(batch)` is any callable that delivers a list of observation dicts and raises on
failure (HTTP in scripts/replay.py, in-process in the backend replay). Records are removed
from the buffer only after a successful send, and the server treats duplicates as
idempotent, so retries never lose or double-count data. The buffer is persisted to a
JSONL file when `buffer_path` is given, so it also survives a station restart.
"""
import json
from collections.abc import Callable
from pathlib import Path


class StationClient:
    def __init__(self, station_id: str, send: Callable[[list[dict]], object], buffer_path: str | Path | None = None,
                 max_batch: int = 500, edge=None):
        self.station_id = station_id
        self.send = send
        self.max_batch = max_batch
        self.edge = edge  # optional edge checker with .check(obs) -> dict (see edge/edge_checks.py)
        self.online = True
        self.path = Path(buffer_path) if buffer_path else None
        self.buffer: list[dict] = []
        if self.path and self.path.exists():
            self.buffer = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        self.stats = {"observed": 0, "sent": 0, "send_failures": 0, "max_buffered": len(self.buffer), "batches": 0}

    def _persist(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("".join(json.dumps(o) + "\n" for o in self.buffer))

    def observe(self, obs: dict) -> None:
        obs = dict(obs, station_id=self.station_id)
        if self.edge is not None:
            obs.update(self.edge.check(obs))
        self.buffer.append(obs)
        self.stats["observed"] += 1
        self.stats["max_buffered"] = max(self.stats["max_buffered"], len(self.buffer))
        self._persist()
        if self.online:
            self.flush()

    def set_link(self, up: bool) -> None:
        self.online = up
        if up:
            self.flush()

    def flush(self) -> int:
        """Upload buffered records in observation order. Returns records delivered."""
        sent = 0
        while self.buffer and self.online:
            batch = self.buffer[: self.max_batch]
            try:
                self.send(batch)
            except Exception:  # link dropped mid-upload: keep everything, try again later
                self.stats["send_failures"] += 1
                self.online = False
                break
            del self.buffer[: len(batch)]
            sent += len(batch)
            self.stats["sent"] += len(batch)
            self.stats["batches"] += 1
            self._persist()
        return sent

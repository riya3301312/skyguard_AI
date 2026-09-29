import pytest

from edge.station_client import StationClient

try:
    from edge.edge_checks import EdgeChecker
    EdgeChecker()
    HAVE_EDGE = True
except (FileNotFoundError, OSError):
    HAVE_EDGE = False


def obs(i, t=25.0, p=1008.0, rh=60.0):
    return {"timestamp": f"2024-01-01T{i:02d}:00:00Z", "temperature_c": t, "pressure_hpa": p, "relative_humidity_pct": rh}


@pytest.mark.skipif(not HAVE_EDGE, reason="edge C library not built (python edge/build_edge.py)")
def test_edge_c_checks():
    e = EdgeChecker()
    assert e.check(obs(0))["edge_flags"] == []
    assert "T_SPIKE" in e.check(obs(1, t=40.0))["edge_flags"]
    assert "P_RANGE" in e.check(obs(2, p=-9999.0))["edge_flags"]
    assert "RH_MISSING" in e.check(obs(3, rh=None))["edge_flags"]
    e2 = EdgeChecker()
    flags = [e2.check(obs(i, t=26.0 + (i % 2), rh=61.0))["edge_flags"] for i in range(12)]
    assert any("RH_FLATLINE" in f for f in flags) and any("P_FLATLINE" in f for f in flags)
    e3 = EdgeChecker()
    e3.check(obs(0, t=25.0, rh=60.0))
    assert "T_RH_INCONSISTENT" in e3.check(obs(1, t=30.0, rh=60.0))["edge_flags"]
    assert 0 <= e3.check(obs(2))["edge_health_score"] <= 100


def test_station_client_buffers_offline_and_syncs_without_loss(tmp_path):
    delivered, fail = [], {"on": False}

    def send(batch):
        if fail["on"]:
            raise ConnectionError("link down")
        delivered.extend(batch)

    c = StationClient("S1", send, buffer_path=tmp_path / "buf.jsonl", max_batch=3)
    c.observe(obs(0))
    c.set_link(False)
    for i in range(1, 8):
        c.observe(obs(i))
    assert len(delivered) == 1 and len(c.buffer) == 7
    # station restart while offline: buffer survives on disk
    c2 = StationClient("S1", send, buffer_path=tmp_path / "buf.jsonl", max_batch=3)
    assert len(c2.buffer) == 7
    fail["on"] = True
    c2.set_link(True)  # link flaps mid-upload: nothing lost
    assert not c2.online and len(c2.buffer) == 7
    fail["on"] = False
    c2.set_link(True)
    assert len(c2.buffer) == 0 and [d["timestamp"] for d in delivered] == [obs(i)["timestamp"] for i in range(8)]
    assert c2.stats["batches"] == 3

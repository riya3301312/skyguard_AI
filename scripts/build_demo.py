"""Build the deterministic SIH demo replay (data/demo/sih_demo.csv) from real processed observations.

The story (hourly observations, live phase ~72 h after an 8-day warm-up):
  h12      temperature spike 55.2 degC at one station (isolated)              -> SPIKE + estimate
  h20-25   regional weather event across the whole cluster (T drop, RH up)   -> NOT a sensor fault
  h22-23   one cluster station also has its own +9 hPa pressure fault        -> flagged as station exception
  h28-33   isolated smooth warming at one station, physically consistent     -> spatial inconsistency
  h36-50   frozen temperature at one station                                 -> FLATLINE
  h30-72   slow pressure drift at one station                                -> DRIFT, maintenance risk
  h40-46   remote island station link down (store-and-forward)               -> offline, buffer, reconnect, sync
  h52-55   records never transmitted (no buffer)                             -> MISSING_COMMUNICATION gap
  h60      sentinel -9999 pressure                                           -> SENTINEL_CORRUPTION
Injected values are recorded in data/demo/sih_demo_truth.csv; raw real data comes from clean.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.config import load_config  # noqa: E402
from ml.features import dewpoint, rh_from  # noqa: E402
from ml.schema import VARS, read_canonical  # noqa: E402


def pick_cluster(stations: dict, present: set) -> list[str]:
    anchor = max(sorted(present), key=lambda s: len([n for n in stations[s]["neighbors"] if n["station_id"] in present]))
    return [anchor] + [n["station_id"] for n in stations[anchor]["neighbors"] if n["station_id"] in present][:4]


def main() -> None:
    cfg = load_config()
    stations = json.loads((ROOT / "data" / "processed" / "stations.json").read_text())["stations"]
    clean = read_canonical(ROOT / "data" / "processed" / "clean.csv")
    clean = clean[clean["timestamp"] >= pd.Timestamp(cfg["split"]["val_end"], tz="UTC")]  # demo uses the test period only
    present = set(clean["station_id"])
    cluster = pick_cluster(stations, present)
    remote = min((s for s in present if not stations[s]["neighbors"]), key=lambda s: stations[s]["latitude"])  # southern-most isolated
    ids = cluster + [remote]
    live_h, warm_d = 72, cfg["injection"]["warmup_days"]
    piv = clean[clean["station_id"].isin(ids)].pivot(index="timestamp", columns="station_id", values="temperature_c")
    ok = piv.notna().all(axis=1).rolling(24 * warm_d + live_h).sum() == 24 * warm_d + live_h
    end = ok[ok].index[0]
    start = end - pd.Timedelta(hours=24 * warm_d + live_h - 1)
    d = clean[clean["station_id"].isin(ids) & clean["timestamp"].between(start, end)].copy()
    d = d.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    t_live = start + pd.Timedelta(days=warm_d)
    d["phase"] = np.where(d["timestamp"] < t_live, "warmup", "live")
    d["link_up"] = 1
    d["note"] = ""
    d["hour"] = ((d["timestamp"] - t_live).dt.total_seconds() // 3600).astype(int)
    truth = []

    def at(sid, h0, h1=None):
        return (d["station_id"] == sid) & d["hour"].between(h0, h1 if h1 is not None else h0)

    def note(sid, h, text):
        d.loc[at(sid, h), "note"] = text

    a, b, c, e = cluster[0], cluster[1], cluster[2], cluster[3]
    f = cluster[4] if len(cluster) > 4 else cluster[1]
    # h12 spike at the anchor station
    m = at(a, 12)
    truth.append({"hour": 12, "station_id": a, "kind": "spike", "parameter": "temperature_c", "raw_injected": 55.2,
                  "clean_value": float(d.loc[m, "temperature_c"].iloc[0])})
    d.loc[m, "temperature_c"] = 55.2
    note(a, 12, f"Injected fault: 55.2 °C temperature spike at {a}")
    # h20-25 regional event on the whole cluster (plausible physics: T down, RH up via dew point, P up)
    shape = np.array([0.5, 1.0, 0.9, 0.6, 0.35, 0.15])
    for sid in cluster:
        m = at(sid, 20, 25)
        t0 = d.loc[m, "temperature_c"].to_numpy(float)
        td = dewpoint(pd.Series(t0), pd.Series(d.loc[m, "relative_humidity_pct"].to_numpy(float))).to_numpy()
        dt = -7.0 * shape[: m.sum()]
        d.loc[m, "temperature_c"] = np.round(t0 + dt)
        d.loc[m, "relative_humidity_pct"] = np.round(np.clip(rh_from(pd.Series(np.round(t0 + dt)), pd.Series(td + 1.0)).to_numpy(), 1, 100), 1)
        d.loc[m, "pressure_hpa"] = np.round(d.loc[m, "pressure_hpa"].to_numpy(float) + 2.0 * shape[: m.sum()], 1)
    truth.append({"hour": 20, "station_id": ",".join(cluster), "kind": "regional_weather_event", "parameter": "all"})
    note(a, 20, "Regional weather event: every station in the cluster cools together")
    # h22-23: during the regional event, one station ALSO has its own pressure fault (station exception)
    exc = cluster[-1]
    m = at(exc, 22, 23)
    d.loc[m, "pressure_hpa"] = np.round(d.loc[m, "pressure_hpa"].to_numpy(float) + 9.0, 1)
    truth.append({"hour": 22, "station_id": exc, "kind": "regional_with_station_exception", "parameter": "pressure_hpa"})
    note(exc, 22, f"During the regional event, {exc} also shows a +9 hPa pressure fault of its own")
    # h28-33 isolated physically-consistent warming at station b
    m = at(b, 28, 33)
    t0 = d.loc[m, "temperature_c"].to_numpy(float)
    td = dewpoint(pd.Series(t0), pd.Series(d.loc[m, "relative_humidity_pct"].to_numpy(float))).to_numpy()
    dt = np.array([2, 4, 6, 6, 6, 5.0])[: m.sum()]
    d.loc[m, "temperature_c"] = np.round(t0 + dt)
    d.loc[m, "relative_humidity_pct"] = np.round(np.clip(rh_from(pd.Series(np.round(t0 + dt)), pd.Series(td)).to_numpy(), 1, 100), 1)
    truth.append({"hour": 28, "station_id": b, "kind": "isolated_event", "parameter": "temperature_c"})
    note(b, 28, f"Isolated change: only {b} warms; neighbours stay stable")
    # h36-50 flatline at c
    m = at(c, 36, 50)
    d.loc[m, "temperature_c"] = float(d.loc[at(c, 35), "temperature_c"].iloc[0])
    truth.append({"hour": 36, "station_id": c, "kind": "flatline", "parameter": "temperature_c"})
    note(c, 36, f"Injected fault: frozen temperature sensor at {c}")
    # h30-72 slow pressure drift at e
    m = at(e, 30, live_h)
    d.loc[m, "pressure_hpa"] = np.round(d.loc[m, "pressure_hpa"].to_numpy(float) + np.linspace(0.2, 9.0, m.sum()), 1)
    truth.append({"hour": 30, "station_id": e, "kind": "drift", "parameter": "pressure_hpa"})
    note(e, 30, f"Injected fault: slow pressure drift begins at {e}")
    # h40-46 remote station offline with store-and-forward
    d.loc[at(remote, 40, 46), "link_up"] = 0
    truth.append({"hour": 40, "station_id": remote, "kind": "offline_buffered", "parameter": "all"})
    note(remote, 40, f"Remote station {remote} loses its link: records are buffered on-station")
    note(remote, 47, f"{remote} reconnects: buffered records upload as a batch")
    # h52-55 records never transmitted by f
    d = d[~at(f, 52, 55)].reset_index(drop=True)
    truth.append({"hour": 52, "station_id": f, "kind": "communication_gap", "parameter": "all"})
    note(f, 56, f"{f}: 4 hourly records were never transmitted (no buffer)")
    # h60 sentinel
    m = at(f, 60)
    d.loc[m, "pressure_hpa"] = -9999.0
    truth.append({"hour": 60, "station_id": f, "kind": "sentinel", "parameter": "pressure_hpa"})
    note(f, 60, f"Injected fault: sentinel -9999 pressure at {f}")

    out = ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", "station_id", *VARS, "phase", "link_up", "note"]
    d[cols].to_csv(out / "sih_demo.csv", index=False)
    pd.DataFrame(truth).assign(t_live_start=t_live.isoformat()).to_csv(out / "sih_demo_truth.csv", index=False)
    print(f"sih_demo.csv: {len(d)} rows, stations={ids}, live from {t_live} for {live_h} h")


if __name__ == "__main__":
    main()

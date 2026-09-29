"""Execute SIH scenarios A-J against a RUNNING SkyGuard API and record observed evidence.

python scripts/run_scenarios.py --base-url http://127.0.0.1:8000

Replays data/demo/sih_demo.csv over HTTP through per-station StationClients (store-and-forward,
edge checks via the C library), probes station status during the outage, then checks each
scenario's expectation against what the API actually returned. Writes artifacts/scenarios/.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.replay import load_scenario  # noqa: E402
from edge.station_client import StationClient  # noqa: E402
from ml.config import load_config  # noqa: E402

try:
    from edge.edge_checks import EdgeChecker
    EdgeChecker()
except (FileNotFoundError, OSError):
    EdgeChecker = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("SKYGUARD_API_TOKEN", ""))
    ap.add_argument("--scenario", default="sih_demo")
    args = ap.parse_args()
    H = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    cli = httpx.Client(base_url=args.base_url, headers=H, timeout=120)
    health = cli.get("/api/v1/health").json()
    cli.post("/api/v1/admin/reset").raise_for_status()

    df = load_scenario(ROOT / "data" / "demo" / f"{args.scenario}.csv")
    truth = pd.read_csv(ROOT / "data" / "demo" / f"{args.scenario}_truth.csv")
    t_live = pd.Timestamp(truth["t_live_start"].iloc[0])
    hour = lambda ts: int((pd.Timestamp(ts) - t_live).total_seconds() // 3600)  # noqa: E731

    sent_log, results = [], {}

    def sender(batch):
        r = cli.post("/api/v1/observations", json={"observations": batch})
        r.raise_for_status()
        body = r.json()
        sent_log.append({"n": len(batch), "accepted": body["accepted"], "duplicates": body["duplicates"]})
        for x in body["results"]:
            results[(x["station_id"], x["observation_time"])] = x
        return body

    # warm-up (fast), then live through StationClients
    warm = df[df["phase"] == "warmup"]
    for sid, g in warm.groupby("station_id"):
        recs = [{"station_id": sid, "timestamp": r.timestamp.isoformat(), "temperature_c": _f(r.temperature_c),
                 "pressure_hpa": _f(r.pressure_hpa), "relative_humidity_pct": _f(r.relative_humidity_pct)} for r in g.itertuples()]
        for i in range(0, len(recs), 500):
            sender(recs[i:i + 500])
    live = df[df["phase"] == "live"]
    clients = {s: StationClient(s, sender, buffer_path=ROOT / "artifacts" / "scenarios" / f"buffer_{s}.jsonl",
                                edge=EdgeChecker() if EdgeChecker else None) for s in live["station_id"].unique()}
    for c in clients.values():
        c.buffer.clear()
        c._persist()
    outage_probe, max_buffer = {}, {}
    t_start = time.perf_counter()
    for ts, tick in live.groupby("timestamp", sort=True):
        for r in tick.itertuples():
            c = clients[r.station_id]
            if bool(r.link_up) != c.online:
                c.set_link(bool(r.link_up))
            c.observe({"timestamp": r.timestamp.isoformat(), "temperature_c": _f(r.temperature_c),
                       "pressure_hpa": _f(r.pressure_hpa), "relative_humidity_pct": _f(r.relative_humidity_pct)})
            max_buffer[r.station_id] = max(max_buffer.get(r.station_id, 0), len(c.buffer))
        h = hour(ts)
        if h == 44:  # remote station has been offline 4 h (stream clock) -> probe status + buffer
            st = {s["station_id"]: s for s in cli.get("/api/v1/stations").json()}
            off = truth.loc[truth["kind"] == "offline_buffered", "station_id"].iloc[0]
            outage_probe = {"station": off, "status_during_outage": st[off]["status"], "label": st[off]["label"],
                            "buffered_on_station": len(clients[off].buffer)}
    for c in clients.values():
        c.set_link(True)
    live_wall = time.perf_counter() - t_start

    anomalies = cli.get("/api/v1/anomalies", params={"limit": 2000}).json()
    stations = {s["station_id"]: s for s in cli.get("/api/v1/stations").json()}
    metrics = cli.get("/api/v1/metrics").json()
    A = pd.DataFrame(anomalies)
    A["hour"] = A["observation_time"].map(hour)
    # judge the FINAL stored results (provisional results may have been revised once neighbour data arrived)
    stored = pd.concat([pd.DataFrame(cli.get("/api/v1/observations/history", params={"station_id": s, "limit": 5000}).json())
                        for s in df["station_id"].unique()], ignore_index=True)
    revised = int(sum((results[(r.station_id, r.observation_time)]["is_anomaly"] != r.is_anomaly)
                      for r in stored.itertuples() if (r.station_id, r.observation_time) in results))
    for r in stored.to_dict("records"):
        key = (r["station_id"], r["observation_time"])
        if key in results:
            results[key] = {**results[key], **{k: r[k] for k in ("is_anomaly", "anomaly_probability", "fault_pattern", "parameter",
                                                                  "severity", "regional_evidence", "regional_suppressed",
                                                                  "estimated_temperature_c", "replacement_recommended")}}
            ex = A[(A["station_id"] == key[0]) & (A["observation_time"] == key[1]) & A["state"].isin(["OPEN", "ACKNOWLEDGED"])]
            if len(ex):
                results[key]["explanation"] = ex.iloc[0]["explanation"]
    R = pd.DataFrame(results.values())
    R["hour"] = R["observation_time"].map(hour)
    live_R = R[R["hour"] >= 0]

    def alerts(sid, h0, h1, pattern=None):
        x = A[(A["station_id"] == sid) & A["hour"].between(h0, h1) & (A["state"] != "INFO")]
        return x if pattern is None else x[x["fault_pattern"].isin(pattern if isinstance(pattern, list) else [pattern])]

    T = {r.kind: r for r in truth.itertuples()}
    cluster = T["regional_weather_event"].station_id.split(",")
    remote = T["offline_buffered"].station_id
    out = {}

    # A - clean station: remote island station outside its outage and cluster stations before any event
    clean_rows = live_R[((live_R["station_id"] == remote) & ~live_R["hour"].between(40, 47)) |
                        (live_R["station_id"].isin(cluster) & live_R["hour"].between(0, 11))]
    fa = clean_rows[clean_rows["is_anomaly"]]
    out["A_clean_station"] = _res(len(fa) == 0, {"clean_observations": int(len(clean_rows)), "false_alerts": int(len(fa)),
                                                 "false_alert_details": fa[["station_id", "observation_time", "fault_pattern", "anomaly_probability"]].to_dict("records")},
                                  "normal, no false alert")
    # B - temperature spike
    b = T["spike"]
    rb = results.get((b.station_id, (t_live + pd.Timedelta(hours=12)).isoformat()))
    ok = bool(rb and rb["is_anomaly"] and rb["fault_pattern"] in ("SPIKE", "COMPOSITE_FAULT") and rb["parameter"] == "temperature_c"
              and (rb["anomaly_probability"] or 0) >= 0.9 and rb["explanation"] and rb["explanation"]["evidence"])
    out["B_temperature_spike"] = _res(ok, {k: rb.get(k) for k in ("temperature_c", "is_anomaly", "fault_pattern", "parameter", "anomaly_probability",
                                                                  "severity", "estimated_temperature_c", "replacement_recommended", "edge_flags")} |
                                      {"explanation": rb["explanation"], "clean_value": b.clean_value} if rb else {},
                                      "anomaly, high probability, fault pattern = spike, human-readable explanation")
    # C - flatline
    c = T["flatline"]
    fl = alerts(c.station_id, 36, 52, "FLATLINE")
    allc = alerts(c.station_id, 36, 52)
    out["C_flatline"] = _res(len(fl) > 0, {"flatline_alerts": int(len(fl)), "all_alerts_in_window": allc["fault_pattern"].value_counts().to_dict(),
                                           "first_flatline_hour": int(fl["hour"].min()) if len(fl) else None},
                             "flatline/frozen pattern, not merely generic anomaly")
    # D - drift
    d = T["drift"]
    dr = alerts(d.station_id, 30, 80, ["DRIFT", "STEP_OFFSET"])
    maint = stations[d.station_id]["maintenance"]
    out["D_gradual_drift"] = _res(len(dr) > 0, {"drift_or_offset_alerts": int(len(dr)), "first_hour": int(dr["hour"].min()) if len(dr) else None,
                                                "patterns": alerts(d.station_id, 30, 80)["fault_pattern"].value_counts().to_dict(),
                                                "maintenance": maint, "health": stations[d.station_id]["health_score"]},
                                  "drift detected; maintenance risk may increase")
    # E - communication failure (records never transmitted)
    e = T["communication_gap"]
    gap = alerts(e.station_id, 50, 57, "MISSING_COMMUNICATION")
    out["E_communication_failure"] = _res(len(gap) > 0, {"gap_alerts": gap[["observation_time", "explanation"]].to_dict("records")},
                                          "missing/communication pattern detected")
    # F - regional weather event
    reg = R[R["station_id"].isin(cluster) & R["hour"].between(20, 25)]
    sensor_alerts = reg[reg["is_anomaly"]]
    info = A[(A["state"] == "INFO") & A["hour"].between(20, 25)]
    out["F_regional_weather_event"] = _res(
        len(sensor_alerts) <= 1 and bool(reg["regional_evidence"].any()),
        {"cluster_observations": int(len(reg)), "sensor_fault_alerts": int(len(sensor_alerts)),
         "alerts": sensor_alerts[["station_id", "hour", "fault_pattern", "anomaly_probability"]].to_dict("records"),
         "rows_with_regional_evidence": int(reg["regional_evidence"].sum()), "suppressed_regional_alerts_logged": int(len(info)),
         "max_probability": float(reg["anomaly_probability"].max())},
        "regional event evidence; reduced sensor-fault false alarm")
    # G - isolated station event
    g = T["isolated_event"]
    iso = R[(R["station_id"] == g.station_id) & R["hour"].between(28, 33)]
    iso_alerts = iso[iso["is_anomaly"]]
    out["G_isolated_station_event"] = _res(
        len(iso_alerts) > 0 and float(iso["anomaly_probability"].max()) > float(reg["anomaly_probability"].max()),
        {"alerts": iso_alerts[["hour", "fault_pattern", "anomaly_probability", "severity"]].to_dict("records"),
         "max_probability_isolated": float(iso["anomaly_probability"].max()), "max_probability_regional": float(reg["anomaly_probability"].max()),
         "evidence_example": iso_alerts["explanation"].iloc[0]["evidence"] if len(iso_alerts) else None},
        "spatial inconsistency; higher sensor-fault likelihood than the regional event")
    # H - held-out station (from the frozen final evaluation; this station is also held out and replayed here)
    fe = ROOT / "artifacts" / "final_evaluation" / "metrics.json"
    ho = json.loads(fe.read_text())["held_out"] if fe.exists() else None
    out["H_held_out_station"] = _res(bool(ho) and ho["held_out_stations"]["f1"] >= 0.7 * ho["seen_stations"]["f1"],
                                     {"final_eval_held_out": ho and {k: ho["held_out_stations"][k] for k in ("precision", "recall", "f1", "pr_auc", "false_alerts_per_station_day")},
                                      "final_eval_seen": ho and {k: ho["seen_stations"][k] for k in ("precision", "recall", "f1", "pr_auc", "false_alerts_per_station_day")},
                                      "criterion": "held-out F1 >= 70% of seen-station F1"},
                                     "reasonable generalisation")
    # I - offline remote station
    sent = R[(R["station_id"] == remote) & R["hour"].between(40, 46)]
    expected = int(((live["station_id"] == remote) & live["timestamp"].map(hour).between(40, 46)).sum())
    late = sent["lag_minutes"] > load_config()["qc"]["delayed_minutes"]
    out["I_offline_remote_station"] = _res(
        outage_probe.get("status_during_outage") == "GRAY" and len(sent) == expected and bool(sent.loc[late, "delayed"].all())
        and int(sent["is_anomaly"].sum()) == 0,
        {"probe": outage_probe, "records_generated_during_outage": expected, "records_received_after_reconnect": int(len(sent)),
         "lag_minutes": sent["lag_minutes"].tolist(), "records_older_than_delay_threshold_marked_delayed": bool(sent.loc[late, "delayed"].all()),
         "alerts_on_buffered_records": int(sent["is_anomaly"].sum()),
         "max_buffer": max_buffer.get(remote), "client_stats": clients[remote].stats},
        "buffer -> reconnect -> sync, no data loss")
    # J - edge
    em = json.loads((ROOT / "artifacts" / "edge" / "edge_metrics.json").read_text())
    rb_flags = (rb or {}).get("edge_flags") or []
    sent_rec = results.get((T["sentinel"].station_id, (t_live + pd.Timedelta(hours=60)).isoformat()), {})
    out["J_edge"] = _res(EdgeChecker is not None and "T_SPIKE" in rb_flags and "P_RANGE" in (sent_rec.get("edge_flags") or []) and em["within_budget"],
                         {"mode": "faithful simulation: same C source compiled for host and called via ctypes; not run on ESP32 hardware",
                          "spike_edge_flags": rb_flags, "sentinel_edge_flags": sent_rec.get("edge_flags"),
                          "edge_metrics": {k: em[k] for k in ("flash_bytes_rv32imc_object", "ram_bytes", "host_runtime", "budget", "within_budget")}},
                         "lightweight checks execute within reported resource budget")

    summary = {"provisional_results_revised_after_neighbour_data": revised, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "api": args.base_url, "detector_mode": health["detector_mode"],
               "model_version": health["model_version"], "live_replay_wall_seconds": round(live_wall, 2),
               "observations_processed": metrics["observations"], "processing_latency_ms": metrics["processing_latency_ms"],
               "scenarios": out, "passed": sum(v["passed"] for v in out.values()), "total": len(out)}
    dest = ROOT / "artifacts" / "scenarios"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "scenario_results.json").write_text(json.dumps(summary, indent=2, default=str))
    for k, v in out.items():
        print(f"{'PASS' if v['passed'] else 'FAIL'}  {k}: expected {v['expected']}")
    print(f"{summary['passed']}/{summary['total']} scenarios passed -> artifacts/scenarios/scenario_results.json")


def _f(x):
    return None if pd.isna(x) else float(x)


def _res(passed: bool, evidence: dict, expected: str) -> dict:
    return {"passed": bool(passed), "expected": expected, "evidence": evidence}


if __name__ == "__main__":
    main()

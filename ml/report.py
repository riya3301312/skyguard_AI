"""Evaluation runner used by scripts/evaluate.py (validation) and scripts/evaluate_final.py (frozen test)."""
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .benchmarking import APPROACH_NAMES, baseline_scores, compare, hybrid_scores  # noqa: E402
from .config import ROOT, load_config  # noqa: E402
from .evaluation import (calibration, classification, detection, eval_mask, event_recall, imputation,  # noqa: E402
                         latency_stats)
from .features import station_features  # noqa: E402
from .models import FEATURE_SETS, fixed_threshold_score, robust_stat_score  # noqa: E402
from .pipeline import ModelBundle, features_for, load_stations, score_features  # noqa: E402
from .qc import qc_frame  # noqa: E402
from .schema import VARS, read_canonical  # noqa: E402
from .stream import StreamEngine  # noqa: E402


def weather_vs_sensor(R: pd.DataFrame) -> dict:
    """Weather-vs-sensor discrimination on the evaluated rows (injected regional events are labelled normal)."""
    reg = R["injection_kind"] == "regional_event"
    status = R["weather_status"]
    regional_status = status.isin(["REGIONAL_WEATHER", "UNCERTAIN"])
    ev = R[reg].groupby("injection_id").agg(any_alert=("flag", "any"), recognised=("weather_status", lambda x: (x == "REGIONAL_WEATHER").any()))
    exc = R[status == "REGIONAL_WEATHER_WITH_STATION_EXCEPTION"]
    examples = [{"station_id": r.station_id, "timestamp": str(r.timestamp), "pattern": r.pattern, "parameter": r.param,
                 "probability": round(float(r.prob), 3), "reason": r.exception_reason, "truth": r.anomaly_type}
                for r in exc.head(5).itertuples()]
    return {
        "status_counts": status.value_counts().to_dict(),
        "regional_event_instances": int(len(ev)),
        "regional_event_instances_with_any_alert": int(ev["any_alert"].sum()),
        "regional_event_instances_recognised_as_regional_weather": int(ev["recognised"].sum()),
        "regional_event_rows": int(reg.sum()),
        "regional_event_rows_status_regional": int((reg & regional_status).sum()),
        "rows_status_regional_weather": int((status == "REGIONAL_WEATHER").sum()),
        "rows_status_regional_weather_inside_injected_events": int(((status == "REGIONAL_WEATHER") & reg).sum()),
        "note": "Precision of the REGIONAL_WEATHER status against injected events is a lower bound: genuine regional weather "
                "in the real observations is unlabelled.",
        "station_exception_rows": int(len(exc)), "station_exception_rows_true_anomalies": int((exc["is_anomaly"] == 1).sum()),
        "station_exception_examples": examples,
        "sensor_anomaly_rows": int((status == "SENSOR_ANOMALY").sum()), "uncertain_rows": int((status == "UNCERTAIN").sum()),
    }


def verify_frozen(dataset: Path) -> dict:
    man = json.loads((dataset / "MANIFEST.json").read_text())
    digest = hashlib.sha256(gzip.decompress((dataset / "benchmark.csv.gz").read_bytes())).hexdigest()
    return {"frozen": bool(man.get("frozen")), "frozen_at": man.get("frozen_at"), "manifest_sha256": man["sha256_uncompressed_csv"],
            "file_sha256": digest, "integrity_ok": digest == man["sha256_uncompressed_csv"], "manifest": man}


# ------------------------------------------------------------------------------------------ latency
def _stream_rows(df: pd.DataFrame, stations: list[str], days: int) -> pd.DataFrame:
    d = df[df["station_id"].isin(stations)].sort_values(["timestamp", "station_id"])
    t_end = d["timestamp"].min() + pd.Timedelta(days=8 + days)
    return d[d["timestamp"] < t_end]


def measure_hybrid_stream(bundle: ModelBundle | None, df: pd.DataFrame, stations_meta: dict, stations: list[str], days: int = 4) -> dict:
    """Replay real rows through the real-time engine; latency is measured only after an 8-day warm-up."""
    rows = _stream_rows(df, stations, days)
    eng = StreamEngine({s: stations_meta[s] for s in stations}, bundle)
    warm_end = rows["timestamp"].min() + pd.Timedelta(days=8)
    proc = psutil.Process()
    lat, n_live, t0 = [], 0, None
    rss0 = proc.memory_info().rss
    proc.cpu_percent(interval=None)
    for r in rows.itertuples():
        obs = {"station_id": r.station_id, "timestamp": r.timestamp.isoformat(),
               **{v: (None if pd.isna(getattr(r, v)) else float(getattr(r, v))) for v in VARS}}
        if all(obs[v] is None for v in VARS):
            continue  # absent record in a live stream
        if t0 is None and r.timestamp >= warm_end:
            t0 = time.perf_counter()
        out = eng.process(obs, ingestion_time=r.timestamp.to_pydatetime())
        if r.timestamp >= warm_end and out["status"] == "accepted":
            lat.append(out["processing_latency_ms"])
            n_live += 1
    wall = time.perf_counter() - (t0 or time.perf_counter())
    return {"latency": latency_stats(np.array(lat)), "samples_ms": lat, "throughput_obs_per_s": n_live / max(wall, 1e-9),
            "cpu_percent_during_run": proc.cpu_percent(interval=None), "rss_mb": proc.memory_info().rss / 2**20,
            "rss_growth_mb": (proc.memory_info().rss - rss0) / 2**20, "stations": stations, "live_observations": n_live,
            "includes": "QC + temporal/multivariate/spatial features + autoencoder + calibrated detector + pattern/param "
                        "classifiers + severity/health/correction + explanation (for flagged rows)"}


def measure_baseline_latency(bundle: ModelBundle, df: pd.DataFrame, station: str, n: int = 150) -> dict:
    """Per-observation latency of each baseline: its own per-station computation on a 10-day buffer."""
    s = df[df["station_id"] == station].sort_values("timestamp").set_index("timestamp")[VARS]
    start = s.index.min() + pd.Timedelta(days=10)
    idx = s.index[s.index >= start][:n]
    out = {}
    for name in ("fixed_threshold", "robust_statistical", "isolation_forest", "temporal_autoencoder"):
        ms = []
        for t in idx:
            buf = s[(s.index <= t) & (s.index > t - pd.Timedelta(days=10))]
            t0 = time.perf_counter()
            if name == "fixed_threshold":
                fixed_threshold_score(qc_frame(buf).iloc[[-1]])
            else:
                f = station_features(buf)
                if name == "robust_statistical":
                    robust_stat_score(f.iloc[[-1]])
                elif name == "isolation_forest":
                    bundle.iforest.score(f.iloc[[-1]])
                else:
                    bundle.ae.score(f.tail(12))
            ms.append((time.perf_counter() - t0) * 1000)
        out[name] = latency_stats(np.array(ms))
    return out


# ------------------------------------------------------------------------------------------ plots
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plots(out: Path, R: pd.DataFrame, approaches: dict, y: np.ndarray, m: np.ndarray, cls: dict, cal: dict, lat_ms: list):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (score, _) in approaches.items():
        if name.startswith("hybrid[") and name != "hybrid[full]":
            continue
        s = np.nan_to_num(np.where(np.isinf(score), 1e9, np.asarray(score, float)), nan=0.0)[m]
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=APPROACH_NAMES.get(name, name))
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (frozen final test)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    _save(fig, out / "precision_recall_curve.png")

    labels = cls["labels"]
    cm = np.array(cls["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicted fault pattern", ylabel="True fault pattern", title="Fault-pattern confusion matrix (row-normalised colour)")
    _save(fig, out / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    rel = cal["reliability"]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot([b["mean_predicted"] for b in rel], [b["observed_rate"] for b in rel], "o-", label="SkyGuard (isotonic)")
    ax.set(xlabel="Predicted probability", ylabel="Observed anomaly rate", title=f"Reliability (Brier {cal['brier_score']:.4f})")
    ax.legend()
    _save(fig, out / "calibration_curve.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lat_ms, bins=60, color="#0ea5e9")
    for q, c in ((50, "g"), (95, "orange"), (99, "r")):
        v = np.percentile(lat_ms, q)
        ax.axvline(v, color=c, ls="--", label=f"p{q} = {v:.1f} ms")
    ax.set(xlabel="Per-observation processing latency (ms)", ylabel="Observations", title="Real-time engine latency")
    ax.legend()
    _save(fig, out / "latency_distribution.png")

    # anomaly timeline for the station with most injected events
    ev = R[m & (R["is_anomaly"] == 1)]
    sid = ev["station_id"].value_counts().index[0]
    S = R[R["station_id"] == sid].set_index("timestamp")
    S = S[S.index >= S.index.min() + pd.Timedelta(days=8)].iloc[: 24 * 45]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for ax, v, u in zip(axes, VARS, ("°C", "hPa", "%")):
        x = pd.to_numeric(S[v], errors="coerce").where(lambda s: s.abs() < 5000)
        ax.plot(S.index, x, lw=0.8, color="#475569", label="observed (raw)")
        ax.plot(S.index, S[f"clean_{v}"], lw=0.6, color="#16a34a", alpha=0.6, label="ground truth (clean)")
        tr = S["is_anomaly"] == 1
        ax.scatter(S.index[tr], x[tr], s=10, color="#f59e0b", label="injected anomaly", zorder=3)
        fl = S["flag"] & (S["param"].str.contains(v, regex=False) | (S["param"] == "all"))
        ax.scatter(S.index[fl], x[fl], s=22, facecolors="none", edgecolors="#dc2626", label="SkyGuard alert", zorder=4)
        ax.set_ylabel(f"{v.split('_')[0]} ({u})")
    axes[0].legend(fontsize=7, ncol=4, loc="upper left")
    axes[0].set_title(f"Anomaly timeline - {sid} (first 45 days of the frozen test period)")
    _save(fig, out / "anomaly_timeline.png")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for sid, g in R[m].groupby("station_id"):
        h = g.set_index("timestamp")["health_score"].resample("1D").mean()
        ax.plot(h.index, h.values, lw=0.9, label=sid)
    ax.set(ylabel="Sensor health (daily mean)", ylim=(0, 102), title="Sensor-health score per station (frozen final test)")
    ax.legend(fontsize=6, ncol=11, loc="lower left")
    _save(fig, out / "station_health.png")


# ------------------------------------------------------------------------------------------ main
def evaluate(dataset: Path, model_dir: Path, out: Path, final: bool) -> dict:
    cfg = load_config()
    out.mkdir(parents=True, exist_ok=True)
    integrity = verify_frozen(dataset) if final else None
    if final and not (integrity["frozen"] and integrity["integrity_ok"]):
        raise SystemExit(f"frozen benchmark integrity check FAILED: {integrity}")
    bundle = ModelBundle.load(model_dir)
    md = bundle.metadata
    if final:
        recorded = md.get("frozen_benchmarks", {}).get(dataset.name)
        if recorded:  # the model records every frozen benchmark that existed when it was trained
            assert recorded["sha256"] == integrity["manifest_sha256"], "frozen benchmark changed after training"
        assert md["trained_at"] > integrity["frozen_at"], "benchmark must be frozen before training"
    stations = load_stations()
    t0 = time.time()
    df = read_canonical(dataset / "benchmark.csv.gz").sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    F = features_for(df, stations, bundle.ae, cfg)
    R = score_features(F, bundle, cfg)
    for c in df.columns:
        if c not in R.columns:
            R[c] = df[c].to_numpy()
    m = eval_mask(R)
    y = R.loc[m, "is_anomaly"].to_numpy()

    approaches = baseline_scores(bundle, F)
    abl_dir = ROOT / "artifacts" / "ablation_models"
    for fs in FEATURE_SETS:
        h = joblib.load(abl_dir / f"{fs.replace('+', '_')}.joblib")
        approaches[f"hybrid[{fs}]"] = hybrid_scores(h, F)
    approaches["skyguard_hybrid"] = (R["prob"].to_numpy(), R["flag"].to_numpy())
    # same model with the weather-vs-sensor decision rules switched off (quantifies the rule trade-off on identical data)
    rules = getattr(bundle.hybrid, "rules", None)
    bundle.hybrid.rules = None
    approaches["skyguard_hybrid_without_decision_rules"] = hybrid_scores(bundle.hybrid, F)
    bundle.hybrid.rules = rules
    comp = compare(F, R, approaches)
    weather = weather_vs_sensor(R[m])

    final_det = detection(R[m], R.loc[m, "flag"].to_numpy(), R.loc[m, "prob"].to_numpy()) | event_recall(R[m], R.loc[m, "flag"].to_numpy())
    cls = classification(R.loc[m, "anomaly_type"].to_numpy(), R.loc[m, "pattern"].to_numpy())
    mlrows = m & ~R["qc_hard"].to_numpy()
    cal = calibration(R.loc[mlrows, "is_anomaly"].to_numpy(), R.loc[mlrows, "prob"].to_numpy())
    cal_all = calibration(y, R.loc[m, "prob"].to_numpy())
    imp = imputation(R[m])
    # parameter attribution accuracy on single-parameter injected anomalies that were flagged
    sp = m & R["flag"].to_numpy() & R["injected_parameter"].isin(VARS).to_numpy() & (R["is_anomaly"] == 1).to_numpy()
    param_acc = float((R.loc[sp, "param"] == R.loc[sp, "injected_parameter"]).mean()) if sp.any() else None

    held = integrity["manifest"]["held_out_stations"] if final else []
    per_station = {}
    for sid, g in R[m].groupby("station_id"):
        per_station[sid] = detection(g, g["flag"].to_numpy(), g["prob"].to_numpy()) | {"held_out": sid in held, "neighbors": int(g["sp_n"].max())}
    subsets = {}
    if held:
        for name, sel in (("held_out_stations", R["station_id"].isin(held)), ("seen_stations", ~R["station_id"].isin(held))):
            mm = m & sel.to_numpy()
            subsets[name] = detection(R[mm], R.loc[mm, "flag"].to_numpy(), R.loc[mm, "prob"].to_numpy()) | \
                event_recall(R[mm], R.loc[mm, "flag"].to_numpy()) | \
                {"macro_f1": classification(R.loc[mm, "anomaly_type"].to_numpy(), R.loc[mm, "pattern"].to_numpy())["macro_f1"]}

    # real-time: a neighbour cluster (spatial path) through the real engine + baseline per-observation latency
    present = set(df["station_id"])
    cluster = max(sorted(present), key=lambda s: len([n for n in stations[s].get("neighbors", []) if n["station_id"] in present]))
    members = [cluster] + [n["station_id"] for n in stations[cluster]["neighbors"] if n["station_id"] in set(df["station_id"])][:3]
    rt = measure_hybrid_stream(bundle, df, stations, members, days=4)
    base_lat = measure_baseline_latency(bundle, df, members[0])

    table = []
    for name in ("fixed_threshold", "robust_statistical", "isolation_forest", "temporal_autoencoder", "skyguard_hybrid"):
        r = comp[name]
        table.append({"approach": APPROACH_NAMES[name], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                      "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"], "false_alerts_per_station_day": r["false_alerts_per_station_day"],
                      "event_recall": r["event_recall"],
                      "p95_latency_ms": (rt["latency"] if name == "skyguard_hybrid" else base_lat[name])["p95_ms"]})
    abl = []
    labels = {"temporal": "Temporal only", "temporal+multivariate": "Temporal + Multivariate", "temporal+spatial": "Temporal + Spatial",
              "temporal+multivariate+spatial": "Temporal + Multivariate + Spatial", "full": "Full SkyGuard (+autoencoder, regional rule)"}
    for fs in FEATURE_SETS:
        r = comp[f"hybrid[{fs}]"]
        abl.append({"variant": labels[fs], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "pr_auc": r["pr_auc"],
                    "false_alerts_per_station_day": r["false_alerts_per_station_day"], "event_recall": r["event_recall"],
                    "regional_events_falsely_alerted": r["regional_events_falsely_alerted"]})

    edge_path = ROOT / "artifacts" / "edge" / "edge_metrics.json"
    result = {
        "split": "final_test" if final else "validation", "dataset": str(dataset.relative_to(ROOT)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "benchmark_integrity": {k: v for k, v in (integrity or {}).items() if k != "manifest"},
        "eval_rows": int(m.sum()), "excluded_rows": {"warmup_or_real_source_gap": int((~m).sum())},
        "selected_variant": md["selected_variant"], "alert_threshold": bundle.hybrid.threshold,
        "detection": final_det, "classification": cls, "parameter_attribution_accuracy": param_acc,
        "calibration_ml_scored_rows": cal, "calibration_all_rows": {k: v for k, v in cal_all.items() if k != "reliability"},
        "imputation": imp, "baseline_comparison": table, "ablation": abl, "all_approaches": comp,
        "decision_rules": rules, "weather_vs_sensor": weather,
        "decision_rule_tradeoff": {k: {x: comp[k][x] for x in ("precision", "recall", "f1", "false_alerts_per_station_day", "event_recall",
                                                                "regional_events_falsely_alerted")}
                                   for k in ("skyguard_hybrid_without_decision_rules", "skyguard_hybrid")},
        "held_out": subsets, "per_station": per_station,
        "realtime": {k: v for k, v in rt.items() if k != "samples_ms"}, "baseline_latency_ms": base_lat,
        "edge": json.loads(edge_path.read_text()) if edge_path.exists() else "not built (run python edge/build_edge.py)",
        "compute_seconds": round(time.time() - t0, 1),
        "hardware": {"cpu": __import__("platform").processor(), "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=_json))
    flat = [{"metric": k, "value": v} for k, v in final_det.items() if not isinstance(v, dict)]
    flat += [{"metric": "macro_f1", "value": cls["macro_f1"]}, {"metric": "brier_score", "value": cal["brier_score"]},
             {"metric": "expected_calibration_error", "value": cal["expected_calibration_error"]},
             {"metric": "latency_p50_ms", "value": rt["latency"]["p50_ms"]}, {"metric": "latency_p95_ms", "value": rt["latency"]["p95_ms"]},
             {"metric": "latency_p99_ms", "value": rt["latency"]["p99_ms"]}, {"metric": "throughput_obs_per_s", "value": rt["throughput_obs_per_s"]}]
    flat += [{"metric": f"imputation_{v}_{k}", "value": x} for v, d in imp.items() for k, x in d.items()]
    pd.DataFrame(flat).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(table).to_csv(out / "baseline_comparison.csv", index=False)
    pd.DataFrame(abl).to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(cls["per_class"]).T.to_csv(out / "per_class.csv")
    pd.DataFrame(per_station).T.to_csv(out / "per_station.csv")
    (out / "model_metadata.json").write_text(json.dumps(md, indent=2, default=str))
    plots(out, R, approaches, y, m, cls, cal, rt["samples_ms"])
    with open(out / "run_log.jsonl", "a") as f:
        f.write(json.dumps({"evaluated_at": result["evaluated_at"], "f1": final_det["f1"], "pr_auc": final_det["pr_auc"],
                            "benchmark_sha256": (integrity or {}).get("file_sha256"), "model_trained_at": md["trained_at"]}) + "\n")
    return result


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

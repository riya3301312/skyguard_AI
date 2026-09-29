"""ML tests on the real benchmark/artifacts (skipped with a reason if they were not built)."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.config import ROOT, load_config
from ml.models import SkyGuardHybrid
from ml.pipeline import ModelBundle, features_for, load_stations, score_frame
from ml.report import verify_frozen
from ml.schema import VARS, read_canonical
from ml.stream import StreamEngine

BENCH = ROOT / "data" / "benchmark"
MODEL = ROOT / "artifacts" / "final_model"
need_bench = pytest.mark.skipif(not (BENCH / "final_eval" / "MANIFEST.json").exists(), reason="run scripts/build_benchmark.py")
need_model = pytest.mark.skipif(not (MODEL / "model.joblib").exists(), reason="run scripts/train.py")


@need_bench
def test_final_benchmark_frozen_and_intact():
    v = verify_frozen(BENCH / "final_eval")
    assert v["frozen"] and v["integrity_ok"], v


@need_bench
def test_leakage_safe_splits():
    cfg = load_config()
    man = {s: json.loads((BENCH / s / "MANIFEST.json").read_text()) for s in ("train", "val", "final_eval")}
    assert len({man[s]["seed"] for s in man}) == 3
    assert len({man[s]["injection_profile"] for s in man}) == 3
    held = set(man["final_eval"]["held_out_stations"])
    assert held and not held & set(man["train"]["stations"]) and not held & set(man["val"]["stations"])
    assert held <= set(man["final_eval"]["stations"])
    t_end, v_end = pd.Timestamp(cfg["split"]["train_end"], tz="UTC"), pd.Timestamp(cfg["split"]["val_end"], tz="UTC")
    tr = read_canonical(BENCH / "train" / "benchmark.csv.gz")
    va = read_canonical(BENCH / "val" / "benchmark.csv.gz")
    assert tr["timestamp"].max() < t_end
    ev = va[va["split_role"] == "eval"]
    assert ev["timestamp"].min() >= t_end and ev["timestamp"].max() < v_end
    assert (va.loc[va["split_role"] == "warmup", "is_anomaly"].eq(0) | va.loc[va["split_role"] == "warmup", "injection_id"].eq("REAL_GAP")).all()
    assert pd.Timestamp(man["final_eval"]["period_start"]) >= v_end


@need_model
@need_bench
def test_model_metadata_and_freeze_order():
    md = json.loads((MODEL / "model_metadata.json").read_text())
    for k in ("feature_version", "config_version", "seed", "selected_variant", "alert_threshold", "features", "trained_at",
              "final_benchmark_sha256", "baseline_thresholds", "correction_method"):
        assert k in md, k
    fz = json.loads((BENCH / "final_eval" / "MANIFEST.json").read_text())
    assert md["final_benchmark_sha256"] == fz["sha256_uncompressed_csv"]
    assert md["trained_at"] > fz["frozen_at"]
    assert not set(md["held_out_stations"]) & set(md["train_stations"])
    assert not {"latitude", "longitude", "altitude_m"} & set(md["features"])  # model inputs: T, P, RH derived only


@need_bench
def test_training_is_reproducible_with_fixed_seed():
    tr = read_canonical(BENCH / "train" / "benchmark.csv.gz")
    sub = tr[tr["station_id"].isin(sorted(tr["station_id"].unique())[:3]) & (tr["timestamp"] < "2023-07-01")]
    F = features_for(sub, load_stations(), None)
    sub = sub.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    y, c, p = sub["is_anomaly"].to_numpy(), sub["anomaly_type"].to_numpy(), sub["injected_parameter"].to_numpy()
    a = SkyGuardHybrid("temporal+multivariate+spatial", seed=3).fit(F, y, c, p)
    b = SkyGuardHybrid("temporal+multivariate+spatial", seed=3).fit(F, y, c, p)
    np.testing.assert_array_equal(a._raw(F), b._raw(F))


@need_model
@need_bench
def test_stream_engine_matches_batch_pipeline():
    """Training/serving parity on validation data for a station without neighbours."""
    stations = load_stations()
    va = read_canonical(BENCH / "val" / "benchmark.csv.gz")
    sid = next(s for s in sorted(va["station_id"].unique()) if not stations[s]["neighbors"])
    d = va[va["station_id"] == sid].sort_values("timestamp").iloc[: 24 * 11]
    bundle = ModelBundle.load(MODEL)
    batch = score_frame(d, {sid: stations[sid]}, bundle)
    eng = StreamEngine({sid: stations[sid]}, bundle)
    got = []
    for r in d.itertuples():
        obs = {"station_id": sid, "timestamp": r.timestamp.isoformat(), **{v: None if pd.isna(getattr(r, v)) else getattr(r, v) for v in VARS}}
        if all(obs[v] is None for v in VARS):
            continue
        out = eng.process(obs, ingestion_time=r.timestamp.to_pydatetime())
        got.append((r.timestamp, out["anomaly_probability"], out["fault_pattern"]))
    g = pd.DataFrame(got, columns=["timestamp", "p", "pattern"]).set_index("timestamp")
    b = batch.set_index("timestamp").loc[g.index]
    live = g.index >= g.index[0] + pd.Timedelta(days=8)
    np.testing.assert_allclose(g.loc[live, "p"].to_numpy(float), b.loc[live, "prob"].to_numpy(float), atol=1e-6)
    assert (g.loc[live, "pattern"] == b.loc[live, "pattern"]).all()


@need_model
def test_final_evaluation_artifacts_complete():
    out = ROOT / "artifacts" / "final_evaluation"
    if not (out / "metrics.json").exists():
        pytest.skip("run scripts/evaluate_final.py")
    for f in ("metrics.json", "metrics.csv", "confusion_matrix.png", "precision_recall_curve.png", "calibration_curve.png",
              "latency_distribution.png", "anomaly_timeline.png", "station_health.png", "model_metadata.json"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    assert m["benchmark_integrity"]["integrity_ok"]
    for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_alerts_per_1000_obs", "false_alerts_per_station_day"):
        assert m["detection"][k] is not None
    assert {"macro_f1", "per_class", "confusion_matrix"} <= set(m["classification"])
    assert m["held_out"]["held_out_stations"]["n"] > 0

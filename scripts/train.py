"""Train all detectors on TRAIN, select thresholds/calibration/variant on VALIDATION.

Never reads data/benchmark/final_eval (only its MANIFEST, to prove it was frozen first).
Outputs: artifacts/final_model/ (selected bundle + metadata), artifacts/ablation_models/,
artifacts/validation/selection.json
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.benchmarking import baseline_scores, compare, hybrid_scores  # noqa: E402
from ml.config import load_config  # noqa: E402
from ml.evaluation import eval_mask, imputation  # noqa: E402
from ml.models import (AE_WIN, FEATURE_SETS, IForestDetector, SkyGuardHybrid, TemporalAutoencoder,  # noqa: E402
                       ae_score, best_threshold, robust_stat_score, station_days)
from ml.pipeline import ModelBundle, add_ae, features_for, load_stations, score_features  # noqa: E402
from ml.schema import read_canonical  # noqa: E402

RULE_GRID = [{"confirm_persistent": False, "k6": 1.0, "c3": 0.5, "max_shared": 0.5, "regional_suppress": rs, "regional_min": 0.5}
             for rs in (False, True)] +             [{"confirm_persistent": True, "k6": k, "c3": c, "max_shared": 0.5, "regional_suppress": rs, "regional_min": 0.5}
             for k in (0.75, 1.0, 1.5) for c in (0.5, 1.0) for rs in (False, True)]

BENCH = ROOT / "data" / "benchmark"
ART = ROOT / "artifacts"


def load_split(name: str, stations: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_canonical(BENCH / name / "benchmark.csv.gz")
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    F = features_for(df, stations, None, cfg)
    return df, F


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["seed"]
    np.random.seed(seed)
    man_path = BENCH / "final_eval" / "MANIFEST.json"
    if not man_path.exists() or not json.loads(man_path.read_text()).get("frozen"):
        raise SystemExit("Freeze the final benchmark first: python scripts/build_benchmark.py")
    frozen = json.loads(man_path.read_text())
    stations = load_stations()
    t0 = time.time()
    tr, Ftr = load_split("train", stations, cfg)
    va, Fva = load_split("val", stations, cfg)
    assert not set(frozen["held_out_stations"]) & (set(tr["station_id"]) | set(va["station_id"])), "station leakage"
    print(f"features: train {Ftr.shape}, val {Fva.shape} in {time.time() - t0:.0f}s")

    # temporal autoencoder on windows that contain no injected anomaly (training labels only)
    frames, masks = [], []
    for sid, g in Ftr.groupby("station_id", sort=False):
        lab = tr.loc[g.index, "is_anomaly"].rolling(AE_WIN, min_periods=1).max().to_numpy() == 0
        frames.append(g)
        masks.append(lab & ~g["qc_hard"].to_numpy())
    ae = TemporalAutoencoder(seed).fit(frames, masks)
    Ftr, Fva = add_ae(Ftr, ae), add_ae(Fva, ae)
    print(f"autoencoder trained ({time.time() - t0:.0f}s)")

    y, cls = tr["is_anomaly"].to_numpy(), tr["anomaly_type"].to_numpy()
    param = tr["injected_parameter"].to_numpy()
    mv = eval_mask(va)
    yv = va["is_anomaly"].to_numpy()

    # ---- baselines: thresholds chosen on validation
    iforest = IForestDetector(seed).fit(Ftr[~Ftr["qc_hard"]])
    thresholds = {}
    for name, fn in {"robust_statistical": robust_stat_score, "isolation_forest": iforest.score,
                     "temporal_autoencoder": ae_score}.items():
        s = fn(Fva)[mv]
        finite = np.isfinite(s)
        thresholds[name] = best_threshold(yv[mv][finite], s[finite], None, station_days(Fva[mv]),
                                          cfg["model"]["max_false_alerts_per_station_day"])
    bundle_tmp = ModelBundle(hybrid=None, ae=ae, iforest=iforest, thresholds=thresholds)
    approaches = baseline_scores(bundle_tmp, Fva)

    # ---- hybrid variants (ablation) trained on TRAIN, calibrated/thresholded on VALIDATION
    variants = {}
    for fs in FEATURE_SETS:
        h = SkyGuardHybrid(fs, seed=seed, regional_rule=(fs == "full"))
        h.fit(Ftr, y, cls, param).calibrate(Fva[mv], yv[mv], cfg)
        h.rule_trace = h.select_rules(Fva[mv].reset_index(drop=True), yv[mv], cfg, RULE_GRID,
                                      va.loc[mv, "injection_id"].to_numpy(), va.loc[mv, "injection_kind"].to_numpy())
        h.selection = next(r for r in h.rule_trace if r["threshold"] == h.threshold and all(r[k] == h.rules[k] for k in h.rules))
        variants[fs] = h
        approaches[f"hybrid[{fs}]"] = hybrid_scores(h, Fva)
        print(f"hybrid[{fs}] threshold={h.threshold:.2f} rules={h.rules} event_recall={h.selection['event_recall']:.3f} "
              f"regional_alerted={h.selection['regional_events_alerted']:.3f} ({time.time() - t0:.0f}s)")
    val = compare(Fva, va, approaches)

    # ---- select the final variant on validation F1 (ties -> fewer false alerts per station-day)
    # pre-registered operational criterion (same as rule selection): feasible budgets, then event recall, then F1
    chosen = max(FEATURE_SETS, key=lambda fs: (variants[fs].selection["feasible"], round(variants[fs].selection["event_recall"], 3),
                                                round(variants[fs].selection["f1"], 3)))
    final = variants[chosen]

    # ---- correction estimator chosen on validation
    best_m, best_err, corr_val = None, np.inf, {}
    for method in ("auto", "temporal"):
        b = ModelBundle(hybrid=final, ae=ae, correction_method=method)
        R = score_features(Fva, b, cfg).assign(**{c: va[c].to_numpy() for c in va.columns if c not in Fva.columns})
        imp = imputation(R[eval_mask(R)])
        corr_val[method] = imp
        rel = np.mean([v["mae"] / max(v["raw_value_mae_for_reference"] or 1, 1e-6) for v in imp.values()])
        if rel < best_err:
            best_m, best_err = method, rel

    meta = {
        "model_name": "SkyGuard Hybrid", "selected_variant": chosen, "selection_rule": "validation: false alerts/station-day <= budget and regional-event alert share <= budget; maximise event recall, then F1",
        "decision_rules": final.rules, "validation_operating_point": final.selection,
        "operating_point_rule": f"max validation F1 subject to <= {cfg['model']['max_false_alerts_per_station_day']} false alerts per station-day (all approaches)",
        "validation_f1_by_variant": {fs: val[f"hybrid[{fs}]"]["f1"] for fs in FEATURE_SETS},
        "validation_selection_by_variant": {fs: variants[fs].selection for fs in FEATURE_SETS},
        "alert_threshold": final.threshold, "regional_suppression": final.regional_suppression,
        "regional_fraction": final.regional_fraction, "calibration": "isotonic regression on validation",
        "features": final.cols, "feature_version": cfg["feature_version"], "config_version": cfg["config_version"],
        "seed": seed, "baseline_thresholds": thresholds, "correction_method": best_m,
        "train_rows": int(len(tr)), "val_rows": int(len(va)),
        "train_stations": sorted(tr["station_id"].unique()), "val_stations": sorted(va["station_id"].unique()),
        "train_period": [str(tr["timestamp"].min()), str(tr["timestamp"].max())],
        "val_period": [str(va.loc[mv, "timestamp"].min()), str(va.loc[mv, "timestamp"].max())],
        "held_out_stations": frozen["held_out_stations"],
        "final_benchmark_sha256": frozen["sha256_uncompressed_csv"], "final_benchmark_frozen_at": frozen["frozen_at"],
        "frozen_benchmarks": {p.parent.name: {"sha256": json.loads(p.read_text())["sha256_uncompressed_csv"],
                                              "frozen_at": json.loads(p.read_text())["frozen_at"]}
                              for p in sorted(BENCH.glob("final_eval*/MANIFEST.json")) if json.loads(p.read_text()).get("frozen")},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_inputs": "temperature_c, pressure_hpa, relative_humidity_pct only (lat/lon used only for neighbour selection)",
        "libraries": {"scikit-learn": __import__("sklearn").__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    bundle = ModelBundle(hybrid=final, ae=ae, iforest=iforest, thresholds=thresholds, correction_method=best_m, metadata=meta)
    bundle.save(ART / "final_model")
    (ART / "ablation_models").mkdir(parents=True, exist_ok=True)
    for fs, h in variants.items():
        joblib.dump(h, ART / "ablation_models" / f"{fs.replace('+', '_')}.joblib", compress=3)
    (ART / "validation").mkdir(parents=True, exist_ok=True)
    (ART / "validation" / "selection.json").write_text(json.dumps(
        {"validation_metrics": val, "selected_variant": chosen, "correction_validation": corr_val,
         "correction_method": best_m, "thresholds": thresholds,
         "rule_selection_trace": {fs: variants[fs].rule_trace for fs in FEATURE_SETS}}, indent=2, default=str))
    print(f"selected variant: {chosen}; correction: {best_m}; total {time.time() - t0:.0f}s")
    for k, v in val.items():
        print(f"  {k:32s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} PR-AUC={v['pr_auc'] or 0:.3f} "
              f"FA/st-day={v['false_alerts_per_station_day']:.3f} ev-recall={v['event_recall']:.3f}")


if __name__ == "__main__":
    main()

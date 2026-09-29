"""Weather-vs-sensor and drift diagnostics on the VALIDATION split only (never the final benchmark).

python scripts/diagnose_validation.py
Writes artifacts/regression/validation_diagnostics.json and per-event CSVs.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.config import load_config  # noqa: E402
from ml.evaluation import detection, eval_mask, event_recall  # noqa: E402
from ml.pipeline import ModelBundle, features_for, load_stations, score_features  # noqa: E402
from ml.schema import read_canonical  # noqa: E402

OUT = ROOT / "artifacts" / "regression"
CACHE = OUT / "_val_features.joblib"


def load(bundle):
    va = read_canonical(ROOT / "data" / "benchmark" / "val" / "benchmark.csv.gz").sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    if CACHE.exists() and CACHE.stat().st_mtime > (ROOT / "ml" / "features.py").stat().st_mtime:
        F = joblib.load(CACHE)
    else:
        F = features_for(va, load_stations(), bundle.ae)
        joblib.dump(F, CACHE)
    return va, F


def regional_report(R: pd.DataFrame) -> dict:
    reg = R[(R["injection_kind"] == "regional_event") & eval_mask(R)]
    by_ev = reg.groupby("injection_id").agg(station=("station_id", "first"), alerted=("flag", "any"), rows=("flag", "size"),
                                            max_prob=("prob", "max"), sp_n=("sp_n", "max"))
    alerted = reg[reg["flag"]]
    first_row = reg.groupby("injection_id").cumcount() == 0
    return {
        "regional_event_station_instances": int(len(by_ev)),
        "instances_with_any_alert": int(by_ev["alerted"].sum()),
        "falsely_alerted_rate": float(by_ev["alerted"].mean()),
        "alerted_rows": int(len(alerted)),
        "alerted_rows_at_event_onset_row": int((reg["flag"] & first_row).sum()),
        "alerted_patterns": alerted["pattern"].value_counts().to_dict(),
        "alerted_parameters": alerted["param"].value_counts().to_dict(),
        "alert_rate_by_neighbours": reg.assign(n=reg["sp_n"].fillna(0).clip(0, 3)).groupby("n")["flag"].mean().round(3).to_dict(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    bundle = ModelBundle.load(ROOT / "artifacts" / "final_model")
    va, F = load(bundle)
    R = score_features(F, bundle, cfg)
    for c in va.columns:
        if c not in R:
            R[c] = va[c].to_numpy()
    m = eval_mask(R)
    out = {"model": bundle.metadata.get("selected_variant"), "threshold": bundle.hybrid.threshold,
           "detection": {k: v for k, v in detection(R[m], R.loc[m, "flag"].to_numpy(), R.loc[m, "prob"].to_numpy()).items()},
           "events": event_recall(R[m], R.loc[m, "flag"].to_numpy()), "regional": regional_report(R)}
    fp = R[m & R["flag"] & (R["is_anomaly"] == 0)]
    out["false_positive_patterns"] = fp["pattern"].value_counts().to_dict()
    out["false_positive_kinds"] = fp["injection_kind"].replace("", "clean").value_counts().to_dict()
    (OUT / "validation_diagnostics.json").write_text(json.dumps(out, indent=2, default=str))
    reg = R[(R["injection_kind"] == "regional_event") & R["flag"]]
    cols = ["station_id", "timestamp", "injection_id", "prob", "pattern", "param", "sp_n", "t_tod_z", "t_sp_frac", "t_sp_z",
            "rh_tod_z", "rh_sp_frac", "p_tod_z", "p_sp_frac", "mv_rh_resid", "t_rate_z"]
    reg[cols].to_csv(OUT / "validation_regional_false_alerts.csv", index=False)
    print(json.dumps(out, indent=2, default=str))
    joblib.dump(R, OUT / "_val_results.joblib")


if __name__ == "__main__":
    main()

"""Frozen final evaluation (run once, after training). Produces artifacts/final_evaluation/*.

python scripts/evaluate_final.py --dataset data/benchmark/final_eval_v2 --model artifacts/final_model
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.report import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/benchmark/final_eval_v2")
    ap.add_argument("--model", default="artifacts/final_model")
    ap.add_argument("--out", default="artifacts/final_evaluation")
    a = ap.parse_args()
    r = evaluate(ROOT / a.dataset, ROOT / a.model, ROOT / a.out, final=True)
    d = r["detection"]
    print(json.dumps({"benchmark_integrity": r["benchmark_integrity"], "precision": d["precision"], "recall": d["recall"],
                      "f1": d["f1"], "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"],
                      "false_alerts_per_station_day": d["false_alerts_per_station_day"], "macro_f1": r["classification"]["macro_f1"],
                      "brier": r["calibration_ml_scored_rows"]["brier_score"], "latency_ms": r["realtime"]["latency"]}, indent=2))
    for row in r["baseline_comparison"]:
        print(row)
    for row in r["ablation"]:
        print(row)


if __name__ == "__main__":
    main()

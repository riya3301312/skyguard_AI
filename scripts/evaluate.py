"""Validation evaluation (model development). Never touches the frozen final benchmark.

python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.report import evaluate  # noqa: E402

if __name__ == "__main__":
    r = evaluate(ROOT / "data" / "benchmark" / "val", ROOT / "artifacts" / "final_model", ROOT / "artifacts" / "validation", final=False)
    print(json.dumps({k: r["detection"][k] for k in ("precision", "recall", "f1", "pr_auc", "false_alerts_per_station_day")}, indent=2))
    for row in r["baseline_comparison"] + r["ablation"]:
        print(row)

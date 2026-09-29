"""Inject reproducible anomalies into a canonical clean CSV.

python scripts/inject_anomalies.py --input data/processed/clean.csv --output data/benchmark.csv --seed 42
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from injection.injector import PROFILES, inject  # noqa: E402
from ml.schema import read_canonical  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "processed" / "clean.csv"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="train")
    args = ap.parse_args()
    df = read_canonical(args.input)
    out, events = inject(df, args.seed, args.profile)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    ev_path = Path(args.output).with_suffix("").as_posix().removesuffix(".csv") + "_events.csv"
    events.to_csv(ev_path, index=False)
    print(f"{len(out)} rows, {int(out['is_anomaly'].sum())} anomalous rows, {len(events)} events -> {args.output}")
    print(events.groupby("anomaly_type").size().to_string())


if __name__ == "__main__":
    main()

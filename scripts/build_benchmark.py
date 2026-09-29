"""Leakage-safe benchmark construction and freezing.

TRAIN      : period < split.train_end, non-held-out stations, seed injection.train_seed, profile 'train'
VALIDATION : train_end <= t < val_end, non-held-out stations, seed val_seed, profile 'val'
FINAL TEST : t >= val_end, ALL stations (incl. held-out), unseen seed test_seed, profile 'test'

Held-out stations never appear in train/validation (not even as spatial neighbours).
Val/test files carry `warmup_days` of clean prior history (split_role='warmup') so that
past-only features are defined; warm-up rows are never scored.

The final benchmark is written once to data/benchmark/final_eval/ with a SHA-256
MANIFEST and marked frozen. Re-running verifies the hash instead of overwriting.
"""
import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from injection.injector import inject  # noqa: E402
from ml.config import load_config  # noqa: E402
from ml.schema import read_canonical  # noqa: E402

BENCH = ROOT / "data" / "benchmark"


def to_gz_bytes(df: pd.DataFrame) -> tuple[bytes, str]:
    raw = df.to_csv(index=False).encode()
    return gzip.compress(raw, mtime=0), hashlib.sha256(raw).hexdigest()


def make_split(clean, stations, t0, t1, seed, profile, cfg, warmup):
    lo = t0 - pd.Timedelta(days=warmup) if t0 is not None else None
    m = clean["station_id"].isin(stations)
    if lo is not None:
        m &= clean["timestamp"] >= lo
    if t1 is not None:
        m &= clean["timestamp"] < t1
    df = clean[m].copy()
    out, events = inject(df, seed, profile, cfg=cfg)
    out["split_role"] = "eval" if t0 is None else out["timestamp"].ge(t0).map({True: "eval", False: "warmup"})
    return out, events


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(ROOT / "data" / "processed" / "clean.csv"))
    args = ap.parse_args()
    clean = read_canonical(args.clean)
    sp, inj = cfg["split"], cfg["injection"]
    all_st = sorted(clean["station_id"].unique())
    holdout = [s for s in sp["holdout_stations"] if s in all_st]
    if not holdout:
        raise SystemExit("no configured held-out station is available; station-holdout cannot be claimed")
    seen = [s for s in all_st if s not in holdout]
    train_end, val_end = pd.Timestamp(sp["train_end"], tz="UTC"), pd.Timestamp(sp["val_end"], tz="UTC")
    wd = inj["warmup_days"]

    specs = {
        "train": (seen, None, train_end, inj["train_seed"], "train"),
        "val": (seen, train_end, val_end, inj["val_seed"], "val"),
        "final_eval": (all_st, val_end, None, inj["test_seed"], "test"),
        "final_eval_v2": (all_st, val_end, None, inj["test_seed_v2"], "test"),
    }
    for name, (sts, t0, t1, seed, profile) in specs.items():
        d = BENCH / name
        d.mkdir(parents=True, exist_ok=True)
        out, events = make_split(clean, sts, t0, t1, seed, profile, cfg, wd)
        body, digest = to_gz_bytes(out)
        manifest_path = d / "MANIFEST.json"
        if name.startswith("final_eval") and manifest_path.exists() and json.loads(manifest_path.read_text()).get("frozen"):
            man = json.loads(manifest_path.read_text())
            ok = man["sha256_uncompressed_csv"] == digest
            on_disk = hashlib.sha256(gzip.decompress((d / "benchmark.csv.gz").read_bytes())).hexdigest()
            print(f"{name} is FROZEN (since {man['frozen_at']}); regenerated hash matches: {ok}; "
                  f"on-disk hash matches: {on_disk == man['sha256_uncompressed_csv']}. Not overwritten.")
            if not (ok and on_disk == man["sha256_uncompressed_csv"]):
                raise SystemExit("frozen benchmark integrity check FAILED")
            continue
        (d / "benchmark.csv.gz").write_bytes(body)
        events.to_csv(d / "events.csv", index=False)
        ev = out[out["split_role"] == "eval"]
        man = {
            "split": name, "seed": seed, "injection_profile": profile, "stations": sts,
            "held_out_stations": holdout if name.startswith("final_eval") else [],
            "period_start": str(t0) if t0 is not None else str(out["timestamp"].min()),
            "period_end": str(t1) if t1 is not None else str(out["timestamp"].max()),
            "warmup_days": wd if t0 is not None else 0, "rows": int(len(out)), "eval_rows": int(len(ev)),
            "anomalous_eval_rows": int(ev["is_anomaly"].sum()),
            "rows_by_type": ev["anomaly_type"].value_counts().to_dict(),
            "events_by_kind": events["kind"].value_counts().to_dict() if len(events) else {},
            "sha256_uncompressed_csv": digest, "config_version": cfg["config_version"],
            "feature_version": cfg["feature_version"],
        }
        if name.startswith("final_eval"):
            man["frozen"] = True
            man["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            man["note"] = "Frozen before model training/selection. Never used for tuning."
        manifest_path.write_text(json.dumps(man, indent=2))
        if name.startswith("final_eval"):
            for f in (d / "benchmark.csv.gz", d / "events.csv", manifest_path):
                os.chmod(f, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)  # read-only guard
        print(f"{name}: {len(out)} rows ({len(ev)} eval), {int(ev['is_anomaly'].sum())} anomalous, "
              f"{len(events)} events, stations={len(sts)}, sha256={digest[:12]}")
    print(f"held-out stations: {holdout}")


if __name__ == "__main__":
    main()

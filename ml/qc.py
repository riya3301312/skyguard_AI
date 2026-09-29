"""Deterministic, configuration-driven data-quality checks (run before any ML)."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import load_config
from .schema import SHORT, VARS


def run_length(changed: np.ndarray) -> np.ndarray:
    """Length (in rows) of the current run of 'not changed' values, counting the current row."""
    out = np.zeros(len(changed), dtype=float)
    run = 0
    for i, c in enumerate(changed):
        run = 1 if c else run + 1
        out[i] = run
    return out


def qc_frame(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Per-station QC. `df` is indexed by observation_time (sorted, unique) with the VARS columns.

    Returns flag columns plus cleaned values `x_<v>` (sentinel / out-of-range -> NaN).
    Raw values are never modified; cleaned copies are separate columns.
    """
    cfg = cfg or load_config()
    q = cfg["qc"]
    out = pd.DataFrame(index=df.index)
    sentinels = np.array(q["sentinels"], dtype=float)
    for v in VARS:
        s = SHORT[v]
        x = pd.to_numeric(df[v], errors="coerce").astype(float)
        lim = q[v]
        missing = x.isna()
        sentinel = x.isin(sentinels)
        rng = ~missing & ~sentinel & ((x < lim["min"]) | (x > lim["max"]))
        clean = x.mask(sentinel | rng)
        prev = clean.ffill().shift(1)
        prev_t = pd.Series(clean.index, index=clean.index).where(clean.notna()).ffill().shift(1)
        recent = (clean.index.to_series() - prev_t) <= pd.Timedelta("2h")
        step = (clean - prev).abs().gt(lim["max_step"]) & recent
        out[f"qc_{s}_missing"] = missing
        out[f"qc_{s}_sentinel"] = sentinel
        out[f"qc_{s}_range"] = rng
        out[f"qc_{s}_step"] = step
        out[f"x_{s}"] = clean
    xs = out[[f"x_{SHORT[v]}" for v in VARS]]
    out["qc_all_missing"] = xs.isna().all(axis=1) & ~out[[f"qc_{SHORT[v]}_sentinel" for v in VARS]].any(axis=1)
    # hours since the previous row that carried any measurement (missing rows / absent rows both count as gap)
    idx = df.index.to_series()
    last_obs = idx.where(xs.notna().any(axis=1).to_numpy()).ffill().shift(1)
    out["qc_gap_h"] = (idx - last_obs).dt.total_seconds().div(3600).fillna(0.0)
    out["qc_gap"] = out["qc_gap_h"] * 60 > q["abnormal_gap_minutes"]
    changed = (xs.diff().abs().fillna(1.0) > 0).any(axis=1).to_numpy() | xs.isna().any(axis=1).to_numpy()
    out["qc_flat_all_run"] = run_length(changed)
    out["qc_flat_all"] = out["qc_flat_all_run"] >= q["flatline_slots"]
    sent_any = out[[f"qc_{SHORT[v]}_sentinel" for v in VARS]].any(axis=1)
    rng_any = out[[f"qc_{SHORT[v]}_range" for v in VARS]].any(axis=1)
    out["qc_hard"] = sent_any | rng_any | out["qc_all_missing"]
    out["qc_hard_class"] = np.select(
        [out["qc_all_missing"], sent_any, rng_any],
        ["MISSING_COMMUNICATION", "SENTINEL_CORRUPTION", "SENTINEL_CORRUPTION"], default="")
    return out


def ingest_check(observation_time: datetime, ingestion_time: datetime, cfg: dict | None = None) -> dict:
    """Checks that depend on arrival, not on the atmosphere. A delayed record is NOT an anomaly."""
    q = (cfg or load_config())["qc"]
    lag_min = (ingestion_time - observation_time).total_seconds() / 60
    return {
        "future": observation_time > ingestion_time + timedelta(minutes=q["future_tolerance_minutes"]),
        "delayed": lag_min > q["delayed_minutes"],
        "lag_minutes": lag_min,
    }

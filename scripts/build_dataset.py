"""Build the canonical clean dataset from preserved raw files (offline).

data/raw/iem_metar/*.csv.gz  ->  data/processed/clean.csv + data/processed/stations.json

Steps: parse -> canonical units/schema -> routine reports only -> dedupe ->
deterministic QC removal of impossible values (recorded) -> regular grid
(short gaps interpolated and flagged, long gaps left missing) -> completeness filter.
Raw files are only read, never written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.config import load_config  # noqa: E402
from ml.features import neighbors  # noqa: E402
from ml.schema import CANONICAL, VARS, validate_canonical  # noqa: E402

INHG_TO_HPA = 33.8639


def load_raw(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, na_values=["M", "T"], compression="gzip")
    r["timestamp"] = pd.to_datetime(r["valid"], utc=True)
    return r


def canonicalize(r: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    sid = str(r["station"].iloc[0])
    step = cfg["data"]["interval_minutes"]
    r = r[r["timestamp"].dt.minute % step == 0]  # routine reports on the canonical grid only
    r = r.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    d = pd.DataFrame(index=r.index)
    d["temperature_c"] = r["tmpc"].astype(float)
    d["pressure_hpa"] = (r["alti"].astype(float) * INHG_TO_HPA).round(1)
    d["relative_humidity_pct"] = r["relh"].astype(float).round(1)
    q = cfg["qc"]
    removed = 0
    for v in VARS:
        bad = d[v].isin(q["sentinels"]) | (d[v] < q[v]["min"]) | (d[v] > q[v]["max"])
        removed += int(bad.sum())
        d.loc[bad, v] = np.nan
    grid = pd.date_range(pd.Timestamp(cfg["data"]["start"], tz="UTC"), pd.Timestamp(cfg["data"]["end"], tz="UTC"),
                         freq=f"{step}min", inclusive="left")
    d = d.reindex(grid)
    observed = d[VARS].notna().all(axis=1)
    flag = pd.Series(np.where(observed, "ok", "missing"), index=grid, dtype=object)
    lim = cfg["data"]["max_interpolate_slots"]
    for v in VARS:
        filled = d[v].interpolate(method="time", limit=lim, limit_area="inside")
        # only fill gaps that are entirely <= lim slots long
        gap_id = d[v].notna().cumsum()
        gap_len = d[v].isna().groupby(gap_id).transform("sum")
        ok_fill = d[v].isna() & (gap_len <= lim) & filled.notna()
        d.loc[ok_fill, v] = filled[ok_fill].round(1)
    now_full = d[VARS].notna().all(axis=1)
    flag[~observed & now_full] = "interpolated"
    flag[~observed & ~now_full & d[VARS].notna().any(axis=1)] = "partial"
    d["quality_flag"] = flag
    meta = {"station_id": sid, "latitude": float(r["lat"].iloc[0]), "longitude": float(r["lon"].iloc[0]),
            "altitude_m": float(r["elevation"].iloc[0]), "source": f"IEM-METAR/IMD:{sid}",
            "completeness": round(float(observed.mean()), 4), "qc_removed_values": removed,
            "interpolated_rows": int((flag == "interpolated").sum())}
    d = d.reset_index(names="timestamp")
    d["station_id"] = sid
    d["latitude"], d["longitude"], d["altitude_m"] = meta["latitude"], meta["longitude"], meta["altitude_m"]
    d["source"] = meta["source"]
    return d[CANONICAL], meta


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "iem_metar"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {s["id"] for s in cfg["data"]["stations"]}
    frames, metas, excluded = [], {}, {}
    for f in sorted(Path(args.raw).glob("*.csv.gz")):
        sid = f.name.split(".")[0]
        if sid not in wanted:
            continue
        d, m = canonicalize(load_raw(f), cfg)
        if m["completeness"] < cfg["split"]["min_completeness"]:
            excluded[sid] = {**m, "reason": f"completeness {m['completeness']} < {cfg['split']['min_completeness']}"}
            continue
        frames.append(d)
        metas[sid] = m
    df = validate_canonical(pd.concat(frames, ignore_index=True))
    nb = neighbors(metas, cfg)
    for sid in metas:
        metas[sid]["neighbors"] = [{"station_id": o, "distance_km": round(dist, 1)} for o, dist in nb[sid]]
    df.to_csv(out / "clean.csv", index=False)
    provenance = json.loads((Path(args.raw) / "PROVENANCE.json").read_text())
    (out / "stations.json").write_text(json.dumps(
        {"stations": metas, "excluded": excluded, "config_version": cfg["config_version"],
         "grid_minutes": cfg["data"]["interval_minutes"], "provenance": {k: v for k, v in provenance.items() if k != "files"},
         "raw_files": provenance["files"]}, indent=2))
    print(f"clean.csv: {len(df)} rows, {len(metas)} stations kept, {len(excluded)} excluded")
    for sid, m in metas.items():
        print(f"  {sid}: completeness={m['completeness']:.3f} neighbors={[n['station_id'] for n in m['neighbors']]}")
    for sid, m in excluded.items():
        print(f"  EXCLUDED {sid}: {m['reason']}")


if __name__ == "__main__":
    main()

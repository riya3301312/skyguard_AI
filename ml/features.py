"""Temporal, multivariate and spatial features. Past/current information only.

The same functions run in batch evaluation (whole series) and in the real-time
engine (per-station buffer), so training/serving parity holds by construction.
Model inputs are limited to temperature, pressure and relative humidity; station
metadata (lat/lon) is used only to select spatial neighbours.
"""
import math
import warnings

import numpy as np
import pandas as pd

from .config import load_config
from .qc import qc_frame, run_length
from .schema import SHORT, VARS

EPS = {"t": 0.5, "p": 0.3, "rh": 2.0}  # measurement-resolution floors for robust scales
MAGNUS_A, MAGNUS_B = 17.625, 243.04


def dewpoint(t: pd.Series, rh: pd.Series) -> pd.Series:
    g = np.log(rh.clip(1, 100) / 100) + MAGNUS_A * t / (MAGNUS_B + t)
    return MAGNUS_B * g / (MAGNUS_A - g)


def rh_from(t: pd.Series, td: pd.Series) -> pd.Series:
    return 100 * np.exp(MAGNUS_A * td / (MAGNUS_B + td) - MAGNUS_A * t / (MAGNUS_B + t))


def _lookup(x: pd.Series, idx: pd.DatetimeIndex, tol: str) -> np.ndarray:
    """Value of series x nearest to each timestamp in idx within tolerance (NaN otherwise)."""
    xs = x.dropna()
    if xs.empty:
        return np.full(len(idx), np.nan)
    t = xs.index.as_unit("ns").asi8
    q = idx.as_unit("ns").asi8
    v = xs.to_numpy(float)
    pos = np.clip(np.searchsorted(t, q), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(q), dtype=int)
    left = np.clip(pos - 1, 0, len(t) - 1)
    pick = np.where(np.abs(t[pos] - q) < np.abs(q - t[left]), pos, left)
    ok = np.abs(t[pick] - q) <= pd.Timedelta(tol).value
    return np.where(ok, v[pick], np.nan)


def _rolling_slope(x: pd.Series, window: str) -> pd.Series:
    th = pd.Series((x.index - x.index[0]).total_seconds() / 3600.0, index=x.index).where(x.notna())
    r = lambda s: s.rolling(window, min_periods=3)  # noqa: E731
    mt, mx = r(th).mean(), r(x).mean()
    cov = r(th * x).mean() - mt * mx
    var = r(th * th).mean() - mt * mt
    return cov / var.where(var > 1e-9)


def station_features(raw: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Stage-1 features for one station. `raw` indexed by observation_time (sorted, unique)."""
    cfg = cfg or load_config()
    fc = cfg["features"]
    sw, lw, mp = fc["short_window"], fc["long_window"], fc["min_periods"]
    qc = qc_frame(raw, cfg)
    f = pd.DataFrame(index=raw.index)
    idx = raw.index
    dt_slots = qc["qc_gap_h"].replace(0, np.nan).fillna(cfg["data"]["interval_minutes"] / 60) / (
        cfg["data"]["interval_minutes"] / 60)
    for v in VARS:
        s = SHORT[v]
        x = qc[f"x_{s}"]
        eps = EPS[s]
        med_l = x.rolling(lw, closed="left", min_periods=mp).median()
        mad_l = (x - med_l).abs().rolling(lw, closed="left", min_periods=mp).median()
        f[f"{s}_rz24"] = (x - med_l) / (1.4826 * mad_l + eps)
        # station-specific time-of-day baseline: median of the same clock time on the past 7 days
        lags = np.column_stack([_lookup(x, idx - pd.Timedelta(days=k), "20min") for k in range(1, 8)])
        with np.errstate(all="ignore"):
            n_ok = np.isfinite(lags).sum(axis=1)
            tod_med = np.where(n_ok >= 3, np.nanmedian(np.where(n_ok[:, None] > 0, lags, 0.0), axis=1), np.nan)
        tod_med = pd.Series(tod_med, index=idx)
        tod_dev = x - tod_med
        tod_mad = (tod_dev - tod_dev.rolling("3D", closed="left", min_periods=6).median()).abs() \
            .rolling("3D", closed="left", min_periods=6).median()
        f[f"{s}_tod_base"] = tod_med
        f[f"{s}_tod_dev"] = tod_dev
        f[f"{s}_tod_z"] = tod_dev / (1.4826 * tod_mad + eps)
        # decision-rule context (not model inputs): robust scale and envelope of the deviation over the persistence window
        f[f"{s}_tod_scale"] = 1.4826 * tod_mad + eps
        pw = fc["persistence_window"]
        f[f"{s}_dev_lo_win"] = tod_dev.rolling(pw, min_periods=3).min()
        f[f"{s}_dev_hi_win"] = tod_dev.rolling(pw, min_periods=3).max()
        prev = x.ffill().shift(1)
        rate = (x - prev) / dt_slots
        f[f"{s}_rate"] = rate
        rate_mad = rate.abs().rolling(lw, closed="left", min_periods=mp).median()
        f[f"{s}_rate_z"] = rate / (1.4826 * rate_mad + eps)
        f[f"{s}_accel"] = rate - rate.shift(1)
        f[f"{s}_slope3h"] = _rolling_slope(x, sw)
        f[f"{s}_slope24h"] = _rolling_slope(x, lw)
        f[f"{s}_vol3h"] = rate.rolling(sw, min_periods=2).std()
        changed = (x.diff().abs().fillna(1.0) > 0).to_numpy() | x.isna().to_numpy()
        f[f"{s}_run_h"] = run_length(changed) * cfg["data"]["interval_minutes"] / 60
        f[f"{s}_dev_mean6h"] = tod_dev.rolling("6h", min_periods=2).mean()
        f[f"{s}_dev_mean24h"] = tod_dev.rolling(lw, min_periods=6).mean()
        f[f"{s}_missing"] = x.isna().astype(float)
    # multivariate consistency (derived only from T, P, RH)
    t, rh = qc["x_t"], qc["x_rh"]
    td = dewpoint(t, rh)
    td_med = td.rolling(sw, closed="left", min_periods=2).median()
    f["mv_td_rate"] = td - td.ffill().shift(1)
    f["mv_rh_resid"] = rh - rh_from(t, td_med).clip(0, 100)  # RH expected if moisture stayed constant
    zc = [f"{SHORT[v]}_tod_z" for v in VARS]
    f["mv_joint_z"] = np.sqrt((f[zc].fillna(0) ** 2).sum(axis=1))
    f["mv_n_dev"] = (f[zc].abs() > 3).sum(axis=1).astype(float)
    rz = [f"{SHORT[v]}_rate_z" for v in VARS]
    f["mv_joint_rate"] = np.sqrt((f[rz].fillna(0) ** 2).sum(axis=1))
    f["mv_flat_all_h"] = qc["qc_flat_all_run"] * cfg["data"]["interval_minutes"] / 60
    f["gap_h"] = qc["qc_gap_h"]
    return pd.concat([f, qc], axis=1)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def neighbors(stations: dict[str, dict], cfg: dict | None = None) -> dict[str, list[tuple[str, float]]]:
    """Configurable neighbour selection: stations within radius, nearest first, capped."""
    fc = (cfg or load_config())["features"]
    out = {}
    for sid, m in stations.items():
        d = sorted((haversine_km((m["latitude"], m["longitude"]), (o["latitude"], o["longitude"])), oid)
                   for oid, o in stations.items() if oid != sid)
        out[sid] = [(oid, dist) for dist, oid in d if dist <= fc["neighbor_radius_km"]][: fc["neighbor_max"]]
    return out


SPATIAL_COLS = [f"{SHORT[v]}_{k}" for v in VARS for k in ("sp_resid", "sp_z", "sp_frac", "sp_wdiff", "sp_rate_resid")] + ["sp_n"]
# spatial context used by decision rules / explanations only (not model inputs)
NBMED_COLS = [f"{SHORT[v]}_{k}" for v in VARS for k in ("sp_nbmed", "sp_frac_lag", "sp_outliers")]


def spatial_features(own: pd.DataFrame, nbrs: list[tuple[pd.DataFrame, float]], cfg: dict | None = None) -> pd.DataFrame:
    """Stage-2 spatial features for `own` (stage-1 frame) given neighbour stage-1 frames and distances.

    Missing neighbours are fine: features become NaN and sp_n = 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN medians are expected (neighbour gaps)
        return _spatial(own, nbrs, cfg)


def _spatial(own, nbrs, cfg):
    cfg = cfg or load_config()
    tol = f"{cfg['features']['neighbor_time_tolerance_minutes']}min"
    idx = own.index
    out = pd.DataFrame(index=idx)
    if not nbrs:
        for c in SPATIAL_COLS + NBMED_COLS:
            out[c] = 0.0 if c == "sp_n" else np.nan
        return out
    w = np.array([1.0 / max(d, 1.0) for _, d in nbrs])
    n_avail = None
    lag = idx - pd.Timedelta(minutes=cfg["data"]["interval_minutes"])
    for v in VARS:
        s = SHORT[v]
        dev = np.column_stack([_lookup(nf[f"{s}_tod_dev"], idx, tol) for nf, _ in nbrs])
        dev_prev = np.column_stack([_lookup(nf[f"{s}_tod_dev"], lag, tol) for nf, _ in nbrs])
        rate = np.column_stack([_lookup(nf[f"{s}_rate"], idx, tol) for nf, _ in nbrs])
        with np.errstate(all="ignore"):
            # robust neighbourhood: a neighbour far from the others (e.g. itself drifting) is excluded
            med0 = np.nanmedian(dev, axis=1)
            mad0 = np.nanmedian(np.abs(dev - med0[:, None]), axis=1)
            outlier = (np.isfinite(dev).sum(axis=1, keepdims=True) >= 3) &                 (np.abs(dev - med0[:, None]) > 3 * 1.4826 * np.nan_to_num(mad0, nan=0.0)[:, None] + 4 * EPS[s])
            out[f"{s}_sp_outliers"] = outlier.sum(axis=1).astype(float)
            dev = np.where(outlier, np.nan, dev)
            ok = np.isfinite(dev)
            n = ok.sum(axis=1)
            n_avail = n if n_avail is None else np.maximum(n_avail, n)
            nb_med = np.where(n > 0, np.nanmedian(np.where(n[:, None] > 0, dev, 0.0), axis=1), np.nan)
            nb_mad = np.where(n > 1, np.nanmedian(np.abs(dev - nb_med[:, None]), axis=1), np.nan)
            own_dev = own[f"{s}_tod_dev"].to_numpy()
            out[f"{s}_sp_nbmed"] = nb_med  # used for correction/explanation only, not a model input
            resid = own_dev - nb_med
            out[f"{s}_sp_resid"] = resid
            out[f"{s}_sp_z"] = resid / (1.4826 * np.nan_to_num(nb_mad, nan=0.0) + 2 * EPS[s])
            big = np.abs(own_dev) > 2 * EPS[s]

            def share(d):
                similar = (np.sign(d) == np.sign(own_dev)[:, None]) & (np.abs(d) >= 0.5 * np.abs(own_dev)[:, None]) & np.isfinite(d)
                return np.where(n > 0, np.where(big, similar.sum(axis=1) / np.maximum(n, 1), 1.0), np.nan)
            out[f"{s}_sp_frac"] = share(dev)
            # temporal alignment: neighbours may lead by one observation interval
            out[f"{s}_sp_frac_lag"] = np.fmax(share(dev), share(np.where(outlier, np.nan, dev_prev)))
            wd = np.nansum(w * (own_dev[:, None] - dev) * ok, axis=1) / np.maximum((w * ok).sum(axis=1), 1e-9)
            out[f"{s}_sp_wdiff"] = np.where(n > 0, wd, np.nan)
            rn = np.isfinite(rate).sum(axis=1)
            nb_rate = np.where(rn > 0, np.nanmedian(np.where(rn[:, None] > 0, rate, 0.0), axis=1), np.nan)
            out[f"{s}_sp_rate_resid"] = own[f"{s}_rate"].to_numpy() - nb_rate
    out["sp_n"] = n_avail.astype(float)
    return out


TEMPORAL_COLS = [f"{SHORT[v]}_{k}" for v in VARS for k in (
    "rz24", "tod_dev", "tod_z", "rate", "rate_z", "accel", "slope3h", "slope24h", "vol3h", "run_h",
    "dev_mean6h", "dev_mean24h", "missing")] + ["gap_h"]
MV_COLS = ["mv_td_rate", "mv_rh_resid", "mv_joint_z", "mv_n_dev", "mv_joint_rate", "mv_flat_all_h"]
GROUPS = {"temporal": TEMPORAL_COLS, "multivariate": MV_COLS, "spatial": SPATIAL_COLS}


def build_features(df: pd.DataFrame, stations: dict[str, dict], cfg: dict | None = None) -> pd.DataFrame:
    """Batch feature build for a canonical multi-station frame. Returns rows aligned to df (same order)."""
    cfg = cfg or load_config()
    nb = neighbors({s: stations[s] for s in df["station_id"].unique()}, cfg)
    stage1 = {}
    for sid, g in df.groupby("station_id", sort=False):
        raw = g.set_index("timestamp").sort_index()[VARS]
        stage1[sid] = station_features(raw, cfg)
    parts = []
    for sid, g in df.groupby("station_id", sort=False):
        s1 = stage1[sid]
        sp = spatial_features(s1, [(stage1[o], d) for o, d in nb[sid] if o in stage1], cfg)
        part = pd.concat([s1, sp], axis=1)
        part.insert(0, "station_id", sid)
        parts.append(part.reset_index(names="timestamp"))
    feats = pd.concat(parts, ignore_index=True)
    key = df[["station_id", "timestamp"]].reset_index(names="_row")
    return key.merge(feats, on=["station_id", "timestamp"], how="left").sort_values("_row").drop(columns="_row").reset_index(drop=True)

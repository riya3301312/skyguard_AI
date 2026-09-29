"""Deterministic anomaly injection engine with preserved ground truth.

Every injected row keeps its original clean values in `clean_<var>` and the
primary parameter's clean value in `ground_truth_value`. Output depends only on
(input data, seed, profile).

Injection kinds -> observable fault class (the evaluation target):
  positive_spike -> SPIKE              negative_spike -> NEGATIVE_SPIKE
  pressure_spike -> SPIKE/NEGATIVE_SPIKE (by sign)
  flatline -> FLATLINE                 drift -> DRIFT
  step -> STEP_OFFSET                  missing_block -> MISSING_COMMUNICATION
  sentinel -> SENTINEL_CORRUPTION      humidity_corruption -> MULTIVARIATE_INCONSISTENCY
  multivariate_inconsistency -> MULTIVARIATE_INCONSISTENCY
  spatial_inconsistency -> SPATIAL_INCONSISTENCY
  composite -> COMPOSITE_FAULT
  regional_event -> NORMAL (a plausible weather event across neighbours; must NOT be flagged)
"""
import numpy as np
import pandas as pd

from ml.config import load_config
from ml.features import dewpoint, neighbors, rh_from
from ml.schema import VARS

KINDS = ["positive_spike", "negative_spike", "pressure_spike", "flatline", "drift", "step", "missing_block",
         "sentinel", "humidity_corruption", "multivariate_inconsistency", "spatial_inconsistency", "composite"]

# Magnitude / duration profiles (durations in rows of the canonical grid, hourly by default). Train, validation and test differ on purpose.
PROFILES = {
    "train": dict(spike_t=(6, 15), spike_rh=(25, 45), spike_p=(4, 10), spike_len=(1, 1),
                  flat_len=(6, 16), drift_len=(24, 72), drift_t=(3, 7), drift_p=(3, 8), drift_rh=(12, 25),
                  step_len=(6, 48), step_t=(3, 7), step_p=(3, 8), step_rh=(12, 25), miss_len=(2, 12),
                  sent_len=(1, 2), hum_len=(2, 6), mv_len=(2, 6), mv_t=(3, 5), mv_rh=(18, 30),
                  sp_len=(3, 8), sp_t=(4, 7), regional_t=(4, 8), regional_len=(3, 8)),
    "val": dict(spike_t=(5, 18), spike_rh=(20, 50), spike_p=(3, 12), spike_len=(1, 2),
                flat_len=(5, 20), drift_len=(18, 96), drift_t=(2.5, 8), drift_p=(2.5, 9), drift_rh=(10, 28),
                step_len=(4, 60), step_t=(2.5, 8), step_p=(2.5, 9), step_rh=(10, 28), miss_len=(1, 18),
                sent_len=(1, 3), hum_len=(2, 8), mv_len=(2, 8), mv_t=(2.5, 5.5), mv_rh=(15, 32),
                sp_len=(2, 10), sp_t=(3.5, 7.5), regional_t=(3.5, 9), regional_len=(2, 10)),
    "test": dict(spike_t=(4, 25), spike_rh=(18, 60), spike_p=(3, 15), spike_len=(1, 3),
                 flat_len=(4, 24), drift_len=(12, 120), drift_t=(2, 10), drift_p=(2, 10), drift_rh=(8, 30),
                 step_len=(3, 72), step_t=(2, 10), step_p=(2, 10), step_rh=(8, 30), miss_len=(1, 24),
                 sent_len=(1, 4), hum_len=(2, 10), mv_len=(2, 10), mv_t=(2.5, 6), mv_rh=(15, 35),
                 sp_len=(2, 12), sp_t=(3, 8), regional_t=(3, 10), regional_len=(2, 12)),
}
SEASON = {1: "winter", 2: "winter", 3: "pre-monsoon", 4: "pre-monsoon", 5: "pre-monsoon", 6: "monsoon",
          7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post-monsoon", 11: "post-monsoon", 12: "post-monsoon"}
PARAM_KEY = {"temperature_c": "t", "pressure_hpa": "p", "relative_humidity_pct": "rh"}


class _Station:
    def __init__(self, frame: pd.DataFrame):
        self.idx = frame.index.to_numpy()  # positions in the full frame
        self.vals = {v: frame[v].to_numpy(dtype=float).copy() for v in VARS}
        self.busy = np.zeros(len(frame), dtype=bool)
        self.valid = ~np.isnan(np.column_stack([self.vals[v] for v in VARS])).any(axis=1)


def _sev(q: float) -> str:
    return "LOW" if q < 1 / 3 else "MEDIUM" if q < 2 / 3 else "HIGH"


def inject(df: pd.DataFrame, seed: int, profile: str = "train", stations_meta: dict | None = None,
           cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (benchmark frame, event log). `df` is a canonical frame on a regular per-station grid."""
    cfg = cfg or load_config()
    P = PROFILES[profile]
    warm = cfg["injection"]["warmup_days"] * 24 * 60 // cfg["data"]["interval_minutes"]  # never inject into warm-up
    rng = np.random.default_rng(seed)
    out = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True).copy()
    for v in VARS:
        out[f"clean_{v}"] = out[v]
    out["is_anomaly"] = 0
    out["anomaly_type"] = "NORMAL"
    out["injection_kind"] = ""
    out["severity"] = ""
    out["injected_parameter"] = ""
    out["ground_truth_value"] = np.nan
    out["injection_id"] = ""
    # real long gaps in the source data are genuine missing-communication rows (not injected)
    real_gap = out[VARS].isna().all(axis=1)
    out.loc[real_gap, ["is_anomaly", "anomaly_type", "injection_id"]] = [1, "MISSING_COMMUNICATION", "REAL_GAP"]

    meta = stations_meta or {s: {"latitude": g["latitude"].iloc[0], "longitude": g["longitude"].iloc[0]}
                             for s, g in out.groupby("station_id")}
    nb = neighbors({s: meta[s] for s in out["station_id"].unique()}, cfg)
    st = {s: _Station(g) for s, g in out.groupby("station_id", sort=True)}
    events = []
    counter = [0]

    def label(sid, a, b, kind, cls, param, sev, gt_param):
        s = st[sid]
        rows = s.idx[a:b]
        iid = f"s{seed}-{counter[0]:05d}"
        counter[0] += 1
        out.loc[rows, ["is_anomaly", "anomaly_type", "injection_kind", "severity", "injected_parameter", "injection_id"]] = \
            [0 if cls == "NORMAL" else 1, cls, kind, sev, param, iid]
        if gt_param:
            out.loc[rows, "ground_truth_value"] = out.loc[rows, f"clean_{gt_param}"].to_numpy()
        s.busy[max(a - 4, 0): b + 4] = True
        t0 = out.at[rows[0], "timestamp"]
        events.append(dict(injection_id=iid, station_id=sid, kind=kind, anomaly_type=cls, parameter=param,
                           severity=sev, start=t0, end=out.at[rows[-1], "timestamp"], n_rows=len(rows),
                           season=SEASON[t0.month], seed=seed, profile=profile))
        return iid

    def pick_window(sid, length, tries=40):
        s = st[sid]
        n = len(s.idx)
        for _ in range(tries):
            a = int(rng.integers(warm, max(warm + 1, n - length - 1)))
            b = a + length
            if b <= n and not s.busy[a - 1:b + 1].any() and s.valid[a - 1:b].mean() > 0.9:
                return a, b
        return None

    def mag(key):
        lo, hi = P[key]
        m = float(rng.uniform(lo, hi))
        return m, (m - lo) / max(hi - lo, 1e-9)

    def ilen(key):
        lo, hi = P[key]
        return int(rng.integers(lo, hi + 1))

    def apply_regional(anchor):
        members = [anchor] + [o for o, _ in nb[anchor]]
        length = ilen("regional_len")
        w = pick_window(anchor, length + 4)
        if w is None:
            return
        a, _ = w
        t_start = out.at[st[anchor].idx[a], "timestamp"]
        m, q = mag("regional_t")
        sign = -1.0 if rng.random() < 0.7 else 1.0  # mostly convective cold pools, sometimes warm surges
        for sid in members:
            s = st[sid]
            j = int(pd.DatetimeIndex(out.loc[s.idx, "timestamp"]).searchsorted(t_start))
            j += int(rng.integers(0, 2))  # small arrival lag between stations
            b = j + length
            if j < warm or b >= len(s.idx) or s.busy[j - 1:b + 1].any() or s.valid[j:b].mean() < 0.9:
                continue
            k = np.arange(b - j)
            shape = np.minimum(1.0, (k + 1) / 2.0) * np.clip((b - j - k) / max((b - j) * 0.6, 1), 0, 1)
            dm = sign * m * float(rng.uniform(0.8, 1.2)) * shape
            t0 = s.vals["temperature_c"][j:b].copy()
            td0 = dewpoint(pd.Series(t0), pd.Series(s.vals["relative_humidity_pct"][j:b])).to_numpy()
            t1 = t0 + dm
            s.vals["temperature_c"][j:b] = np.round(t1, 1)
            s.vals["relative_humidity_pct"][j:b] = np.round(np.clip(rh_from(pd.Series(t1), pd.Series(td0 + 0.3 * np.abs(dm))).to_numpy(), 1, 100), 1)
            s.vals["pressure_hpa"][j:b] = np.round(s.vals["pressure_hpa"][j:b] - 0.3 * dm * float(rng.uniform(0.5, 1.0)), 1)
            label(sid, j, b, "regional_event", "NORMAL", "temperature_c+pressure_hpa+relative_humidity_pct", _sev(q), "")

    def apply(sid, kind):
        s = st[sid]
        V = s.vals
        if kind in ("positive_spike", "negative_spike"):
            L = ilen("spike_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            param = rng.choice(["temperature_c", "relative_humidity_pct"])
            sign = 1.0 if kind == "positive_spike" else -1.0
            if param == "relative_humidity_pct":
                room = (100 - V[param][a:b]).min() if sign > 0 else V[param][a:b].min()
                if room < P["spike_rh"][0]:
                    param = "temperature_c"
            m, q = mag("spike_t" if param == "temperature_c" else "spike_rh")
            V[param][a:b] = np.clip(V[param][a:b] + sign * m, 0 if "humidity" in param else -80, 100 if "humidity" in param else 80)
            label(sid, a, b, kind, "SPIKE" if sign > 0 else "NEGATIVE_SPIKE", param, _sev(q), param)
        elif kind == "pressure_spike":
            L = ilen("spike_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            m, q = mag("spike_p")
            sign = 1.0 if rng.random() < 0.5 else -1.0
            V["pressure_hpa"][a:b] += sign * m
            label(sid, a, b, kind, "SPIKE" if sign > 0 else "NEGATIVE_SPIKE", "pressure_hpa", _sev(q), "pressure_hpa")
        elif kind == "flatline":
            L = ilen("flat_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            params = list(VARS) if rng.random() < 0.3 else [str(rng.choice(VARS))]
            for p in params:
                V[p][a:b] = V[p][a - 1]
            lo, hi = P["flat_len"]
            label(sid, a, b, kind, "FLATLINE", "+".join(params), _sev((L - lo) / max(hi - lo, 1)), params[0])
        elif kind in ("drift", "step"):
            L = ilen(f"{kind}_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            param = str(rng.choice(VARS))
            m, q = mag(f"{kind}_{PARAM_KEY[param]}")
            sign = 1.0 if rng.random() < 0.5 else -1.0
            off = sign * m * (np.arange(1, L + 1) / L if kind == "drift" else np.ones(L))
            x = V[param][a:b] + off
            V[param][a:b] = np.clip(x, 0, 100) if "humidity" in param else x
            label(sid, a, b, kind, "DRIFT" if kind == "drift" else "STEP_OFFSET", param, _sev(q), param)
        elif kind == "missing_block":
            L = ilen("miss_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            for p in VARS:
                V[p][a:b] = np.nan
            lo, hi = P["miss_len"]
            label(sid, a, b, kind, "MISSING_COMMUNICATION", "+".join(VARS), _sev((L - lo) / max(hi - lo, 1)), "")
        elif kind == "sentinel":
            L = ilen("sent_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            param = str(rng.choice(VARS))
            V[param][a:b] = float(rng.choice(cfg["qc"]["sentinels"]))
            label(sid, a, b, kind, "SENTINEL_CORRUPTION", param, "HIGH", param)
        elif kind == "humidity_corruption":
            L = ilen("hum_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            noise = rng.normal(0, 22, L)
            noise = np.where(np.abs(noise) < 10, np.sign(noise + 1e-9) * 10, noise)
            V["relative_humidity_pct"][a:b] = np.round(np.clip(V["relative_humidity_pct"][a:b] + noise, 3, 100), 1)
            label(sid, a, b, kind, "MULTIVARIATE_INCONSISTENCY", "relative_humidity_pct", "MEDIUM", "relative_humidity_pct")
        elif kind == "multivariate_inconsistency":
            L = ilen("mv_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            if rng.random() < 0.5:  # temperature rises but RH does not respond (moisture would have to jump)
                m, q = mag("mv_t")
                V["temperature_c"][a:b] += m
                param = "temperature_c"
            else:  # RH jumps with temperature unchanged
                m, q = mag("mv_rh")
                sign = 1.0 if V["relative_humidity_pct"][a:b].max() + m <= 100 else -1.0
                V["relative_humidity_pct"][a:b] = np.clip(V["relative_humidity_pct"][a:b] + sign * m, 1, 100)
                param = "relative_humidity_pct"
            label(sid, a, b, kind, "MULTIVARIATE_INCONSISTENCY", param, _sev(q), param)
        elif kind == "spatial_inconsistency":
            if len(nb[sid]) < 2:
                return
            L = ilen("sp_len")
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            m, q = mag("sp_t")
            sign = 1.0 if rng.random() < 0.6 else -1.0
            k = np.arange(L)
            dm = sign * m * np.minimum(1.0, (k + 1) / 3.0)  # smooth onset: no temporal spike
            t0 = V["temperature_c"][a:b].copy()
            td0 = dewpoint(pd.Series(t0), pd.Series(V["relative_humidity_pct"][a:b])).to_numpy()
            V["temperature_c"][a:b] = t0 + dm
            V["relative_humidity_pct"][a:b] = np.clip(rh_from(pd.Series(t0 + dm), pd.Series(td0)).to_numpy(), 1, 100)
            label(sid, a, b, kind, "SPATIAL_INCONSISTENCY", "temperature_c", _sev(q), "temperature_c")
        elif kind == "composite":
            L = max(ilen("step_len") // 2, 4)
            w = pick_window(sid, L)
            if not w:
                return
            a, b = w
            m1, q1 = mag("spike_t")
            V["temperature_c"][a] += m1  # temperature spike at onset ...
            m2, q2 = mag("step_p")
            V["pressure_hpa"][a:b] += (1 if rng.random() < 0.5 else -1) * m2  # ... plus pressure offset
            params = "temperature_c+pressure_hpa"
            if rng.random() < 0.5:
                V["relative_humidity_pct"][a:b] = V["relative_humidity_pct"][a - 1]  # ... plus frozen RH
                params += "+relative_humidity_pct"
            label(sid, a, b, kind, "COMPOSITE_FAULT", params, _sev(max(q1, q2)), "pressure_hpa")

    months = max(1, round((out["timestamp"].max() - out["timestamp"].min()).days / 30))
    n_events = cfg["injection"]["events_per_station_month"] * months
    order = sorted(st)
    # regional weather events first (they need free windows on several stations)
    for sid in order:
        if len(nb[sid]) >= 2:
            for _ in range(max(1, months // 2)):
                apply_regional(sid)
    for sid in order:
        for kind in rng.choice(KINDS, size=n_events):
            apply(sid, str(kind))

    res = cfg["injection"]["resolution"]
    sentinels = np.array(cfg["qc"]["sentinels"], dtype=float)
    for sid, s in st.items():
        for v in VARS:
            x = s.vals[v]
            orig = out.loc[s.idx, v].to_numpy(dtype=float)
            changed = ~np.isclose(x, orig, equal_nan=True) & ~np.isin(x, sentinels) & np.isfinite(x)
            q = np.where(changed, np.round(np.round(x / res[v]) * res[v], 1), x)  # injected cells only
            out.loc[s.idx, v] = q
    return out, pd.DataFrame(events)

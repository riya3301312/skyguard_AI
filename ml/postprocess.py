"""Severity, sensor health, maintenance risk, correction recommendation and explanations.

All functions are deterministic and operate on one station's result frame in time
order, so batch evaluation and the real-time engine share them.
"""
import math

import numpy as np
import pandas as pd

from .config import load_config
from .schema import LABEL, SHORT, UNIT, VARS

PATTERN_TEXT = {"SPIKE": "spike", "NEGATIVE_SPIKE": "negative spike", "FLATLINE": "flatline / frozen value",
                "DRIFT": "gradual drift", "STEP_OFFSET": "step / offset", "MISSING_COMMUNICATION": "missing data / communication gap",
                "SENTINEL_CORRUPTION": "corrupt value (sentinel / impossible range)",
                "MULTIVARIATE_INCONSISTENCY": "multivariate inconsistency", "SPATIAL_INCONSISTENCY": "spatial inconsistency",
                "COMPOSITE_FAULT": "composite fault", "UNKNOWN_SUSPECT": "unclassified suspect", "NORMAL": "normal"}


def run_count(flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(flag))
    c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0
        out[i] = c
    return out


def severity(R: pd.DataFrame, cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """score = 40*p + 20*min(|z|/10,1) + 15*min(persist/6,1) + 10*(n_vars-1)/2 + 10*isolation + 5*qc_hard
    (weights and level thresholds from config [severity])."""
    c = (cfg or load_config())["severity"]
    w, th = c["weights"], c["thresholds"]
    flag = R["flag"].to_numpy(bool)
    z = np.zeros(len(R))
    iso = np.full(len(R), 0.5)
    for v in VARS:
        s = SHORT[v]
        sel = R["param"].to_numpy() == v
        z = np.where(sel, R[f"{s}_tod_z"].abs().fillna(R[f"{s}_rz24"].abs()).fillna(0).to_numpy(), z)
        frac = R[f"{s}_sp_frac"].to_numpy()
        iso = np.where(sel & (R["sp_n"].fillna(0).to_numpy() >= 2) & np.isfinite(frac), 1 - np.nan_to_num(frac), iso)
    other = ~np.isin(R["param"].to_numpy(), VARS)
    z = np.where(other, R[[f"{SHORT[v]}_tod_z" for v in VARS]].abs().max(axis=1).fillna(0).to_numpy(), z)
    hard = R["qc_hard"].to_numpy(bool)
    z = np.where(hard, 10.0, z)
    nvars = np.clip(R["mv_n_dev"].fillna(0).to_numpy(), 1, 3)
    persist = run_count(flag)
    score = (w["probability"] * R["prob"].to_numpy() + w["magnitude"] * np.minimum(z / 10, 1)
             + w["persistence"] * np.minimum(persist / 6, 1) + w["variables"] * (nvars - 1) / 2
             + w["isolation"] * iso + w["qc"] * hard)
    level = np.select([score >= th["CRITICAL"], score >= th["HIGH"], score >= th["MEDIUM"]], ["CRITICAL", "HIGH", "MEDIUM"], "LOW")
    return np.where(flag, score, 0.0), np.where(flag, level, "")


def health(R: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Deterministic 0-100 data-quality health score over a rolling window (not a hardware diagnostic)."""
    cfg = cfg or load_config()
    h = cfg["health"]
    win = h["window"]
    step_h = cfg["data"]["interval_minutes"] / 60
    pat = R["pattern"]
    flag = R["flag"].astype(float)
    missing = (pat == "MISSING_COMMUNICATION").astype(float)
    rates = pd.DataFrame({
        "anomaly_rate": (flag - missing).clip(0),
        "missing_rate": missing,
        "flatline_rate": (pat == "FLATLINE").astype(float),
        "drift": pat.isin(["DRIFT", "STEP_OFFSET"]).astype(float),
        "comm_gaps": (R["gap_h"].fillna(0) - step_h).clip(0),
    }, index=R.index)
    roll = rates.rolling(win, min_periods=1)
    agg = roll.mean()
    agg["drift"] = roll["drift"].max()
    agg["comm_gaps"] = (roll["comm_gaps"].sum() / pd.Timedelta(win).total_seconds() * 3600).clip(0, 1)
    penalty = sum(h[k] * agg[k] for k in ("anomaly_rate", "missing_rate", "flatline_rate", "drift", "comm_gaps"))
    agg["health_score"] = (100 - penalty).clip(0, 100).round(1)
    st = h["status"]
    agg["health_status"] = np.select([agg["health_score"] >= st["NORMAL"], agg["health_score"] >= st["WATCH"],
                                      agg["health_score"] >= st["DEGRADED"]], ["NORMAL", "WATCH", "DEGRADED"], "CRITICAL")
    return agg


def health_reasons(row: pd.Series) -> list[str]:
    r = []
    if row["anomaly_rate"] > 0.02:
        r.append(f"{row['anomaly_rate']:.0%} of recent observations flagged anomalous")
    if row["missing_rate"] > 0.02:
        r.append(f"{row['missing_rate']:.0%} of recent observations missing")
    if row["flatline_rate"] > 0:
        r.append("frozen / flatline values in the last 24 h")
    if row["drift"] > 0:
        r.append("drift or offset pattern in the last 24 h")
    if row["comm_gaps"] > 0.02:
        r.append(f"communication gaps covering {row['comm_gaps']:.0%} of the last 24 h")
    return r or ["no quality issues in the last 24 h"]


def maintenance_risk(R: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Conservative LOW/MEDIUM/HIGH with a reason; never a time-to-failure prediction."""
    m = (cfg or load_config())["maintenance"]
    if R.empty or (R.index[-1] - R.index[0]) < pd.Timedelta(days=m["min_history_days"]):
        return {"risk": "INSUFFICIENT_HISTORY", "reason": "Insufficient history for maintenance prediction."}
    end = R.index[-1]
    d = pd.Timedelta(days=m["recent_days"])
    recent = R[R.index > end - d]
    prior = R[(R.index <= end - d) & (R.index > end - 2 * d)]
    faults = lambda x: int((x["flag"] & (x["pattern"] != "MISSING_COMMUNICATION")).sum())  # noqa: E731
    n_recent, n_prior = faults(recent), faults(prior)
    persistent = recent["pattern"].isin(["DRIFT", "STEP_OFFSET", "FLATLINE"]).any()
    hs = float(R["health_score"].iloc[-1])
    if persistent and hs < 70:
        return {"risk": "HIGH", "reason": f"Persistent drift/offset/flatline pattern in the last {m['recent_days']} days and health {hs:.0f}/100. Inspect/recalibrate sensor."}
    if persistent or (n_recent >= 5 and n_recent > 2 * max(n_prior, 1)) or hs < 85:
        why = "persistent fault pattern present" if persistent else \
            f"anomaly count rose from {n_prior} to {n_recent} over consecutive {m['recent_days']}-day windows" if n_recent > 2 * max(n_prior, 1) else \
            f"health {hs:.0f}/100"
        return {"risk": "MEDIUM", "reason": f"Watch: {why}."}
    return {"risk": "LOW", "reason": f"{n_recent} sensor-fault alerts in the last {m['recent_days']} days; no persistent pattern."}


def corrections(R: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    """Recommendation-only estimates of the normal value. Raw values are never touched."""
    out = pd.DataFrame(index=R.index)
    flag = R["flag"].to_numpy(bool)
    for v in VARS:
        s = SHORT[v]
        affected = flag & ((R["param"] == v) | (R["param"] == "all") | R["param"].str.contains(v, regex=False)).to_numpy()
        x = R[f"x_{s}"].where(~affected)
        temporal = x.rolling("3h", closed="left", min_periods=1).median()
        temporal = temporal.fillna(x.rolling("24h", closed="left", min_periods=1).median()).fillna(R[f"{s}_tod_base"])
        nb = R[f"{s}_tod_base"] + R[f"{s}_sp_nbmed"]
        use_nb = (R["sp_n"].fillna(0) >= 2) & nb.notna()
        est = temporal if method == "temporal" else nb.where(use_nb, temporal)
        out[f"estimated_{v}"] = est.where(affected).round(2)
        out[f"_affected_{s}"] = affected
    aff = out[[f"_affected_{SHORT[v]}" for v in VARS]].to_numpy()
    ests = out[[f"estimated_{v}" for v in VARS]].to_numpy()
    out["replacement_recommended"] = (aff & np.isfinite(ests)).any(axis=1)
    return out.drop(columns=[f"_affected_{SHORT[v]}" for v in VARS])


def _f(x, nd=1):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def explain(row: pd.Series, contrib: dict | None, threshold: float, detector_mode: str) -> dict:
    """Human-readable evidence built only from the values the detector actually used."""
    ev = []
    param = row.get("param", "")
    pattern = row.get("pattern", "NORMAL")
    vs = [param] if param in VARS else list(VARS)
    if row.get("qc_all_missing"):
        ev.append("All three parameters are missing in this record (deterministic QC).")
    for v in VARS:
        s = SHORT[v]
        if row.get(f"qc_{s}_sentinel"):
            ev.append(f"{LABEL[v]} value {row[v]} is a known sentinel/error code (deterministic QC).")
        if row.get(f"qc_{s}_range"):
            ev.append(f"{LABEL[v]} value {row[v]} is outside the physical range (deterministic QC).")
    gap = row.get("gap_h", 0) or 0
    if gap * 60 > 90:
        ev.append(f"No observations for {gap:.1f} h before this record (observation-time gap).")
    for v in vs:
        s, u, L = SHORT[v], UNIT[v], LABEL[v]
        rate, rz, tz = row.get(f"{s}_rate"), row.get(f"{s}_rate_z"), row.get(f"{s}_tod_z")
        if rz is not None and math.isfinite(rz) and abs(rz) > 3:
            ev.append(f"{L} changed {rate:+.1f} {u} since the previous observation ({abs(rz):.1f}x its usual rate of change).")
        if tz is not None and math.isfinite(tz) and abs(tz) > 2:
            ev.append(f"{abs(tz):.1f} robust deviations {'above' if tz > 0 else 'below'} this station's usual {L.lower()} for this time of day "
                      f"(7-day baseline {row.get(f'{s}_tod_base'):.1f} {u}, observed {row.get(f'x_{s}'):.1f} {u}).")
        run = row.get(f"{s}_run_h") or 0
        if run >= 3:
            ev.append(f"{L} has not changed for {run:.1f} h.")
        dm = row.get(f"{s}_dev_mean24h")
        if pattern in ("DRIFT", "STEP_OFFSET") and dm is not None and math.isfinite(dm):
            ev.append(f"{L} has deviated persistently from its baseline: mean {dm:+.1f} {u} over 24 h, trend {24 * (row.get(f'{s}_slope24h') or 0):+.1f} {u}/day.")
        n = int(row.get("sp_n") or 0)
        if n > 0 and math.isfinite(row.get(f"{s}_sp_nbmed", np.nan)):
            frac = row.get(f"{s}_sp_frac")
            ev.append(f"Nearby stations (n={n}) deviated by a median {row[f'{s}_sp_nbmed']:+.1f} {u}; this station {row.get(f'{s}_tod_dev'):+.1f} {u}"
                      + (f"; {frac:.0%} of neighbours show a similar change." if frac is not None and math.isfinite(frac) else "."))
    if int(row.get("sp_n") or 0) == 0:
        ev.append("No spatial neighbours available: decision uses temporal and multivariate evidence only.")
    if param != "pressure_hpa":
        pr = row.get("p_rate_z")
        if pr is not None and math.isfinite(pr) and abs(pr) < 2:
            ev.append(f"Pressure showed no comparable change ({row.get('p_rate'):+.1f} hPa).")
    resid = row.get("mv_rh_resid")
    if resid is not None and math.isfinite(resid) and abs(resid) > 10:
        ev.append(f"Temperature/RH relationship is inconsistent: RH {row.get('x_rh'):.0f}% vs {row.get('x_rh') - resid:.0f}% expected if moisture were unchanged.")
    if row.get("regional_evidence"):
        ev.append("Regional evidence: most neighbours show a similar change, consistent with a weather event; sensor-fault probability reduced.")
    name = (LABEL.get(param, "Multi-parameter") + " " + PATTERN_TEXT.get(pattern, pattern)) if pattern != "NORMAL" else "none"
    contrib = contrib or {}
    top = sorted(((k, round(float(v), 3)) for k, v in contrib.items() if v > 0.01), key=lambda kv: -kv[1])
    return {
        "status": "ANOMALY DETECTED" if row.get("flag") else "NORMAL",
        "likely_fault_pattern": f"Likely fault pattern: {name}" if pattern != "NORMAL" else "No fault pattern",
        "pattern": pattern, "parameter": param,
        "anomaly_probability": _f(row.get("prob"), 3), "alert_threshold": round(threshold, 3),
        "severity": row.get("severity") or None, "detector_mode": detector_mode,
        "evidence": ev, "feature_group_contributions": top,
        "note": "Observable data pattern, not a confirmed physical hardware diagnosis.",
    }

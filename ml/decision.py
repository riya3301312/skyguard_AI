"""Weather-vs-sensor decision layer: deterministic, auditable, parameters selected on VALIDATION.

* regional_evidence(): how strongly the neighbourhood shares the station's deviation (regional weather)
  and whether the station nevertheless shows an exception (isolated fault during a regional event).
* persistence_confirmed(): a DRIFT / STEP_OFFSET alert needs sustained, one-signed, station-specific
  deviation of the attributed parameter - a single transient departure is not a drift.
"""
import numpy as np
import pandas as pd

from .schema import SHORT, VARS

PERSISTENT = ("DRIFT", "STEP_OFFSET")
DEFAULT_RULES = {"confirm_persistent": False, "k6": 1.5, "c3": 1.0, "max_shared": 0.5,
                 "regional_suppress": False, "regional_min": 0.5}


def _col(F: pd.DataFrame, c: str) -> np.ndarray:
    return F[c].to_numpy(float) if c in F else np.full(len(F), np.nan)


def regional_evidence(F: pd.DataFrame, regional_min: float = 0.5) -> pd.DataFrame:
    n = np.nan_to_num(_col(F, "sp_n"))
    cover = np.clip(n / 3.0, 0, 1)
    num = np.zeros(len(F))
    cnt = np.zeros(len(F))
    align = np.zeros(len(F))
    cons = np.zeros(len(F))
    best_frac = np.zeros(len(F))
    isolated = np.zeros(len(F), dtype=bool)
    iso_var = np.full(len(F), "", dtype=object)
    for v in VARS:
        s = SHORT[v]
        z = np.abs(np.nan_to_num(_col(F, f"{s}_tod_z")))
        dev = z > 2
        frac_lag = np.nan_to_num(_col(F, f"{s}_sp_frac_lag"), nan=np.nan_to_num(_col(F, f"{s}_sp_frac")))
        frac0 = np.nan_to_num(_col(F, f"{s}_sp_frac"))
        spz = np.abs(np.nan_to_num(_col(F, f"{s}_sp_z")))
        num += np.where(dev, frac_lag, 0)
        cnt += dev
        align += np.where(dev, np.where(frac_lag > 0, frac0 / np.maximum(frac_lag, 1e-9), 0), 0)
        cons += np.where(dev, 1 - np.minimum(spz / 5, 1), 0)
        best_frac = np.where(dev, np.maximum(best_frac, frac_lag), best_frac)
        iso = dev & (z > 3) & (frac_lag < 0.34) & (n >= 2)
        iso_var = np.where(iso & (iso_var == ""), v, iso_var)
        isolated |= iso
    with np.errstate(all="ignore"):
        score = np.where((cnt > 0) & (n >= 2), num / np.maximum(cnt, 1) * cover, 0.0)
    mv = np.abs(np.nan_to_num(_col(F, "mv_rh_resid"))) > 15
    hard = F["qc_hard"].to_numpy(bool) if "qc_hard" in F else np.zeros(len(F), dtype=bool)
    regional = score >= regional_min
    exception = regional & (isolated | mv | hard)
    reason = np.where(~exception, None, np.where(hard, "deterministic QC violation",
                      np.where(isolated, np.char.add("isolated deviation in ", iso_var.astype(str)),
                               "temperature/humidity relationship inconsistent")))
    return pd.DataFrame({
        "regional_score": np.round(score, 3), "stations_considered": n.astype(int),
        "stations_supporting_event": np.round(best_frac * n).astype(int),
        "temporal_alignment": np.round(np.where(cnt > 0, align / np.maximum(cnt, 1), 0), 3),
        "spatial_consistency": np.round(np.where(cnt > 0, cons / np.maximum(cnt, 1), 0), 3),
        "regional": regional, "station_exception": exception, "exception_reason": reason,
    }, index=F.index)


def persistence_confirmed(F: pd.DataFrame, pattern: np.ndarray, param: np.ndarray, rules: dict) -> np.ndarray:
    """True where the row is not a DRIFT/STEP alert, or where sustained station-specific evidence exists."""
    ok = np.ones(len(F), dtype=bool)
    persistent = np.isin(pattern, PERSISTENT)
    if not persistent.any():
        return ok
    n = np.nan_to_num(_col(F, "sp_n"))
    for v in VARS:
        s = SHORT[v]
        rows = persistent & (param == v)
        if not rows.any():
            continue
        scale = _col(F, f"{s}_tod_scale")
        dm6, dm24 = _col(F, f"{s}_dev_mean6h"), _col(F, f"{s}_dev_mean24h")
        lo, hi = _col(F, f"{s}_dev_lo_win"), _col(F, f"{s}_dev_hi_win")
        with np.errstate(invalid="ignore"):
            up = (dm6 > rules["k6"] * scale) & (lo > rules["c3"] * scale)
            down = (dm6 < -rules["k6"] * scale) & (hi < -rules["c3"] * scale)
            one_signed = np.where(pattern == "DRIFT", np.sign(dm24) == np.sign(dm6), True)
            shared = (n >= 2) & (np.nan_to_num(_col(F, f"{s}_sp_frac_lag")) >= rules["max_shared"])
        ok[rows] = ((up | down) & one_signed & ~shared)[rows]
    return ok


def apply_rules(F: pd.DataFrame, pattern: np.ndarray, param: np.ndarray, rules: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Demote alerts that the decision rules do not confirm. Deterministic-QC rows are never demoted."""
    if not rules:
        return pattern, param
    pattern, param = pattern.copy(), param.copy()
    hard = F["qc_hard"].to_numpy(bool)
    demote = np.zeros(len(F), dtype=bool)
    if rules.get("confirm_persistent"):
        demote |= ~persistence_confirmed(F, pattern, param, rules)
    if rules.get("regional_suppress"):
        ev = regional_evidence(F, rules["regional_min"])
        demote |= (pattern != "NORMAL") & ev["regional"].to_numpy() & ~ev["station_exception"].to_numpy()
    demote &= ~hard
    pattern[demote] = "NORMAL"
    param[demote] = ""
    return pattern, param


def weather_status(flag: np.ndarray, ev: pd.DataFrame) -> np.ndarray:
    reg, exc = ev["regional"].to_numpy(), ev["station_exception"].to_numpy()
    return np.select(
        [flag & reg & exc, flag & reg, flag, reg],
        ["REGIONAL_WEATHER_WITH_STATION_EXCEPTION", "UNCERTAIN", "SENSOR_ANOMALY", "REGIONAL_WEATHER"],
        "NORMAL")


EVIDENCE_COLS = ["regional_score", "stations_supporting_event", "stations_considered", "temporal_alignment",
                 "spatial_consistency", "station_exception", "exception_reason", "weather_status"]


def annotate(F: pd.DataFrame, flag: np.ndarray, regional_min: float = 0.5) -> pd.DataFrame:
    """Auditable weather-vs-sensor evidence for every row (added to results and explanations)."""
    ev = regional_evidence(F, regional_min)
    ev["weather_status"] = weather_status(np.asarray(flag, bool), ev)
    ev["regional_evidence"] = ev.pop("regional")
    return ev

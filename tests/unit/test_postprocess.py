import numpy as np
import pandas as pd

from ml.features import station_features
from ml.postprocess import corrections, explain, health, maintenance_risk, severity
from tests.conftest import station_frame


def results(frame, flags=None, pattern="SPIKE", param="temperature_c", prob=0.95):
    R = station_features(frame)
    for v in ("temperature_c", "pressure_hpa", "relative_humidity_pct"):
        R[v] = frame[v]
    for c in ("t_sp_frac", "p_sp_frac", "rh_sp_frac", "t_sp_nbmed", "p_sp_nbmed", "rh_sp_nbmed"):
        R[c] = np.nan
    R["sp_n"] = 0.0
    f = np.zeros(len(R), bool) if flags is None else flags
    R["flag"] = f
    R["prob"] = np.where(f, prob, 0.01)
    R["pattern"] = np.where(f, pattern, "NORMAL")
    R["param"] = np.where(f, param, "")
    return R


def test_severity_increases_with_probability_and_levels(frame):
    f = frame.copy()
    i = 24 * 8
    f.iloc[i, 0] += 20
    flags = np.zeros(len(f), bool)
    flags[i] = True
    hi = severity(results(f, flags, prob=0.99))
    lo = severity(results(f, flags, prob=0.55))
    assert hi[0][i] > lo[0][i]
    assert hi[1][i] in ("HIGH", "CRITICAL") and hi[1][0] == ""


def test_health_score_and_status(frame):
    ok = health(results(frame))
    assert ok["health_score"].iloc[-1] == 100 and ok["health_status"].iloc[-1] == "NORMAL"
    flags = np.zeros(len(frame), bool)
    flags[-12:] = True
    bad = health(results(frame, flags, pattern="FLATLINE"))
    assert bad["health_score"].iloc[-1] < 70 and bad["health_status"].iloc[-1] in ("DEGRADED", "CRITICAL")


def test_maintenance_conservative():
    short = results(station_frame(2))
    short = short.join(health(short))
    assert maintenance_risk(short)["risk"] == "INSUFFICIENT_HISTORY"
    f = station_frame(15)
    flags = np.zeros(len(f), bool)
    flags[-30:] = True
    R = results(f, flags, pattern="DRIFT", param="pressure_hpa")
    R = R.join(health(R))
    m = maintenance_risk(R)
    assert m["risk"] in ("HIGH", "MEDIUM") and "reason" in m and "fail" not in m["reason"].lower()


def test_correction_estimates_without_touching_raw(frame):
    f = frame.copy()
    i = 24 * 8 + 4
    truth = f.iloc[i, 0]
    f.iloc[i, 0] = 55.2
    flags = np.zeros(len(f), bool)
    flags[i] = True
    R = results(f, flags)
    raw_before = R["temperature_c"].copy()
    c = corrections(R)
    assert abs(c["estimated_temperature_c"].iloc[i] - truth) < 4
    assert c["replacement_recommended"].iloc[i] and not c["replacement_recommended"].iloc[i - 1]
    pd.testing.assert_series_equal(R["temperature_c"], raw_before)
    assert R["temperature_c"].iloc[i] == 55.2


def test_explanation_is_evidence_based_and_hedged(frame):
    f = frame.copy()
    i = 24 * 8 + 4
    f.iloc[i, 0] += 18
    flags = np.zeros(len(f), bool)
    flags[i] = True
    R = results(f, flags)
    R["severity"] = "HIGH"
    e = explain(R.iloc[i], {"temporal_temperature": 0.6, "spatial": 0.0}, 0.5, "hybrid")
    assert e["likely_fault_pattern"] == "Likely fault pattern: Temperature spike"
    assert any("changed +" in x for x in e["evidence"]) and any("No spatial neighbours" in x for x in e["evidence"])
    assert "not a confirmed physical hardware" in e["note"]
    assert e["feature_group_contributions"] == [("temporal_temperature", 0.6)]

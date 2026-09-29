"""Regression tests for the weather-vs-sensor decision layer and persistence-aware drift/step confirmation."""
import numpy as np
import pandas as pd

from ml.decision import DEFAULT_RULES, annotate, apply_rules, persistence_confirmed, regional_evidence
from ml.features import dewpoint, rh_from, spatial_features, station_features
from ml.postprocess import explain
from tests.conftest import station_frame

RULES = {**DEFAULT_RULES, "confirm_persistent": True, "k6": 1.0, "c3": 0.5, "regional_suppress": True}
I = 24 * 10 + 6  # evaluated row (after 10 days of history)
DIST = {"B": 20.0, "C": 35.0, "D": 50.0}


def network(modify=None):
    """Station A with three neighbours; `modify(frames)` edits the raw series in place."""
    fr = {s: station_frame(12, seed=i) for i, s in enumerate("ABCD")}
    if modify:
        modify(fr)
    s1 = {k: station_features(v) for k, v in fr.items()}
    sp = spatial_features(s1["A"], [(s1[k], d) for k, d in DIST.items()])
    return pd.concat([s1["A"], sp], axis=1)


def cool(frame, i, dt=-7.0):
    """Physically consistent regional cooling: temperature drops, dew point stays, so RH rises."""
    t, rh = frame.iloc[i, 0], frame.iloc[i, 2]
    td = float(dewpoint(pd.Series([t]), pd.Series([rh]))[0])
    frame.iloc[i, 0] = t + dt
    frame.iloc[i, 2] = float(np.clip(rh_from(pd.Series([t + dt]), pd.Series([td]))[0], 1, 100))


def arr(n, v):
    return np.array([v] * n, dtype=object)


def confirmed(F, pattern, param, i=I):
    return persistence_confirmed(F, arr(len(F), pattern), arr(len(F), param), RULES)[i]


def test_transient_change_not_classified_as_drift():
    for width in (1, 2):  # one-point and two-point jumps
        def mod(fr, w=width):
            fr["A"].iloc[I - w + 1: I + 1, 0] += 7
        F = network(mod)
        assert not confirmed(F, "DRIFT", "temperature_c"), width
        assert not confirmed(F, "STEP_OFFSET", "temperature_c"), width


def test_persistent_drift_detected():
    def mod(fr):
        fr["A"].iloc[I - 30: I + 1, 1] += np.linspace(0.2, 8.0, 31)
    assert confirmed(network(mod), "DRIFT", "pressure_hpa")


def test_persistent_step_detected():
    def mod(fr):
        fr["A"].iloc[I - 8: I + 1, 0] += 6
    assert confirmed(network(mod), "STEP_OFFSET", "temperature_c")


def test_step_followed_by_recovery_not_confirmed():
    def mod(fr):
        fr["A"].iloc[I - 10: I - 4, 0] += 6  # 6-hour offset, recovered 4 hours ago
    assert not confirmed(network(mod), "STEP_OFFSET", "temperature_c")


def test_step_during_regional_weather_is_not_a_station_step():
    def mod(fr):
        for s in fr:
            fr[s].iloc[I - 8: I + 1, 0] -= 6
    F = network(mod)
    assert not confirmed(F, "STEP_OFFSET", "temperature_c")
    assert regional_evidence(F).iloc[I]["regional"]


def test_clean_station_no_false_drift():
    """Regression for the SIH scenario-A failure: humidity swinging from below to above its baseline
    within a few hours (weather) is not a sustained one-signed drift."""
    def mod(fr):
        rh = fr["A"]["relative_humidity_pct"]
        fr["A"].iloc[I - 10: I - 2, 2] = (rh.iloc[I - 10: I - 2] - 8).clip(5, 100)
        fr["A"].iloc[I - 2: I + 1, 2] = (rh.iloc[I - 2: I + 1] + 8).clip(5, 100)
    assert not confirmed(network(mod), "DRIFT", "relative_humidity_pct")


def test_regional_weather_not_sensor_fault():
    def mod(fr):
        for s in fr:
            cool(fr[s], I)
    F = network(mod)
    ev = regional_evidence(F).iloc[I]
    assert ev["regional"] and not ev["station_exception"] and ev["stations_supporting_event"] >= 2
    pat, _ = apply_rules(F, arr(len(F), "SPIKE"), arr(len(F), "temperature_c"), RULES)
    assert pat[I] == "NORMAL"
    assert annotate(F, np.zeros(len(F), bool)).iloc[I]["weather_status"] == "REGIONAL_WEATHER"


def test_regional_weather_with_isolated_station_exception():
    def mod(fr):
        for s in fr:
            cool(fr[s], I)
        fr["A"].iloc[I, 1] += 12  # pressure jump only at A
    F = network(mod)
    ev = regional_evidence(F).iloc[I]
    assert ev["regional"] and ev["station_exception"] and "pressure" in ev["exception_reason"]
    pat, _ = apply_rules(F, arr(len(F), "SPIKE"), arr(len(F), "pressure_hpa"), RULES)
    assert pat[I] == "SPIKE"  # the regional event does not suppress the isolated fault
    flag = np.zeros(len(F), bool)
    flag[I] = True
    assert annotate(F, flag).iloc[I]["weather_status"] == "REGIONAL_WEATHER_WITH_STATION_EXCEPTION"


def test_neighbor_missing_does_not_crash():
    fr = {s: station_frame(12, seed=i) for i, s in enumerate("AB")}
    fr["B"][:] = np.nan  # neighbour registered but silent
    s1 = {k: station_features(v) for k, v in fr.items()}
    for nbrs in ([], [(s1["B"], 20.0)]):
        F = pd.concat([s1["A"], spatial_features(s1["A"], nbrs)], axis=1)
        ev = annotate(F, np.zeros(len(F), bool))
        assert (F["sp_n"] == 0).all() and (ev["stations_considered"] == 0).all() and not ev["regional_evidence"].any()


def test_bad_neighbor_does_not_dominate():
    def mod(fr):
        fr["B"].iloc[I - 24: I + 1, 0] += 10  # neighbour B is drifting / faulty
    F = network(mod)
    assert F.iloc[I]["t_sp_outliers"] == 1
    assert abs(F.iloc[I]["t_sp_resid"]) < 2.0


def test_alert_explanation_contains_regional_evidence():
    def mod(fr):
        for s in fr:
            cool(fr[s], I)
        fr["A"].iloc[I, 1] += 12
    F = network(mod)
    flag = np.zeros(len(F), bool)
    flag[I] = True
    R = pd.concat([F, annotate(F, flag)], axis=1)
    row = R.iloc[I].copy()
    row["flag"], row["prob"], row["pattern"], row["param"] = True, 0.9, "SPIKE", "pressure_hpa"
    e = explain(row, None, 0.3, "hybrid")
    w = e["weather_vs_sensor"]
    assert w["status"] == "REGIONAL_WEATHER_WITH_STATION_EXCEPTION"
    assert w["regional_evidence"]["stations_considered"] == 3 and w["station_exception"]["present"]
    assert any("Regional weather present" in x for x in e["evidence"])

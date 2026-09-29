import numpy as np
import pandas as pd

from ml.features import build_features, dewpoint, neighbors, rh_from, spatial_features, station_features
from tests.conftest import canonical, station_frame


def test_features_use_only_past_information(frame):
    base = station_features(frame)
    fut = frame.copy()
    fut.iloc[-24:] += 50  # change only the future
    alt = station_features(fut)
    cols = [c for c in base.columns if c.startswith(("t_", "p_", "rh_", "mv_"))]
    pd.testing.assert_frame_equal(base[cols].iloc[:-24], alt[cols].iloc[:-24])


def test_spike_robust_z_rate_and_mv_residual(frame):
    f = frame.copy()
    i = 24 * 8 + 5
    f.iloc[i, 0] += 15
    s = station_features(f).iloc[i]
    assert s["t_tod_z"] > 5 and s["t_rate_z"] > 5 and s["t_rate"] > 10
    assert s["mv_rh_resid"] > 10  # RH did not respond to the temperature jump


def test_drift_slope_and_time_since_change(frame):
    f = frame.copy()
    f.iloc[-72:, 1] += np.linspace(0, 8, 72)
    f.iloc[-10:, 0] = f.iloc[-11, 0]
    s = station_features(f).iloc[-1]
    assert s["p_dev_mean24h"] > 2 and s["p_slope24h"] > 0.05
    assert s["t_run_h"] >= 10


def test_dewpoint_round_trip():
    t, rh = pd.Series([10.0, 25.0, 35.0]), pd.Series([90.0, 60.0, 30.0])
    np.testing.assert_allclose(rh_from(t, dewpoint(t, rh)), rh, rtol=1e-6)


def test_neighbor_selection_is_configurable_and_excludes_far():
    st = {"A": {"latitude": 12.0, "longitude": 77.0}, "B": {"latitude": 12.5, "longitude": 77.2},
          "C": {"latitude": 28.0, "longitude": 77.0}}
    nb = neighbors(st)
    assert [n for n, _ in nb["A"]] == ["B"] and nb["C"] == []


def test_spatial_regional_vs_isolated_and_missing_neighbors():
    fr = {s: station_frame(10, seed=i) for i, s in enumerate("ABC")}
    i = 24 * 8 + 3
    iso = {k: v.copy() for k, v in fr.items()}
    iso["A"].iloc[i, 0] += 8
    reg = {k: v.copy() for k, v in fr.items()}
    for k in reg:
        reg[k].iloc[i, 0] += 8
    s1 = lambda d: {k: station_features(v) for k, v in d.items()}  # noqa: E731
    a_iso, a_reg = s1(iso), s1(reg)
    sp_iso = spatial_features(a_iso["A"], [(a_iso["B"], 20), (a_iso["C"], 40)]).iloc[i]
    sp_reg = spatial_features(a_reg["A"], [(a_reg["B"], 20), (a_reg["C"], 40)]).iloc[i]
    assert sp_iso["t_sp_resid"] > 5 and sp_iso["t_sp_frac"] == 0
    assert abs(sp_reg["t_sp_resid"]) < 3 and sp_reg["t_sp_frac"] == 1
    none = spatial_features(a_iso["A"], [])
    assert (none["sp_n"] == 0).all() and none["t_sp_z"].isna().all()


def test_build_features_keeps_row_order():
    df = canonical({"A": station_frame(9), "B": station_frame(9, 1)}, {"A": (12, 77), "B": (12.3, 77.1)})
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    F = build_features(df, {"A": {"latitude": 12, "longitude": 77}, "B": {"latitude": 12.3, "longitude": 77.1}})
    assert (F["station_id"].to_numpy() == df["station_id"].to_numpy()).all()
    assert (F["timestamp"].to_numpy() == df["timestamp"].to_numpy()).all()
    assert F["sp_n"].max() == 1

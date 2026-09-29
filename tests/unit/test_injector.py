import numpy as np
import pandas as pd

from injection.injector import KINDS, PROFILES, inject
from ml.schema import VARS
from tests.conftest import canonical, station_frame

COORDS = {"A": (12.0, 77.0), "B": (12.3, 77.2), "C": (12.6, 77.1), "D": (28.0, 77.0)}


def data():
    return canonical({s: station_frame(90, seed=i, start="2024-01-01") for i, s in enumerate(COORDS)}, COORDS)


def test_deterministic_and_seed_sensitive():
    df = data()
    a, ea = inject(df, 42, "train")
    b, eb = inject(df, 42, "train")
    c, _ = inject(df, 43, "train")
    pd.testing.assert_frame_equal(a, b)
    assert not a[VARS].equals(c[VARS])
    assert ea.equals(eb)


def test_all_mandatory_classes_and_ground_truth_preserved():
    out, ev = inject(data(), 7, "test")
    kinds = set(ev["kind"])
    assert set(KINDS) - {"pressure_spike"} <= kinds | {"pressure_spike"}
    assert {"SPIKE", "NEGATIVE_SPIKE", "FLATLINE", "DRIFT", "STEP_OFFSET", "MISSING_COMMUNICATION", "SENTINEL_CORRUPTION",
            "MULTIVARIATE_INCONSISTENCY", "SPATIAL_INCONSISTENCY", "COMPOSITE_FAULT"} <= set(out["anomaly_type"])
    inj = out[out["is_anomaly"] == 1]
    assert inj["injection_id"].str.len().gt(0).all()
    single = inj[inj["injected_parameter"].isin(VARS) & inj["ground_truth_value"].notna()]
    for v in VARS:
        s = single[single["injected_parameter"] == v]
        np.testing.assert_allclose(s["ground_truth_value"], s[f"clean_{v}"])
    normal = out[(out["is_anomaly"] == 0) & (out["injection_kind"] == "")]
    for v in VARS:
        np.testing.assert_allclose(normal[v].fillna(-1), normal[f"clean_{v}"].fillna(-1))


def test_regional_events_are_labelled_normal_and_spatial_needs_neighbors():
    out, ev = inject(data(), 3, "train")
    reg = out[out["injection_kind"] == "regional_event"]
    assert len(reg) and (reg["is_anomaly"] == 0).all()
    assert "D" not in set(ev.loc[ev["kind"] == "spatial_inconsistency", "station_id"])


def test_profiles_differ_between_train_and_test():
    assert PROFILES["train"] != PROFILES["test"] and PROFILES["val"] != PROFILES["test"]
    assert PROFILES["test"]["spike_len"][1] > PROFILES["train"]["spike_len"][1]


def test_injected_values_quantised_to_source_resolution():
    out, _ = inject(data(), 5, "train")
    inj = out[out["injection_kind"].isin(["positive_spike", "drift", "step"]) & (out["injected_parameter"] == "temperature_c")]
    assert (inj["temperature_c"] % 1 == 0).all()

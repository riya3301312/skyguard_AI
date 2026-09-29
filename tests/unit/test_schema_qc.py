from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml.qc import ingest_check, qc_frame
from ml.schema import validate_canonical
from tests.conftest import canonical, station_frame


def test_canonical_schema_ok_and_errors():
    df = canonical({"A": station_frame(2)}, {"A": (12.0, 77.0)})
    validate_canonical(df)
    with pytest.raises(ValueError, match="missing canonical"):
        validate_canonical(df.drop(columns=["pressure_hpa"]))
    bad = df.copy()
    bad["timestamp"] = bad["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        validate_canonical(bad)
    with pytest.raises(ValueError, match="duplicate"):
        validate_canonical(pd.concat([df, df.iloc[:1]]))


def test_missing_sentinel_range_and_raw_preserved(frame):
    f = frame.copy()
    t = f.index
    f.loc[t[5], "temperature_c"] = np.nan
    f.loc[t[6], "pressure_hpa"] = -9999.0
    f.loc[t[7], "relative_humidity_pct"] = 130.0
    before = f.copy()
    q = qc_frame(f)
    assert q.loc[t[5], "qc_t_missing"] and not q.loc[t[5], "qc_hard"]
    assert q.loc[t[6], "qc_p_sentinel"] and q.loc[t[6], "qc_hard_class"] == "SENTINEL_CORRUPTION"
    assert q.loc[t[7], "qc_rh_range"] and np.isnan(q.loc[t[7], "x_rh"])
    pd.testing.assert_frame_equal(f, before)  # raw input untouched


def test_all_missing_is_communication_class(frame):
    f = frame.copy()
    f.iloc[10] = np.nan
    q = qc_frame(f)
    assert q["qc_all_missing"].iloc[10] and q["qc_hard_class"].iloc[10] == "MISSING_COMMUNICATION"


def test_step_flatline_and_gap(frame):
    f = frame.copy()
    f.iloc[20, 0] = f.iloc[19, 0] + 20  # 20 degC jump in one hour
    f.iloc[40:52] = f.iloc[39].to_numpy()  # all three frozen for 12 h
    q = qc_frame(f)
    assert q["qc_t_step"].iloc[20]
    assert q["qc_flat_all"].iloc[51] and not q["qc_flat_all"].iloc[39]
    g = f.drop(f.index[60:65])  # 5 absent hours
    qg = qc_frame(g)
    assert qg["qc_gap"].iloc[60] and qg["qc_gap_h"].iloc[60] == pytest.approx(6.0)


def test_timestamps_future_and_delayed_are_not_anomalies():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert ingest_check(now + timedelta(hours=1), now)["future"]
    c = ingest_check(now - timedelta(hours=3), now)
    assert c["delayed"] and not c["future"] and c["lag_minutes"] == pytest.approx(180)

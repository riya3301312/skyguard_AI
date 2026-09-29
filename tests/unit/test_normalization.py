import gzip
import hashlib

import numpy as np
import pytest

from ml.config import load_config
from ml.schema import validate_canonical
from scripts.build_dataset import INHG_TO_HPA, canonicalize, load_raw

RAW = """station,valid,lon,lat,elevation,tmpc,relh,alti,mslp
VXXX,2023-01-01 00:00,77.0,12.0,900.0,20.00,80.00,29.92,M
VXXX,2023-01-01 00:30,77.0,12.0,900.0,21.00,78.00,29.92,M
VXXX,2023-01-01 01:00,77.0,12.0,900.0,M,78.00,29.93,M
VXXX,2023-01-01 01:00,77.0,12.0,900.0,21.00,77.00,29.93,M
VXXX,2023-01-01 02:00,77.0,12.0,900.0,-9999,77.00,29.93,M
VXXX,2023-01-01 03:00,77.0,12.0,900.0,23.00,70.00,29.94,M
VXXX,2023-01-01 09:00,77.0,12.0,900.0,30.00,50.00,29.90,M
"""


@pytest.fixture
def raw_file(tmp_path):
    p = tmp_path / "VXXX.csv.gz"
    p.write_bytes(gzip.compress(RAW.encode()))
    return p


def test_canonicalize_units_grid_dedupe_and_flags(raw_file):
    cfg = dict(load_config())
    cfg["data"] = {**cfg["data"], "start": "2023-01-01", "end": "2023-01-01T12:00"}
    before = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    d, meta = canonicalize(load_raw(raw_file), cfg)
    assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == before  # raw file never modified
    validate_canonical(d)
    assert len(d) == 12 and d["timestamp"].dt.minute.eq(0).all()  # hourly grid, :30 report dropped
    r0 = d.iloc[0]
    assert r0["pressure_hpa"] == pytest.approx(round(29.92 * INHG_TO_HPA, 1))
    assert d.iloc[1]["temperature_c"] == 21.0  # duplicate timestamp: last report kept
    assert d.iloc[2]["quality_flag"] == "interpolated" and d.iloc[2]["temperature_c"] == pytest.approx(22.0)  # -9999 removed, 1-slot gap
    assert meta["qc_removed_values"] == 1
    assert d.iloc[5]["quality_flag"] == "missing" and np.isnan(d.iloc[5]["temperature_c"])  # 5-slot gap stays missing
    assert meta["source"].startswith("IEM-METAR/IMD")

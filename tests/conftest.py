import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.schema import VARS  # noqa: E402


def station_frame(days: int = 10, seed: int = 0, start: str = "2024-03-01") -> pd.DataFrame:
    """Synthetic hourly station with a diurnal cycle (for unit tests only; never used for metrics)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    h = idx.hour.to_numpy()
    t = 28 + 5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.4, len(idx))
    p = 1008 + 1.2 * np.sin(4 * np.pi * h / 24) + rng.normal(0, 0.2, len(idx))
    rh = np.clip(70 - 15 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 2, len(idx)), 5, 100)
    return pd.DataFrame({"temperature_c": np.round(t), "pressure_hpa": np.round(p, 1), "relative_humidity_pct": np.round(rh, 1)}, index=idx)


def canonical(frames: dict[str, pd.DataFrame], coords: dict[str, tuple[float, float]]) -> pd.DataFrame:
    parts = []
    for sid, f in frames.items():
        d = f.reset_index(names="timestamp")
        d["station_id"] = sid
        d["latitude"], d["longitude"] = coords[sid]
        d["altitude_m"], d["source"], d["quality_flag"] = 10.0, "test", "ok"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def frame():
    return station_frame()


@pytest.fixture
def vars_():
    return VARS

"""Hermetic API fixtures: a small model trained on synthetic data, temp DB and station registry."""
import json

import numpy as np
import pytest

from injection.injector import inject
from ml.config import load_config
from ml.models import AE_WIN, IForestDetector, SkyGuardHybrid, TemporalAutoencoder
from ml.pipeline import ModelBundle, add_ae, features_for
from ml.config import ROOT
from tests.conftest import canonical, station_frame

COORDS = {"AWS_A": (12.00, 77.00), "AWS_B": (12.25, 77.20), "AWS_C": (12.45, 76.90), "AWS_REMOTE": (11.6, 92.7)}
META = {s: {"latitude": la, "longitude": lo, "altitude_m": 10.0, "source": "test"} for s, (la, lo) in COORDS.items()}


def history(sid: str, days: int = 9, start: str = "2024-05-01"):
    f = station_frame(days, seed=list(COORDS).index(sid), start=start)
    return [{"station_id": sid, "timestamp": t.isoformat(), **{k: float(v) for k, v in row.items()}} for t, row in f.iterrows()]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    real = ROOT / "artifacts" / "final_model"
    if (real / "model.joblib").exists():
        return real  # the trained SkyGuard model (features are station-relative, so synthetic stations are valid input)
    df = canonical({s: station_frame(60, seed=i, start="2024-01-01") for i, s in enumerate(COORDS)}, COORDS)
    tr, _ = inject(df, 1, "train")
    va, _ = inject(df, 2, "val")
    Ftr, Fva = features_for(tr, META, None), features_for(va, META, None)
    tr = tr.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    va = va.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    frames = [g for _, g in Ftr.groupby("station_id", sort=False)]
    masks = [tr.loc[g.index, "is_anomaly"].rolling(AE_WIN, min_periods=1).max().to_numpy() == 0 for g in frames]
    ae = TemporalAutoencoder(0).fit(frames, masks, max_windows=3000)
    Ftr, Fva = add_ae(Ftr, ae), add_ae(Fva, ae)
    h = SkyGuardHybrid("full", seed=0).fit(Ftr, tr["is_anomaly"].to_numpy(), tr["anomaly_type"].to_numpy(),
                                           tr["injected_parameter"].to_numpy())
    h.calibrate(Fva, va["is_anomaly"].to_numpy(), load_config())
    b = ModelBundle(hybrid=h, ae=ae, iforest=IForestDetector(0).fit(Ftr), thresholds={},
                    metadata={"feature_version": "test", "selected_variant": "full", "trained_at": "test"})
    d = tmp_path_factory.mktemp("model")
    b.save(d)
    return d


@pytest.fixture
def make_client(tmp_path, monkeypatch, model_dir):
    from fastapi.testclient import TestClient
    st = tmp_path / "stations.json"
    st.write_text(json.dumps({"stations": META}))

    def _make(model=True, token=None):
        monkeypatch.setenv("SKYGUARD_DB", str(tmp_path / f"db_{np.random.randint(1e9)}.sqlite"))
        monkeypatch.setenv("SKYGUARD_STATIONS", str(st))
        monkeypatch.setenv("SKYGUARD_MODEL_DIR", str(model_dir if model else tmp_path / "no_model"))
        if token:
            monkeypatch.setenv("SKYGUARD_API_TOKEN", token)
        else:
            monkeypatch.delenv("SKYGUARD_API_TOKEN", raising=False)
        from backend.app import app
        return TestClient(app)
    return _make

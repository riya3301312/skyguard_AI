"""End-to-end batch scoring: features -> autoencoder -> hybrid detector -> post-processing."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .decision import annotate
from .features import build_features
from .models import AE_COLS, IForestDetector, SkyGuardHybrid, TemporalAutoencoder
from .postprocess import corrections, health, severity
from .schema import VARS


@dataclass
class ModelBundle:
    hybrid: SkyGuardHybrid
    ae: TemporalAutoencoder
    iforest: IForestDetector | None = None
    thresholds: dict = field(default_factory=dict)  # baseline thresholds chosen on validation
    correction_method: str = "auto"
    metadata: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.joblib", compress=3)
        (path / "model_metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str))

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        return joblib.load(Path(path) / "model.joblib")


def add_ae(F: pd.DataFrame, ae: TemporalAutoencoder) -> pd.DataFrame:
    """F must be sorted by (station_id, timestamp)."""
    parts = [ae.score(g) for _, g in F.groupby("station_id", sort=False)]
    return F.join(pd.concat(parts)[AE_COLS])


def features_for(df: pd.DataFrame, stations: dict, ae: TemporalAutoencoder | None, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    F = build_features(df, stations, cfg)
    for v in VARS:
        F[v] = df[v].to_numpy()
    return add_ae(F, ae) if ae is not None else F


def postprocess(R: pd.DataFrame, bundle: ModelBundle, cfg: dict | None = None) -> pd.DataFrame:
    """Per-station severity, health and correction. R sorted by (station_id, timestamp)."""
    cfg = cfg or load_config()
    parts = []
    for _, g in R.groupby("station_id", sort=False):
        g = g.set_index("timestamp")
        g["severity_score"], g["severity"] = severity(g, cfg)
        g = g.join(health(g, cfg))
        g = g.join(corrections(g, bundle.correction_method))
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def score_features(F: pd.DataFrame, bundle: ModelBundle, cfg: dict | None = None) -> pd.DataFrame:
    h = bundle.hybrid
    R = F.copy()
    R["prob"], R["regional_evidence"] = h.predict_proba(F)
    R["pattern"], R["param"], R["pattern_conf"] = h.classify(F, R["prob"].to_numpy())
    R["flag"] = R["pattern"] != "NORMAL"
    ev = annotate(R, R["flag"].to_numpy(), (getattr(h, "rules", None) or {}).get("regional_min", 0.5))
    R = pd.concat([R.drop(columns=[c for c in ev.columns if c in R.columns]), ev], axis=1)
    return postprocess(R, bundle, cfg)


def score_frame(df: pd.DataFrame, stations: dict, bundle: ModelBundle, cfg: dict | None = None) -> pd.DataFrame:
    return score_features(features_for(df, stations, bundle.ae, cfg), bundle, cfg)


def load_stations(path: Path | None = None) -> dict:
    from .config import ROOT
    p = path or ROOT / "data" / "processed" / "stations.json"
    return json.loads(Path(p).read_text())["stations"]


def event_level(R: pd.DataFrame, flag_col: str = "flag") -> pd.DataFrame:
    """Per injected event: detected if any of its rows is flagged."""
    ev = R[(R["injection_id"] != "") & (R["injection_id"] != "REAL_GAP")]
    return ev.groupby("injection_id").agg(anomaly_type=("anomaly_type", "first"), kind=("injection_kind", "first"),
                                          detected=(flag_col, "any"), n=("is_anomaly", "size"),
                                          station_id=("station_id", "first"))


def np_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

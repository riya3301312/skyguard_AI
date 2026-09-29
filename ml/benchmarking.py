"""Score every approach (baselines, ablation variants, final hybrid) on one feature frame."""
import numpy as np
import pandas as pd

from .evaluation import detection, eval_mask, event_recall
from .models import SkyGuardHybrid, ae_score, fixed_threshold_score, robust_stat_score
from .pipeline import ModelBundle

APPROACH_NAMES = {"fixed_threshold": "Fixed Threshold", "robust_statistical": "Robust Statistical",
                  "isolation_forest": "Isolation Forest", "temporal_autoencoder": "Temporal Autoencoder (MLP)",
                  "skyguard_hybrid": "SkyGuard Hybrid"}


def hybrid_scores(h: SkyGuardHybrid, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p, _ = h.predict_proba(F)
    return p, p >= h.threshold


def baseline_scores(bundle: ModelBundle, F: pd.DataFrame) -> dict:
    th = bundle.thresholds
    out = {"fixed_threshold": (s := fixed_threshold_score(F), s >= 0.5)}
    s = robust_stat_score(F)
    out["robust_statistical"] = (s, s >= th["robust_statistical"])
    s = bundle.iforest.score(F)
    out["isolation_forest"] = (s, s >= th["isolation_forest"])
    s = ae_score(F)
    out["temporal_autoencoder"] = (s, s >= th["temporal_autoencoder"])
    return out


def compare(F: pd.DataFrame, labels: pd.DataFrame, approaches: dict) -> dict:
    """approaches: name -> (score, pred). Metrics on eval rows only."""
    m = eval_mask(labels)
    L = labels[m]
    res = {}
    for name, (score, pred) in approaches.items():
        res[name] = detection(L, np.asarray(pred)[m], np.asarray(score, float)[m]) | event_recall(L, np.asarray(pred)[m])
    return res

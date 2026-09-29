"""Metric computation shared by validation and the frozen final evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from .schema import FAULT_CLASSES, VARS


def eval_mask(R: pd.DataFrame) -> np.ndarray:
    """Rows that count for metrics: not warm-up history, not real (non-injected) source gaps."""
    m = R["injection_id"].fillna("") != "REAL_GAP"
    if "split_role" in R:
        m &= R["split_role"] == "eval"
    return m.to_numpy()


def detection(R: pd.DataFrame, pred: np.ndarray, score: np.ndarray, rows_per_day: int = 48) -> dict:
    y = R["is_anomaly"].to_numpy().astype(int)
    s = np.nan_to_num(np.where(np.isinf(score), 1e9, score), nan=0.0)
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    station_days = R["timestamp"].dt.floor("D").astype(str).str.cat(R["station_id"]).nunique()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "n": int(len(y)), "positives": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else None,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_alerts_per_1000_obs": 1000 * fp / max(len(y), 1),
        "false_alerts_per_station_day": fp / max(station_days, 1),
        "station_days": int(station_days),
    }


def event_recall(R: pd.DataFrame, pred: np.ndarray) -> dict:
    ev = pd.DataFrame({"iid": R["injection_id"].to_numpy(), "cls": R["anomaly_type"].to_numpy(),
                       "kind": R["injection_kind"].to_numpy(), "pred": pred, "y": R["is_anomaly"].to_numpy()})
    ev = ev[(ev["iid"] != "") & (ev["iid"] != "REAL_GAP")]
    g = ev.groupby("iid").agg(cls=("cls", "first"), kind=("kind", "first"), hit=("pred", "any"), y=("y", "first"))
    anomalous = g[g["y"] == 1]
    regional = g[g["kind"] == "regional_event"]
    return {
        "events": int(len(anomalous)),
        "event_recall": float(anomalous["hit"].mean()) if len(anomalous) else None,
        "event_recall_by_class": anomalous.groupby("cls")["hit"].mean().round(4).to_dict(),
        "events_by_class": anomalous.groupby("cls").size().to_dict(),
        "regional_events": int(len(regional)),
        "regional_events_falsely_alerted": float(regional["hit"].mean()) if len(regional) else None,
    }


def classification(true_cls: np.ndarray, pred_cls: np.ndarray) -> dict:
    labels = [c for c in FAULT_CLASSES if c in set(true_cls) | set(pred_cls)]
    present = [c for c in FAULT_CLASSES if c in set(true_cls)]
    p, r, f, s = precision_recall_fscore_support(true_cls, pred_cls, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(true_cls, pred_cls, labels=present, average="macro", zero_division=0)),
        "macro_f1_anomaly_classes": float(f1_score(true_cls, pred_cls, labels=[c for c in present if c != "NORMAL"],
                                                   average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(labels)},
        "labels": labels,
        "confusion_matrix": confusion_matrix(true_cls, pred_cls, labels=labels).tolist(),
    }


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rel = []
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rel.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "mean_predicted": float(p[m].mean()),
                        "observed_rate": float(y[m].mean()), "count": int(m.sum())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier_score": float(brier_score_loss(y, p)), "expected_calibration_error": float(ece), "reliability": rel}


def imputation(R: pd.DataFrame) -> dict:
    """Error of recommended estimates vs injected ground truth, on flagged injected anomaly rows."""
    out = {}
    allerr = []
    for v in VARS:
        m = (R["is_anomaly"] == 1) & R["injected_parameter"].str.contains(v, regex=False) & R[f"estimated_{v}"].notna() \
            & R[f"clean_{v}"].notna() & (R["injection_id"] != "REAL_GAP")
        e = (R.loc[m, f"estimated_{v}"] - R.loc[m, f"clean_{v}"]).to_numpy(float)
        # compare with the raw (corrupted) value: what you would have used without SkyGuard
        raw = pd.to_numeric(R.loc[m, v], errors="coerce").to_numpy(float) - R.loc[m, f"clean_{v}"].to_numpy(float)
        raw = raw[np.isfinite(raw) & (np.abs(raw) < 1000)]
        if len(e):
            out[v] = {"n": int(len(e)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                      "median_abs_error": float(np.median(np.abs(e))),
                      "raw_value_mae_for_reference": float(np.mean(np.abs(raw))) if len(raw) else None}
            allerr.append(e)
    return out


def latency_stats(ms: np.ndarray) -> dict:
    ms = np.asarray(ms, float)
    return {"n": int(len(ms)), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean()), "max_ms": float(ms.max())}

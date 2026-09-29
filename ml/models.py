"""Detectors: baselines + SkyGuard hybrid. All thresholds are selected on validation only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPRegressor

from .features import GROUPS, MV_COLS, SPATIAL_COLS, TEMPORAL_COLS
from .schema import SHORT, VARS

AE_WIN = 12
AE_COLS = ["ae_t", "ae_p", "ae_rh", "ae_total"]
HARD_CLASSES = {"MISSING_COMMUNICATION", "SENTINEL_CORRUPTION"}
PATTERN_CLASSES = ["SPIKE", "NEGATIVE_SPIKE", "FLATLINE", "DRIFT", "STEP_OFFSET", "MULTIVARIATE_INCONSISTENCY",
                   "SPATIAL_INCONSISTENCY", "COMPOSITE_FAULT"]
FEATURE_SETS = {
    "temporal": TEMPORAL_COLS,
    "temporal+multivariate": TEMPORAL_COLS + MV_COLS,
    "temporal+spatial": TEMPORAL_COLS + SPATIAL_COLS,
    "temporal+multivariate+spatial": TEMPORAL_COLS + MV_COLS + SPATIAL_COLS,
    "full": TEMPORAL_COLS + MV_COLS + SPATIAL_COLS + AE_COLS,
}


def best_threshold(y: np.ndarray, score: np.ndarray, grid: np.ndarray | None = None,
                   station_days: int | None = None, max_fa_per_station_day: float | None = None) -> float:
    """Operating point on VALIDATION: maximise F1 subject to a false-alert budget (false alerts per
    station-day); if no threshold meets the budget, take the one with the fewest false alerts."""
    s = score[np.isfinite(score)]
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 400))) if grid is None else grid
    best, fallback = (-1.0, None), (np.inf, None)
    for t in grid:
        pred = score >= t
        fa = (pred & (y == 0)).sum() / max(station_days or 1, 1)
        f = f1_score(y, pred, zero_division=0)
        if max_fa_per_station_day is None or station_days is None or fa <= max_fa_per_station_day:
            if f > best[0]:
                best = (f, t)
        elif fa < fallback[0]:
            fallback = (fa, t)
    return float(best[1] if best[1] is not None else fallback[1])


def station_days(F: pd.DataFrame) -> int:
    return int(F["timestamp"].dt.floor("D").astype(str).str.cat(F["station_id"].astype(str)).nunique())


# ---------------------------------------------------------------- temporal autoencoder
class TemporalAutoencoder:
    """Dense autoencoder over 6-hour windows (12 x [T,P,RH] time-of-day deviations), sklearn MLP."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.scale = None
        self.net = None

    def _z(self, f: pd.DataFrame) -> np.ndarray:
        z = np.column_stack([f[f"{SHORT[v]}_tod_dev"].to_numpy(float) / self.scale[i] for i, v in enumerate(VARS)])
        return np.clip(np.nan_to_num(z, nan=0.0), -10, 10)

    def windows(self, f: pd.DataFrame) -> np.ndarray:
        """Rows of f must be one station in time order. Window i = rows i-11..i (zero-padded)."""
        z = self._z(f)
        n = len(z)
        pad = np.vstack([np.zeros((AE_WIN - 1, 3)), z])
        return np.stack([pad[i: i + n] for i in range(AE_WIN)], axis=1).reshape(n, AE_WIN * 3)

    def fit(self, frames: list[pd.DataFrame], normal_masks: list[np.ndarray], max_windows: int = 60000):
        allf = pd.concat(frames)
        self.scale = [float(np.nanmedian(np.abs(allf[f"{SHORT[v]}_tod_dev"])) * 1.4826 + 1e-3) for v in VARS]
        X = np.vstack([self.windows(f)[m] for f, m in zip(frames, normal_masks)])
        rng = np.random.default_rng(self.seed)
        X = X[rng.choice(len(X), size=min(max_windows, len(X)), replace=False)]
        self.net = MLPRegressor(hidden_layer_sizes=(24, 8, 24), max_iter=80, random_state=self.seed,
                                early_stopping=True, tol=1e-4).fit(X, X)
        return self

    def score(self, f: pd.DataFrame) -> pd.DataFrame:
        W = self.windows(f)
        err = (self.net.predict(W) - W) ** 2
        last = err.reshape(len(W), AE_WIN, 3)[:, -1, :]
        return pd.DataFrame({"ae_t": last[:, 0], "ae_p": last[:, 1], "ae_rh": last[:, 2],
                             "ae_total": err.mean(axis=1)}, index=f.index)


# ---------------------------------------------------------------- baselines
def fixed_threshold_score(F: pd.DataFrame) -> np.ndarray:
    """Fixed rules only: QC hard flags, fixed max-step and all-variable flatline."""
    step = F[[f"qc_{SHORT[v]}_step" for v in VARS]].any(axis=1)
    return (F["qc_hard"] | step | F["qc_flat_all"]).astype(float).to_numpy()


def robust_stat_score(F: pd.DataFrame) -> np.ndarray:
    cols = [f"{SHORT[v]}_{k}" for v in VARS for k in ("tod_z", "rz24", "rate_z")]
    s = F[cols].abs().max(axis=1).fillna(0).to_numpy()
    return np.where(F["qc_hard"].to_numpy(), np.inf, s)


class IForestDetector:
    cols = TEMPORAL_COLS + MV_COLS

    def __init__(self, seed: int = 7):
        self.m = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1)

    def fit(self, F: pd.DataFrame):
        self.m.fit(F[self.cols].fillna(0).to_numpy())
        return self

    def score(self, F: pd.DataFrame) -> np.ndarray:
        s = -self.m.score_samples(F[self.cols].fillna(0).to_numpy())
        return np.where(F["qc_hard"].to_numpy(), np.inf, s)


def ae_score(F: pd.DataFrame) -> np.ndarray:
    return np.where(F["qc_hard"].to_numpy(), np.inf, F["ae_total"].fillna(0).to_numpy())


# ---------------------------------------------------------------- SkyGuard hybrid
def _hgb(seed: int, **kw) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed, **kw)


class SkyGuardHybrid:
    """Deterministic QC + calibrated gradient-boosted detector + pattern/parameter classifiers
    + regional-event reasoning. `feature_set` supports the ablation study."""

    def __init__(self, feature_set: str = "full", seed: int = 7, regional_rule: bool = True):
        self.feature_set = feature_set
        self.cols = FEATURE_SETS[feature_set]
        self.seed = seed
        self.regional_rule = regional_rule
        self.detector = self.calibrator = self.pattern = self.param = None
        self.threshold = 0.5
        self.reference = None
        self.regional_fraction = 0.6
        self.regional_suppression = 1.0

    # raw ML probability (uncalibrated)
    def _raw(self, F: pd.DataFrame) -> np.ndarray:
        return self.detector.predict_proba(F[self.cols])[:, 1]

    def regional_mask(self, F: pd.DataFrame) -> np.ndarray:
        """Evidence that a deviation is shared by neighbours (plausible weather event)."""
        if "sp_n" not in F:
            return np.zeros(len(F), dtype=bool)
        has = F["sp_n"].fillna(0).to_numpy() >= 2
        shared = np.zeros(len(F), dtype=bool)
        isolated = np.zeros(len(F), dtype=bool)
        for v in VARS:
            s = SHORT[v]
            frac = F[f"{s}_sp_frac"].fillna(0).to_numpy()
            dev_big = F[f"{s}_tod_z"].abs().fillna(0).to_numpy() > 3
            shared |= dev_big & (frac >= self.regional_fraction)
            isolated |= dev_big & (frac < self.regional_fraction)
        # regional only if every strongly deviating variable is shared by the neighbours
        return has & shared & ~isolated

    def fit(self, F: pd.DataFrame, y: np.ndarray, cls: np.ndarray, param: np.ndarray):
        ml = ~F["qc_hard"].to_numpy()
        # a feature that is never observed (e.g. spatial features for a network without neighbours) is dropped
        self.cols = [c for c in self.cols if F.loc[ml, c].notna().any()]
        self.detector = _hgb(self.seed).fit(F.loc[ml, self.cols], y[ml])
        self.reference = F.loc[ml & (y == 0), self.cols].median(numeric_only=True)
        pm = ml & (y == 1) & np.isin(cls, PATTERN_CLASSES)
        self.pattern = _hgb(self.seed, class_weight="balanced").fit(F.loc[pm, self.cols], cls[pm])
        qm = pm & np.isin(param, VARS)
        self.param = _hgb(self.seed).fit(F.loc[qm, self.cols], param[qm])
        return self

    def calibrate(self, Fv: pd.DataFrame, yv: np.ndarray, cfg: dict):
        """Isotonic calibration + threshold + regional rule, all on validation data."""
        ml = ~Fv["qc_hard"].to_numpy()
        raw = self._raw(Fv[ml])
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, yv[ml])
        self.regional_fraction = cfg["model"]["regional_fraction"]
        base = self.calibrator.predict(raw)
        reg = self.regional_mask(Fv[ml])
        choices = [1.0] + ([cfg["model"]["regional_suppression"]] if self.regional_rule else [])
        best = (-1.0, 1.0, 0.5)
        sd, budget = station_days(Fv), cfg["model"]["max_false_alerts_per_station_day"]
        for mult in choices:
            p = np.where(reg, base * mult, base)
            t = best_threshold(yv[ml], p, np.linspace(0.05, 0.95, 91), sd, budget)
            f = f1_score(yv[ml], p >= t, zero_division=0)
            if f > best[0] + 1e-4:
                best = (f, mult, t)
        _, self.regional_suppression, self.threshold = best
        return self

    def predict_proba(self, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probability, regional_evidence). QC-hard rows get probability 1.0 (deterministic)."""
        hard = F["qc_hard"].to_numpy()
        p = np.ones(len(F))
        reg = self.regional_mask(F)
        if (~hard).any():
            q = self.calibrator.predict(self._raw(F[~hard]))
            p[~hard] = np.where(reg[~hard], q * self.regional_suppression, q)
        return p, reg

    def classify(self, F: pd.DataFrame, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (fault pattern, parameter, pattern confidence)."""
        n = len(F)
        pattern = np.full(n, "NORMAL", dtype=object)
        param = np.full(n, "", dtype=object)
        conf = np.zeros(n)
        hard = F["qc_hard"].to_numpy()
        flag = (prob >= self.threshold) & ~hard
        if flag.any():
            pp = self.pattern.predict_proba(F.loc[flag, self.cols])
            best = pp.argmax(axis=1)
            conf[flag] = pp.max(axis=1)
            pattern[flag] = np.where(conf[flag] >= 0.4, self.pattern.classes_[best], "UNKNOWN_SUSPECT")
            param[flag] = self.param.predict(F.loc[flag, self.cols])
        if hard.any():
            pattern[hard] = F.loc[hard, "qc_hard_class"].to_numpy()
            conf[hard] = 1.0
            H = F.loc[hard]
            bad = np.column_stack([(H[f"qc_{SHORT[v]}_sentinel"] | H[f"qc_{SHORT[v]}_range"]).to_numpy() for v in VARS])
            param[hard] = np.where(H["qc_all_missing"].to_numpy(), "all",
                                   np.array(VARS, dtype=object)[bad.argmax(axis=1)])
        return pattern, param, conf

    def contributions(self, F: pd.DataFrame) -> pd.DataFrame:
        """Group-occlusion attribution: drop in calibrated probability when a feature group is
        replaced by its typical normal value. Faithful to this model (it re-queries the model)."""
        groups = {"temporal_temperature": [c for c in TEMPORAL_COLS if c.startswith("t_")],
                  "temporal_pressure": [c for c in TEMPORAL_COLS if c.startswith("p_")],
                  "temporal_humidity": [c for c in TEMPORAL_COLS if c.startswith("rh_")],
                  "multivariate": MV_COLS, "spatial": SPATIAL_COLS, "autoencoder": AE_COLS}
        X = F[self.cols].copy()
        base = self.calibrator.predict(self.detector.predict_proba(X)[:, 1])
        out = {}
        for g, cols in groups.items():
            cols = [c for c in cols if c in self.cols]
            if not cols:
                continue
            Xo = X.copy()
            Xo[cols] = self.reference[cols].to_numpy()
            out[g] = base - self.calibrator.predict(self.detector.predict_proba(Xo)[:, 1])
        return pd.DataFrame(out, index=F.index)


def detection_groups() -> dict:
    return {k: list(v) for k, v in GROUPS.items()} | {"autoencoder": AE_COLS}

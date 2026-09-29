"""Real-time scoring engine (per-observation), sharing all feature/post-processing code with batch.

Temporal inference uses observation time only. Duplicates are idempotent, delayed and
out-of-order records are accepted and scored against the history that precedes their
observation time. If no model is available the engine runs an explicitly labelled
'qc_only_fallback' mode (no probability is reported).
"""
import math
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import load_config
from .features import neighbors, spatial_features, station_features
from .models import AE_COLS, fixed_threshold_score
from .postprocess import corrections, explain, health, health_reasons, maintenance_risk, severity
from .qc import ingest_check
from .schema import VARS

RESULT_KEEP_DAYS = 15


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class StreamEngine:
    def __init__(self, stations: dict, bundle=None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.bundle = bundle
        self.mode = "hybrid" if bundle is not None else "qc_only_fallback"
        self.stations = dict(stations)
        self.nb = neighbors(self.stations, self.cfg)
        self.step = pd.Timedelta(minutes=self.cfg["data"]["interval_minutes"])
        self.raw: dict[str, pd.DataFrame] = {}
        self.s1: dict[str, pd.DataFrame] = {}
        self.res: dict[str, pd.DataFrame] = {}
        self.last_obs: dict[str, pd.Timestamp] = {}
        self.clock: pd.Timestamp | None = None  # stream clock = latest observation time seen
        self.revisions: list[dict] = []  # provisional results revised once neighbour data arrived
        self.lock = threading.Lock()  # ponytail: one global lock; per-station locks if throughput matters

    # ------------------------------------------------------------------ stations
    def station_status(self, sid: str) -> dict:
        r = self.res.get(sid)
        last = self.last_obs.get(sid)
        offline_after = pd.Timedelta(minutes=self.cfg["realtime"]["offline_after_minutes"])
        if last is None:
            return {"status": "GRAY", "label": "No data", "online": False}
        if self.clock is not None and self.clock - last > offline_after:
            return {"status": "GRAY", "label": "Offline", "online": False}
        recent = r[(r.index > last - pd.Timedelta("2h")) & ~r["placeholder"].fillna(False).astype(bool)] if r is not None else None
        sev = set(recent.loc[recent["flag"].astype(bool) & (recent["pattern"] != "MISSING_COMMUNICATION"), "severity"]) \
            if recent is not None else set()
        hs = float(r["health_score"].dropna().iloc[-1]) if r is not None and r["health_score"].notna().any() else 100.0
        if "CRITICAL" in sev or hs < 50:
            s = ("RED", "Critical")
        elif "HIGH" in sev or hs < 70:
            s = ("ORANGE", "High")
        elif sev or hs < 85:
            s = ("YELLOW", "Watch")
        else:
            s = ("GREEN", "Normal")
        return {"status": s[0], "label": s[1], "online": True}

    def station_summary(self, sid: str) -> dict:
        r = self.res.get(sid)
        out = {"station_id": sid, **{k: self.stations.get(sid, {}).get(k) for k in ("latitude", "longitude", "altitude_m", "source")},
               "neighbors": [{"station_id": n, "distance_km": round(d, 1)} for n, d in self.nb.get(sid, [])],
               "last_observation_time": self.last_obs[sid].isoformat() if sid in self.last_obs else None,
               **self.station_status(sid)}
        scored = r[r["health_score"].notna()] if r is not None and "health_score" in r else None
        if scored is not None and len(scored):
            last = scored.iloc[-1]
            out["health_score"] = float(last["health_score"])
            out["health_status"] = str(last["health_status"])
            out["health_reasons"] = health_reasons(last)
            out["maintenance"] = maintenance_risk(scored.assign(flag=scored["flag"].astype(bool)), self.cfg)
        else:
            out["health_score"], out["health_status"], out["health_reasons"] = None, None, []
            out["maintenance"] = {"risk": "INSUFFICIENT_HISTORY", "reason": "Insufficient history for maintenance prediction."}
        return out

    def forget(self, sid: str, observation_time: str) -> None:
        """Undo one accepted observation (used when persisting it failed), so a retry is not a duplicate."""
        with self.lock:
            ot = pd.Timestamp(observation_time)
            for store in (self.raw, self.res):
                if sid in store:
                    store[sid] = store[sid].drop(index=[ot], errors="ignore")
            buf = self.raw.get(sid)
            actual = buf.index[buf.notna().any(axis=1)] if buf is not None else []
            if len(actual):
                self.last_obs[sid] = actual.max()
                self.s1[sid] = station_features(buf, self.cfg)
            else:
                self.last_obs.pop(sid, None)
                self.s1.pop(sid, None)

    # ------------------------------------------------------------------ processing
    def process(self, obs: dict, ingestion_time: datetime | None = None) -> dict:
        return self.process_many([obs], ingestion_time)[0]

    def process_many(self, observations: list[dict], ingestion_time: datetime | None = None) -> list[dict]:
        """Score a batch. Per station, in-order new records are scored together (features computed once
        over the buffer, past-only), which keeps buffered uploads and warm-up fast. Out-of-order
        records are scored individually against the history preceding their observation time."""
        with self.lock:
            t0 = time.perf_counter()
            it = pd.Timestamp(ingestion_time or datetime.now(timezone.utc))
            out: list = [None] * len(observations)
            by_station: dict[str, list] = {}
            scored: dict[str, list] = {}
            for i, obs in enumerate(observations):
                ot = pd.Timestamp(obs["timestamp"])
                ot = ot.tz_localize("UTC") if ot.tzinfo is None else ot.tz_convert("UTC")
                by_station.setdefault(obs["station_id"], []).append((ot, i, obs))
            for sid, items in by_station.items():
                pending, seen = [], set()
                for ot, i, obs in sorted(items, key=lambda x: x[0]):
                    base = {"station_id": sid, "observation_time": ot.isoformat(), "ingestion_time": it.isoformat(),
                            **{v: _num(obs.get(v)) for v in VARS}}
                    if sid not in self.stations:
                        out[i] = base | {"status": "rejected", "reason": f"unknown station '{sid}'"}
                        continue
                    if ot > it + pd.Timedelta(minutes=self.cfg["qc"]["future_tolerance_minutes"]):
                        out[i] = base | {"status": "rejected", "reason": "observation time is in the future"}
                        continue
                    buf = self.raw.get(sid)
                    if ot in seen or (buf is not None and ot in buf.index and buf.loc[ot, VARS].notna().any()):
                        out[i] = base | {"status": "duplicate", "reason": "observation already ingested (idempotent)"}
                        continue
                    seen.add(ot)
                    last = self.last_obs.get(sid)
                    if last is not None and ot < last and not pending:
                        self._score(sid, [(ot, i, obs, base)], it, out, out_of_order=True)
                        scored.setdefault(sid, []).append(ot)
                    else:
                        pending.append((ot, i, obs, base))
                if pending:
                    self._score(sid, pending, it, out)
                    scored.setdefault(sid, []).extend(p[0] for p in pending)
            # spatial context completed by these records: revise neighbours' provisional results
            for sid, ts in scored.items():
                for nid, _ in self.nb.get(sid, []):
                    self._revise(nid, pd.DatetimeIndex(ts))
            n = max(1, sum(r["status"] == "accepted" for r in out))
            per = (time.perf_counter() - t0) * 1000 / n
            for r in out:
                if r["status"] == "accepted":
                    r["processing_latency_ms"] = round(per, 2)
                    r["batch_size"] = len(observations)
            return out

    def pop_revisions(self) -> list[dict]:
        with self.lock:
            out, self.revisions = self.revisions, []
            return out

    def _revise(self, nid: str, times: pd.DatetimeIndex) -> None:
        """Re-score rows of station `nid` at `times` whose spatial context was incomplete when first scored
        (in a live network the first station to report an hour has no neighbour data for that hour yet)."""
        if self.bundle is None or nid not in self.res or nid not in self.s1:
            return
        cfg = self.cfg
        r = self.res[nid]
        cand = times[times.isin(r.index) & times.isin(self.s1[nid].index)]
        if not len(cand):
            return
        rows = r.loc[cand]
        rows = rows[~rows["placeholder"].fillna(False).astype(bool) & ~rows["qc_hard"].fillna(False).astype(bool)]
        if rows.empty:
            return
        nbrs = [(self.s1[n], d) for n, d in self.nb.get(nid, []) if n in self.s1 and len(self.s1[n])]
        sp = spatial_features(self.s1[nid].loc[rows.index], nbrs, cfg)
        better = sp["sp_n"].to_numpy() > rows["sp_n"].fillna(0).to_numpy()
        if not better.any():
            return
        idx = rows.index[better]
        F = rows.loc[idx].copy()
        F[sp.columns] = sp.loc[idx]
        h = self.bundle.hybrid
        prob, reg = h.predict_proba(F)
        pattern, param, conf = h.classify(F, prob)
        raw = h._raw(F)
        suppressed = (reg & (prob < h.threshold) & (prob / h.regional_suppression >= h.threshold)) if h.regional_suppression < 1             else np.zeros(len(F), dtype=bool)
        old = F[["flag", "pattern", "prob"]].copy()
        F = F.assign(prob=prob, regional_evidence=reg, pattern=pattern, param=param, pattern_conf=conf)
        F["flag"] = F["pattern"] != "NORMAL"
        touched = old["flag"].astype(bool).to_numpy() | F["flag"].to_numpy(bool) | suppressed
        res = self.res[nid].copy()
        for c in list(sp.columns) + ["prob", "regional_evidence"]:  # refreshed spatial context for every revised row
            res.loc[idx, c] = F[c].to_numpy()
        if not touched.any():  # decisions unchanged and normal: nothing else to recompute (cheap path)
            self.res[nid] = res
            return
        keep = touched
        F, old, raw, suppressed, idx = F[keep], old[keep], raw[keep], suppressed[keep], idx[keep]
        for c in list(sp.columns) + ["prob", "regional_evidence", "pattern", "param", "pattern_conf", "flag"]:
            res[c] = res[c].astype(object) if c in ("pattern", "param") else res[c]
            res.loc[idx, c] = F[c].to_numpy()
        window = res[(res.index > idx[0] - pd.Timedelta("2D")) & (res.index <= idx[-1])].copy()
        for c in ("flag", "qc_hard", "placeholder", "regional_evidence"):
            window[c] = window[c].fillna(False).astype(bool)
        window["pattern"] = window["pattern"].fillna("MISSING_COMMUNICATION")
        window["param"] = window["param"].fillna("all").astype(str)
        window["gap_h"] = window["gap_h"].fillna(0)
        window["prob"] = window["prob"].fillna(1.0)
        sc, lv = severity(window, cfg)
        post = pd.concat([pd.DataFrame({"severity_score": sc, "severity": lv}, index=window.index),
                          corrections(window, self.bundle.correction_method)], axis=1).loc[idx]
        for c in post.columns:
            res.loc[idx, c] = post[c].to_numpy()
        self.res[nid] = res
        F = F.drop(columns=[c for c in post.columns if c in F.columns]).join(post)
        ml_rows = F["flag"].to_numpy(bool) | suppressed
        contrib = h.contributions(F[ml_rows]) if ml_rows.any() else None
        for k, (ot, row) in enumerate(F.iterrows()):
            o = old.loc[ot]
            if not o["flag"] and not row["flag"] and not suppressed[k]:
                continue  # stays normal: nothing to revise (flagged rows get their evidence refreshed)
            exp = explain(row, contrib.loc[ot].to_dict() if contrib is not None and ot in contrib.index else None, h.threshold, self.mode)                 if ml_rows[k] else None
            self.revisions.append({
                "station_id": nid, "observation_time": ot.isoformat(), "was_anomaly": bool(o["flag"]),
                "was_pattern": str(o["pattern"]), "is_anomaly": bool(row["flag"]), "anomaly_probability": _num(row["prob"]),
                "anomaly_score": _num(raw[k]), "fault_pattern": str(row["pattern"]), "parameter": str(row["param"]),
                "severity": str(row["severity"]) or None, "severity_score": round(float(row["severity_score"]), 1),
                "regional_evidence": bool(row["regional_evidence"]), "regional_suppressed": bool(suppressed[k]),
                **{f"estimated_{v}": _num(row.get(f"estimated_{v}")) for v in VARS},
                "replacement_recommended": bool(row.get("replacement_recommended", False)),
                "explanation": exp, "reason": "re-scored after neighbour observations for the same time arrived",
                "raw": {v: _num(row.get(v)) for v in VARS}, "detector_mode": self.mode,
            })

    def _score(self, sid: str, rows: list, it: pd.Timestamp, out: list, out_of_order: bool = False) -> None:
        cfg = self.cfg
        buf = self.raw.get(sid, pd.DataFrame(columns=VARS, dtype=float, index=pd.DatetimeIndex([], tz="UTC")))
        prev_last = None if out_of_order else self.last_obs.get(sid)
        meta = {}
        for ot, i, obs, base in rows:
            clock = self.clock if self.clock is not None else ot
            historical = it - clock > pd.Timedelta(days=1)  # replay of history: arrival judged on the network clock
            ref = max(clock, ot) if historical else it
            chk = ingest_check(ot.to_pydatetime(), ref.to_pydatetime(), cfg)
            gap_event = None
            if prev_last is not None and ot - prev_last > 1.5 * self.step:
                fill = pd.date_range(prev_last + self.step, ot - self.step / 2, freq=self.step)
                fill = fill[~fill.isin(buf.index)]
                if len(fill):
                    buf = pd.concat([buf, pd.DataFrame(np.nan, index=fill, columns=VARS)])
                    ph = pd.DataFrame({"flag": True, "pattern": "MISSING_COMMUNICATION", "param": "all", "prob": 1.0,
                                       "qc_hard": True, "placeholder": True}, index=fill)
                    self.res[sid] = pd.concat([self.res.get(sid, pd.DataFrame()), ph])
                gap_h = (ot - prev_last).total_seconds() / 3600
                gap_event = {"start": (prev_last + self.step).isoformat(), "end": (ot - self.step).isoformat(),
                             "hours": round(gap_h, 2),
                             "explanation": f"No observations received for {gap_h:.1f} h (observation-time gap). "
                                            "Likely pattern: missing data / communication gap."}
            vals = {v: _num(obs.get(v)) for v in VARS}
            row = pd.DataFrame({v: [np.nan if vals[v] is None else vals[v]] for v in VARS}, index=[ot]).astype(float)
            buf = pd.concat([buf.drop(index=[ot], errors="ignore"), row])
            if not out_of_order and any(x is not None for x in vals.values()):
                prev_last = ot
            self.clock = ot if self.clock is None or ot > self.clock else self.clock
            meta[ot] = (i, obs, base, chk, historical, gap_event)
        buf = buf.sort_index()
        buf = buf[buf.index >= buf.index.max() - pd.Timedelta(days=11)]  # 7-day lags + 3-day MAD window + margin
        self.raw[sid] = buf
        actual = buf.index[buf.notna().any(axis=1)]
        self.last_obs[sid] = actual.max() if len(actual) else rows[-1][0]
        times = pd.DatetimeIndex([r[0] for r in rows])

        if out_of_order:
            s1h = station_features(buf[buf.index <= times[-1]], cfg)
            self.s1[sid] = station_features(buf, cfg)
        else:
            s1h = self.s1[sid] = station_features(buf, cfg)
        own = s1h.loc[times]
        nbrs = [(self.s1[n], d) for n, d in self.nb.get(sid, []) if n in self.s1 and len(self.s1[n])]
        F = pd.concat([own, spatial_features(own, nbrs, cfg), buf.loc[times, VARS],
                       pd.DataFrame({"station_id": sid}, index=times)], axis=1)
        n = len(F)
        raw_score = np.full(n, np.nan)
        suppressed = np.zeros(n, dtype=bool)
        if self.bundle is not None:
            F = F.join(self.bundle.ae.score(s1h).loc[times, AE_COLS])
            h = self.bundle.hybrid
            prob, reg = h.predict_proba(F)
            pattern, param, conf = h.classify(F, prob)
            soft = ~F["qc_hard"].to_numpy(bool)
            if soft.any():
                raw_score[soft] = h._raw(F[soft])
            if h.regional_suppression < 1:  # alerts that regional (weather-event) evidence pushed below threshold
                suppressed = soft & reg & (prob < h.threshold) & (prob / h.regional_suppression >= h.threshold)
        else:  # explicit reduced-confidence fallback: deterministic QC + fixed thresholds only
            hard = F["qc_hard"].to_numpy(bool)
            fixed = fixed_threshold_score(F).astype(bool)
            prob, reg = np.full(n, np.nan), np.zeros(n, dtype=bool)
            pattern = np.where(hard, F["qc_hard_class"].to_numpy(object),
                               np.where(fixed, "UNKNOWN_SUSPECT", "NORMAL")).astype(object)
            param, conf = np.where(hard, "all", "").astype(object), (hard | fixed).astype(float)
            raw_score = fixed.astype(float)
        F = F.assign(prob=np.nan_to_num(prob, nan=1.0), regional_evidence=reg, pattern=pattern, param=param,
                     pattern_conf=conf, placeholder=False)
        F["flag"] = F["pattern"] != "NORMAL"

        past = self.res.get(sid, pd.DataFrame())
        if len(past):
            past = past[(past.index > times[0] - pd.Timedelta("2D")) & (past.index <= times[-1]) & ~past.index.isin(times)]
        window = pd.concat([past, F]).sort_index() if len(past) else F.copy()
        for c in ("flag", "qc_hard", "placeholder", "regional_evidence"):
            window[c] = window[c].fillna(False).astype(bool)
        window["pattern"] = window["pattern"].fillna("MISSING_COMMUNICATION")
        window["param"] = window["param"].fillna("all").astype(str)
        window["gap_h"] = window["gap_h"].fillna(0)
        window["prob"] = window["prob"].fillna(1.0)
        sc, lv = severity(window, cfg)
        post = pd.concat([pd.DataFrame({"severity_score": sc, "severity": lv}, index=window.index), health(window, cfg),
                          corrections(window, self.bundle.correction_method if self.bundle else "temporal")], axis=1).loc[times]
        F = pd.concat([F.drop(columns=[c for c in post.columns if c in F.columns]), post], axis=1)
        if self.bundle is None:
            F["prob"] = np.nan
        explain_rows = F["flag"].to_numpy(bool) | suppressed
        contrib = None
        ml_rows = explain_rows & ~F["qc_hard"].to_numpy(bool)
        if self.bundle is not None and ml_rows.any():
            contrib = self.bundle.hybrid.contributions(F[ml_rows])
        thr = self.bundle.hybrid.threshold if self.bundle else float("nan")
        hist_res = pd.concat([self.res.get(sid, pd.DataFrame()).drop(index=times, errors="ignore"), F]).sort_index()
        self.res[sid] = hist_res[hist_res.index >= hist_res.index.max() - pd.Timedelta(days=RESULT_KEEP_DAYS)]

        for k, (ot, r) in enumerate(F.iterrows()):
            i, obs, base, chk, historical, gap_event = meta[ot]
            exp = None
            if explain_rows[k]:
                c = contrib.loc[ot].to_dict() if contrib is not None and ot in contrib.index else None
                exp = explain(r, c, thr, self.mode)
            out[i] = base | {
                "status": "accepted", "delayed": bool(chk["delayed"]), "lag_minutes": round(chk["lag_minutes"], 1),
                "lag_reference": "stream_clock" if historical else "ingestion_time",
                "out_of_order": bool(out_of_order), "detector_mode": self.mode,
                "is_anomaly": bool(r["flag"]), "anomaly_probability": _num(r["prob"]),
                "anomaly_score": _num(raw_score[k]), "fault_pattern": str(r["pattern"]), "parameter": str(r["param"]),
                "severity": str(r["severity"]) or None, "severity_score": round(float(r["severity_score"]), 1),
                "regional_evidence": bool(r["regional_evidence"]), "regional_suppressed": bool(suppressed[k]),
                "health_score": float(r["health_score"]), "health_status": str(r["health_status"]),
                **{f"estimated_{v}": _num(r.get(f"estimated_{v}")) for v in VARS},
                "replacement_recommended": bool(r.get("replacement_recommended", False)),
                "explanation": exp, "gap_event": gap_event,
                "edge_flags": obs.get("edge_flags"), "edge_health_score": obs.get("edge_health_score"),
            }

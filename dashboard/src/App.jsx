import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, getToken, setToken, subscribe } from "./api.js";
import {
  AlertDetail, AlertFeed, Bars, fmtTime, Kpi, LineChart, paramLabel, PATTERN, pct, SevChip, STATUS, StationMap, StatusDot, VARS,
} from "./components.jsx";

const TABS = ["Overview", "Station", "Alerts", "Analytics"];
const SPEEDS = [[1800, "30 min/s"], [3600, "1 h/s"], [7200, "2 h/s"], [21600, "6 h/s"]];

export default function App() {
  const [tab, setTab] = useState("Overview");
  const [health, setHealth] = useState(null);
  const [stations, setStations] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [scenario, setScenario] = useState("sih_demo");
  const [speed, setSpeed] = useState(7200);
  const [replay, setReplay] = useState({});
  const [sse, setSse] = useState("connecting");
  const [sel, setSel] = useState(null);
  const [rows, setRows] = useState([]);
  const [open, setOpen] = useState(null);
  const [err, setErr] = useState("");
  const [reviewErr, setReviewErr] = useState("");
  const [linkLog, setLinkLog] = useState([]);
  const [lastEvent, setLastEvent] = useState(null);
  const [tok, setTok] = useState(getToken());
  const [filter, setFilter] = useState({ severity: "", state: "", station: "" });
  const dirty = useRef(true);
  const selRef = useRef(sel);
  selRef.current = sel;

  const refresh = useCallback(async () => {
    try {
      const [h, s, a, m] = await Promise.all([api.health(), api.stations(), api.anomalies(), api.metrics()]);
      setHealth(h); setStations(s); setAlerts(a); setMetrics(m); setErr("");
      if (selRef.current) setRows(await api.history(selRef.current));
    } catch (e) { setErr(String(e.message || e)); }
  }, []);

  useEffect(() => {
    api.scenarios().then((s) => { setScenarios(s); if (s.length && !s.includes("sih_demo")) setScenario(s[0]); }).catch(() => {});
    api.replayStatus().then(setReplay).catch(() => {});
    const off = subscribe({
      observation: (o) => { dirty.current = true; setLastEvent(o); },
      anomaly: () => { dirty.current = true; },
      replay: (r) => { setReplay(r); dirty.current = true; },
      revision: () => { dirty.current = true; },
      link: (l) => { setLinkLog((x) => [l, ...x].slice(0, 30)); dirty.current = true; },
    }, setSse);
    const id = setInterval(() => { if (dirty.current) { dirty.current = false; refresh(); } }, 1000);
    const slow = setInterval(() => { dirty.current = true; }, 10000);
    return () => { off(); clearInterval(id); clearInterval(slow); };
  }, [refresh]);

  useEffect(() => {
    const withData = stations.find((s) => s.last_observation_time);
    if ((!sel || (withData && !stations.find((s) => s.station_id === sel)?.last_observation_time)) && withData) setSel(withData.station_id);
    else if (!sel && stations.length) setSel(stations[0].station_id);
  }, [stations, sel]);
  useEffect(() => { if (sel) api.history(sel).then(setRows).catch(() => {}); }, [sel]);

  const kpis = useMemo(() => {
    const online = stations.filter((s) => s.online).length;
    const offline = stations.filter((s) => s.label === "Offline").length;
    const degraded = stations.filter((s) => ["ORANGE", "RED"].includes(s.status) || ["DEGRADED", "CRITICAL"].includes(s.health_status)).length;
    const active = stations.reduce((n, s) => n + (s.active_anomalies || 0), 0);
    const critical = stations.reduce((n, s) => n + (s.active_critical || 0), 0);
    const hs = stations.filter((s) => s.health_score != null);
    const avg = hs.length ? hs.reduce((n, s) => n + s.health_score, 0) / hs.length : null;
    return { online, offline, degraded, active, critical, avg };
  }, [stations]);

  const station = stations.find((s) => s.station_id === sel);
  const liveAlerts = alerts.filter((a) => a.state !== "RETRACTED");
  const stationAlerts = liveAlerts.filter((a) => a.station_id === sel);
  const latestAlert = stationAlerts.find((a) => a.state !== "INFO");
  const sortedStations = [...stations].sort((a, b) => (!!b.last_observation_time - !!a.last_observation_time) || a.station_id.localeCompare(b.station_id));
  const lastRow = rows[rows.length - 1];
  const filtered = alerts.filter((a) => (!filter.severity || a.severity === filter.severity)
    && (!filter.state || a.state === filter.state) && (!filter.station || a.station_id === filter.station));

  const start = async () => {
    try { setReplay(await api.replayStart(scenario, Number(speed))); setLinkLog([]); dirty.current = true; } catch (e) { setErr(String(e.message)); }
  };
  const stop = async () => { try { setReplay(await api.replayStop()); } catch (e) { setErr(String(e.message)); } };
  const review = async (a, state) => {
    try { const u = await api.review(a.id, state); setOpen(u); setReviewErr(""); dirty.current = true; } catch (e) { setReviewErr(String(e.message)); }
  };
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") setOpen(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const openRow = (r) => {
    const a = alerts.find((x) => x.station_id === r.station_id && x.observation_time === r.observation_time && x.state !== "INFO");
    if (a) setOpen(a);
  };
  const lat = metrics?.processing_latency_ms;

  return (
    <div className="app">
      <header>
        <div className="brand"><span className="logo">◎</span> SkyGuard AI <span className="muted">AWS observation-quality layer</span></div>
        <div className="head-right">
          <span className={`pill ${sse === "live" ? "ok" : "warn"}`}>{sse === "live" ? "● Live" : "○ " + sse}</span>
          {health && <span className={`pill ${health.model_loaded ? "ok" : "bad"}`} title={health.model_version || ""}>
            Detector: {health.detector_mode === "hybrid" ? "SkyGuard Hybrid" : "QC-only FALLBACK (reduced confidence)"}</span>}
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} aria-label="Replay scenario">
            {scenarios.map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={speed} onChange={(e) => setSpeed(e.target.value)} aria-label="Replay speed">
            {SPEEDS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          {replay.running ? <button className="btn warn" onClick={stop}>Stop replay</button> : <button className="btn" onClick={start}>Start replay</button>}
          <input className="token" type="password" placeholder="Operator token" value={tok}
            onChange={(e) => { setTok(e.target.value); setToken(e.target.value); }} aria-label="Operator token" />
        </div>
      </header>
      {health && !health.model_loaded && <div className="banner bad">Model unavailable — running deterministic QC-only fallback. No anomaly probability is reported.</div>}
      {err && <div className="banner bad">{err}</div>}
      {(replay.running || replay.phase === "finished") && (
        <div className="replay">
          <span>Replay <b>{replay.scenario}</b> · {replay.phase} · stream time {fmtTime(replay.stream_time)}</span>
          <span className="progress"><span style={{ width: `${100 * (replay.progress || 0)}%` }} /></span>
          <span className="note">{replay.note}</span>
          <span className="muted">Edge checks: {replay.edge}</span>
        </div>
      )}
      <nav>{TABS.map((t) => <button key={t} className={tab === t ? "tab on" : "tab"} onClick={() => setTab(t)}>{t}</button>)}</nav>

      {tab === "Overview" && (
        <main>
          <section className="kpis">
            <Kpi label="Stations" value={stations.length} />
            <Kpi label="Online" value={kpis.online} tone="var(--green)" sub={`${kpis.offline} offline · ${stations.length - kpis.online - kpis.offline} no data`} />
            <Kpi label="Degraded" value={kpis.degraded} tone="var(--orange)" />
            <Kpi label="Active anomalies" value={kpis.active} tone="var(--yellow)" sub="last 2 h, open" />
            <Kpi label="Critical alerts" value={kpis.critical} tone="var(--red)" />
            <Kpi label="Avg sensor health" value={kpis.avg == null ? "—" : kpis.avg.toFixed(0)} sub="data-quality score /100" />
            <Kpi label="Data quality" value={metrics ? `${metrics.complete_observations_pct.toFixed(1)}%` : "—"} sub="complete T/P/RH records" />
            <Kpi label="p95 latency" value={lat ? `${lat.p95_ms.toFixed(0)} ms` : "—"} sub={metrics ? `${metrics.throughput_obs_per_s_last_60s.toFixed(1)} obs/s (60 s)` : ""} />
          </section>
          <section className="two">
            <div className="panel"><h3>Station map <span className="muted">(schematic lat/lon)</span></h3>
              <StationMap stations={sortedStations} selected={sel} onSelect={(s) => { setSel(s); setTab("Station"); }} />
              <div className="legend">{Object.entries(STATUS).map(([k, v]) => <StatusDot key={k} status={k} label={`${k} ${v.t}`} />)}</div>
            </div>
            <div className="panel"><h3>Stations</h3>
              <table className="table">
                <thead><tr><th>Station</th><th>Status</th><th>Health</th><th>Last observation</th><th>Active</th><th>Maintenance</th></tr></thead>
                <tbody>{sortedStations.map((s) => (
                  <tr key={s.station_id} onClick={() => { setSel(s.station_id); setTab("Station"); }} tabIndex={0}>
                    <td><b>{s.station_id}</b></td><td><StatusDot status={s.status} label={s.label} /></td>
                    <td>{s.health_score == null ? "—" : `${s.health_score.toFixed(0)} ${s.health_status}`}</td>
                    <td>{fmtTime(s.last_observation_time)}</td><td>{s.active_anomalies || 0}</td>
                    <td className={`risk r-${s.maintenance?.risk}`}>{s.maintenance?.risk?.replace("_", " ") || "—"}</td>
                  </tr>))}</tbody>
              </table>
            </div>
          </section>
          <section className="two">
            <div className="panel"><h3>Live alert feed</h3><AlertFeed alerts={liveAlerts.slice(0, 8)} onOpen={setOpen} compact /></div>
            <div className="panel"><h3>Connectivity log <span className="muted">(offline → buffer → reconnect → sync)</span></h3>
              {!linkLog.length && <div className="muted">No link changes yet.</div>}
              {linkLog.map((l, i) => (
                <div key={i} className="log">{fmtTime(l.stream_time)} <b>{l.station_id}</b> {l.online
                  ? <span className="ok-t">reconnected — uploading {l.buffered} buffered record(s)</span>
                  : <span className="bad-t">link down — buffering locally</span>}</div>))}
              {lastEvent && <div className="muted small">Last observation: {lastEvent.station_id} {fmtTime(lastEvent.observation_time)} · {lastEvent.processing_latency_ms} ms
                {lastEvent.delayed ? " · delayed" : ""}{lastEvent.out_of_order ? " · out-of-order" : ""}</div>}
            </div>
          </section>
        </main>
      )}

      {tab === "Station" && (
        <main>
          <div className="station-head">
            <select value={sel || ""} onChange={(e) => setSel(e.target.value)} aria-label="Station">
              {sortedStations.map((s) => <option key={s.station_id}>{s.station_id}</option>)}
            </select>
            {station && <>
              <StatusDot status={station.status} label={station.label} />
              <span>Sensor health <b>{station.health_score == null ? "—" : `${station.health_score.toFixed(0)}/100`}</b> {station.health_status}</span>
              <span className={`risk r-${station.maintenance?.risk}`}>Maintenance: {station.maintenance?.risk?.replace("_", " ")}</span>
              <span className="muted">{station.latitude?.toFixed(2)}°N {station.longitude?.toFixed(2)}°E · {station.altitude_m?.toFixed(0)} m · {station.source}</span>
            </>}
          </div>
          <section className="station-grid">
            <div>
              {VARS.map((v) => <LineChart key={v.k} rows={rows} v={v.k} label={v.label} unit={v.unit} onPick={openRow} />)}
              <div className="legend small"><span><i className="lg-dot" /> anomaly (colour = severity)</span><span><i className="lg-dia" /> estimated normal value (raw preserved)</span><span><i className="lg-tri" /> regional weather evidence</span></div>
            </div>
            <div className="side">
              <div className="panel"><h3>Latest anomaly</h3>
                {!latestAlert && <div className="muted">No anomaly for this station.</div>}
                {latestAlert && <div className="latest" onClick={() => setOpen(latestAlert)}>
                  <div><SevChip sev={latestAlert.severity} /> <b>{PATTERN[latestAlert.fault_pattern]}</b> · {paramLabel(latestAlert.parameter)}</div>
                  <div>Probability {latestAlert.anomaly_probability == null ? "QC rule" : pct(latestAlert.anomaly_probability)} · {fmtTime(latestAlert.observation_time)}</div>
                  <ul className="evidence small">{(latestAlert.explanation?.evidence || []).slice(0, 4).map((x, i) => <li key={i}>{x}</li>)}</ul>
                  <span className="link">Full explanation →</span>
                </div>}
              </div>
              <div className="panel"><h3>Sensor health</h3>
                <ul className="small">{(station?.health_reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul>
                <div className="small muted">{station?.maintenance?.reason}</div>
                <div className="small muted">Data-quality score, not a physical hardware diagnostic.</div>
              </div>
              <div className="panel"><h3>Raw vs estimated</h3>
                <table className="table small"><thead><tr><th>Time</th><th>Param</th><th>Raw</th><th>Estimated</th></tr></thead>
                  <tbody>{stationAlerts.filter((a) => a.estimated_value != null).slice(0, 6).map((a) => (
                    <tr key={a.id} onClick={() => setOpen(a)}><td>{fmtTime(a.observation_time)}</td><td>{paramLabel(a.parameter)}</td>
                      <td>{a.raw_value?.toFixed(1)}</td><td>{a.estimated_value?.toFixed(1)}</td></tr>))}</tbody></table>
                <div className="small muted">Raw values are stored unchanged; estimates are recommendations.</div>
              </div>
              <div className="panel"><h3>Edge check (on-station)</h3>
                {lastRow?.edge_health_score == null ? <div className="muted small">No edge payload on the latest record.</div> : <>
                  <div>Edge health <b>{lastRow.edge_health_score}</b> · flags: {(lastRow.edge_flags || []).length ? lastRow.edge_flags.join(", ") : "none"}</div>
                  <div className="small muted">Same C library as the ESP32 sketch (host build via ctypes).</div></>}
              </div>
              <div className="panel"><h3>Spatial neighbours</h3>
                {!station?.neighbors?.length ? <div className="small muted">None within radius — temporal + multivariate evidence only.</div>
                  : <div className="small">{station.neighbors.map((n) => `${n.station_id} (${n.distance_km} km)`).join(" · ")}</div>}
              </div>
            </div>
          </section>
        </main>
      )}

      {tab === "Alerts" && (
        <main>
          <div className="filters">
            <select value={filter.station} onChange={(e) => setFilter({ ...filter, station: e.target.value })} aria-label="Filter station"><option value="">All stations</option>{stations.map((s) => <option key={s.station_id}>{s.station_id}</option>)}</select>
            <select value={filter.severity} onChange={(e) => setFilter({ ...filter, severity: e.target.value })} aria-label="Filter severity"><option value="">All severities</option>{["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((s) => <option key={s}>{s}</option>)}</select>
            <select value={filter.state} onChange={(e) => setFilter({ ...filter, state: e.target.value })} aria-label="Filter state"><option value="">All states</option>{["OPEN", "ACKNOWLEDGED", "FALSE_ALARM", "RESOLVED", "RETRACTED", "INFO"].map((s) => <option key={s}>{s}</option>)}</select>
            <span className="muted">{filtered.length} alerts</span>
          </div>
          <div className="panel"><AlertFeed alerts={filtered} onOpen={setOpen} /></div>
        </main>
      )}

      {tab === "Analytics" && (
        <main>
          <section className="three">
            <Bars title="Anomalies per station" data={metrics?.anomalies_by_station} />
            <Bars title="Anomalies per parameter" data={metrics?.anomalies_by_parameter} fmt={paramLabel} />
            <Bars title="Fault-pattern distribution" data={metrics?.anomalies_by_pattern} fmt={(k) => PATTERN[k] || k} />
            <Bars title="Severity" data={metrics?.anomalies_by_severity} color="var(--orange)" />
            <Bars title="False-alarm review (operator states)" data={metrics?.review_states} color="var(--blue)" />
            <div className="panel"><h3>Real-time performance</h3>
              {lat ? <table className="table small"><tbody>
                <tr><td>Processing latency p50 / p95 / p99</td><td>{lat.p50_ms.toFixed(1)} / {lat.p95_ms.toFixed(1)} / {lat.p99_ms.toFixed(1)} ms</td></tr>
                <tr><td>End-to-end (incl. DB) p95</td><td>{metrics.end_to_end_latency_ms?.p95_ms.toFixed(1)} ms</td></tr>
                <tr><td>Throughput (last 60 s)</td><td>{metrics.throughput_obs_per_s_last_60s.toFixed(2)} obs/s</td></tr>
                <tr><td>Observations processed</td><td>{metrics.observations}</td></tr>
                <tr><td>Delayed / out-of-order</td><td>{metrics.delayed_observations} / {metrics.out_of_order_observations}</td></tr>
                <tr><td>Regional events (alerts suppressed)</td><td>{metrics.regional_events_suppressed}</td></tr>
                <tr><td>API process CPU / RSS</td><td>{metrics.cpu_percent.toFixed(0)}% / {metrics.rss_mb.toFixed(0)} MB</td></tr>
              </tbody></table> : <div className="muted">No observations processed yet.</div>}
            </div>
          </section>
          <div className="panel"><h3>Sensor-health trend (hourly mean per station)</h3><HealthTrend data={metrics?.health_trend || []} /></div>
        </main>
      )}
      <AlertDetail alert={open} onClose={() => { setOpen(null); setReviewErr(""); }} onReview={review} error={reviewErr} />
      <footer className="muted small">Observable fault patterns and data-quality scores — not a physical hardware diagnosis. Probabilities are isotonic-calibrated on validation data.</footer>
    </div>
  );
}

function HealthTrend({ data }) {
  const by = {};
  for (const d of data) (by[d.station_id] ||= []).push(d);
  const ids = Object.keys(by).sort();
  if (!ids.length) return <div className="muted">No data yet</div>;
  return (
    <div className="trends">
      {ids.map((id) => {
        const pts = by[id].slice().sort((a, b) => a.hour.localeCompare(b.hour));
        const W = 220, H = 50;
        const d = pts.map((p, i) => `${i ? "L" : "M"}${(i / Math.max(1, pts.length - 1)) * W},${H - (p.health / 100) * H}`).join("");
        const last = pts[pts.length - 1].health;
        return (
          <div key={id} className="trend"><div><b>{id}</b> <span className="muted">{last.toFixed(0)}</span></div>
            <svg viewBox={`0 0 ${W} ${H}`}><line x1={0} x2={W} y1={H * 0.15} y2={H * 0.15} className="grid" /><path d={d} className="series" /></svg></div>
        );
      })}
    </div>
  );
}

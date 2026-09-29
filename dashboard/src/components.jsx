import { useMemo, useState } from "react";

export const STATUS = {
  GREEN: { c: "var(--green)", t: "Normal" }, YELLOW: { c: "var(--yellow)", t: "Watch" },
  ORANGE: { c: "var(--orange)", t: "High" }, RED: { c: "var(--red)", t: "Critical" }, GRAY: { c: "var(--gray)", t: "Offline" },
};
export const SEV = { LOW: "var(--sev-low)", MEDIUM: "var(--yellow)", HIGH: "var(--orange)", CRITICAL: "var(--red)" };
export const VARS = [
  { k: "temperature_c", label: "Temperature", unit: "°C" },
  { k: "pressure_hpa", label: "Pressure", unit: "hPa" },
  { k: "relative_humidity_pct", label: "Relative humidity", unit: "%" },
];
const VLABEL = Object.fromEntries(VARS.map((v) => [v.k, v.label]));
export const PATTERN = {
  SPIKE: "Spike", NEGATIVE_SPIKE: "Negative spike", FLATLINE: "Flatline / frozen", DRIFT: "Gradual drift",
  STEP_OFFSET: "Step / offset", MISSING_COMMUNICATION: "Missing / comm. gap", SENTINEL_CORRUPTION: "Corrupt value",
  MULTIVARIATE_INCONSISTENCY: "Multivariate inconsistency", SPATIAL_INCONSISTENCY: "Spatial inconsistency",
  COMPOSITE_FAULT: "Composite fault", UNKNOWN_SUSPECT: "Unclassified suspect", REGIONAL_WEATHER_EVENT: "Regional weather event",
};

export const fmtTime = (s) => (s ? new Date(s).toISOString().slice(5, 16).replace("T", " ") + "Z" : "—");
export const pct = (p) => (p == null ? "n/a" : `${(100 * p).toFixed(1)}%`);
export const paramLabel = (p) => (!p ? "—" : p === "all" ? "All parameters" : p.split("+").map((x) => VLABEL[x] || x).join(" + "));
const num = (x, d = 1) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));

export function Kpi({ label, value, sub, tone }) {
  return (
    <div className="kpi" style={tone ? { borderTopColor: tone } : undefined}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function StatusDot({ status, label }) {
  const s = STATUS[status] || STATUS.GRAY;
  return (
    <span className="status"><span className="dot" style={{ background: s.c }} />{label || s.t}</span>
  );
}

export function SevChip({ sev }) {
  if (!sev) return <span className="chip muted">—</span>;
  return <span className="chip" style={{ background: SEV[sev], color: sev === "MEDIUM" ? "#1a1a1a" : "#fff" }}>{sev}</span>;
}

// Schematic latitude/longitude plot (not a political boundary map).
export function StationMap({ stations, selected, onSelect }) {
  const W = 360, H = 380, lon0 = 67, lon1 = 98, lat0 = 5, lat1 = 37;
  const x = (lo) => ((lo - lon0) / (lon1 - lon0)) * W;
  const y = (la) => H - ((la - lat0) / (lat1 - lat0)) * H;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="map" role="img" aria-label="Station positions by latitude and longitude">
      {[10, 15, 20, 25, 30, 35].map((la) => (
        <g key={la}><line x1={0} x2={W} y1={y(la)} y2={y(la)} className="grid" /><text x={4} y={y(la) - 3} className="axis">{la}°N</text></g>
      ))}
      {[70, 75, 80, 85, 90, 95].map((lo) => (
        <g key={lo}><line y1={0} y2={H} x1={x(lo)} x2={x(lo)} className="grid" /><text x={x(lo) + 3} y={H - 4} className="axis">{lo}°E</text></g>
      ))}
      {stations.map((s) => s.latitude != null && (
        <g key={s.station_id} className="map-pt" onClick={() => onSelect(s.station_id)} style={{ cursor: "pointer" }}>
          <circle cx={x(s.longitude)} cy={y(s.latitude)} r={selected === s.station_id ? 9 : 6.5}
            fill={(STATUS[s.status] || STATUS.GRAY).c} stroke={selected === s.station_id ? "var(--fg)" : "var(--bg)"} strokeWidth={2} />
          <text x={x(s.longitude) + 9} y={y(s.latitude) + 4} className="map-label">{s.station_id}</text>
          <title>{`${s.station_id}: ${(STATUS[s.status] || STATUS.GRAY).t}, health ${num(s.health_score, 0)}`}</title>
        </g>
      ))}
    </svg>
  );
}

const affects = (row, v) => row.is_anomaly && row.parameter && (row.parameter === "all" || row.parameter.includes(v));

export function LineChart({ rows, v, label, unit, onPick }) {
  const [hover, setHover] = useState(null);
  const W = 720, H = 170, L = 52, R = 12, T = 14, B = 24;
  const pts = useMemo(() => rows.map((r) => ({
    t: new Date(r.observation_time).getTime(), y: r[v], est: r[`estimated_${v}`], row: r,
    corrupt: r.is_anomaly && r.fault_pattern === "SENTINEL_CORRUPTION" && affects(r, v),
  })), [rows, v]);
  if (!pts.length) return <div className="chart-empty">No observations yet</div>;
  const good = pts.filter((p) => p.y != null && !p.corrupt).map((p) => p.y).concat(pts.filter((p) => p.est != null).map((p) => p.est));
  let lo = Math.min(...good), hi = Math.max(...good);
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (hi - lo < 1) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  const t0 = pts[0].t, t1 = Math.max(pts[pts.length - 1].t, t0 + 1);
  const X = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
  const Y = (val) => T + (1 - (Math.min(Math.max(val, lo), hi) - lo) / (hi - lo)) * (H - T - B);
  let d = "", pen = false;
  for (const p of pts) {
    if (p.y == null || p.corrupt) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${X(p.t).toFixed(1)},${Y(p.y).toFixed(1)}`;
    pen = true;
  }
  const move = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const tx = t0 + ((e.clientX - box.left) / box.width * W - L) / (W - L - R) * (t1 - t0);
    let best = 0;
    pts.forEach((p, i) => { if (Math.abs(p.t - tx) < Math.abs(pts[best].t - tx)) best = i; });
    setHover(best);
  };
  const hp = hover != null ? pts[hover] : null;
  return (
    <div className="chart">
      <div className="chart-title">{label} <span className="muted">({unit})</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={move} onMouseLeave={() => setHover(null)}
        onClick={() => hp && hp.row.is_anomaly && onPick && onPick(hp.row)} role="img" aria-label={`${label} time series`}>
        {[lo, (lo + hi) / 2, hi].map((g, i) => (
          <g key={i}><line x1={L} x2={W - R} y1={Y(g)} y2={Y(g)} className="grid" /><text x={L - 6} y={Y(g) + 4} className="axis" textAnchor="end">{g.toFixed(1)}</text></g>
        ))}
        <text x={L} y={H - 6} className="axis">{fmtTime(pts[0].row.observation_time)}</text>
        <text x={W - R} y={H - 6} className="axis" textAnchor="end">{fmtTime(pts[pts.length - 1].row.observation_time)}</text>
        <path d={d} className="series" />
        {pts.map((p, i) => p.row.regional_suppressed && (
          <path key={`r${i}`} d={`M${X(p.t)},${T} l-5,-8 h10z`} transform={`translate(0,8)`} fill="var(--blue)"><title>Regional weather event evidence</title></path>
        ))}
        {pts.map((p, i) => affects(p.row, v) && p.est != null && p.y != null && !p.corrupt && (
          <line key={`e${i}`} x1={X(p.t)} x2={X(p.t)} y1={Y(p.y)} y2={Y(p.est)} className="est-link" />
        ))}
        {pts.map((p, i) => affects(p.row, v) && p.est != null && (
          <rect key={`d${i}`} x={X(p.t) - 4} y={Y(p.est) - 4} width={8} height={8} transform={`rotate(45 ${X(p.t)} ${Y(p.est)})`} className="est-pt" />
        ))}
        {pts.map((p, i) => affects(p.row, v) && (
          <circle key={`a${i}`} cx={X(p.t)} cy={p.corrupt || p.y == null ? H - B - 3 : Y(p.y)} r={4.5}
            fill={SEV[p.row.severity] || "var(--red)"} stroke="var(--bg)" strokeWidth={1.5} />
        ))}
        {hp && <line x1={X(hp.t)} x2={X(hp.t)} y1={T} y2={H - B} className="cursor" />}
      </svg>
      {hp && (
        <div className="tip">
          <b>{fmtTime(hp.row.observation_time)}</b> raw {hp.y == null ? "missing" : `${num(hp.y)} ${unit}`}
          {affects(hp.row, v) && <> · <span style={{ color: SEV[hp.row.severity] }}>{PATTERN[hp.row.fault_pattern] || hp.row.fault_pattern}</span>
            {" "}p={pct(hp.row.anomaly_probability)}{hp.est != null && <> · estimated {num(hp.est)} {unit} (raw preserved)</>}</>}
          {hp.row.regional_suppressed && <> · regional weather evidence</>}
        </div>
      )}
    </div>
  );
}

export function Bars({ title, data, color = "var(--accent)", fmt = (k) => k }) {
  const entries = Object.entries(data || {}).filter(([k]) => k && k !== "null").sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map((e) => e[1]));
  return (
    <div className="panel">
      <h3>{title}</h3>
      {!entries.length && <div className="muted">No data yet</div>}
      {entries.map(([k, n]) => (
        <div className="bar-row" key={k}>
          <span className="bar-label" title={fmt(k)}>{fmt(k)}</span>
          <span className="bar-track"><span className="bar" style={{ width: `${(100 * n) / max}%`, background: color }} /></span>
          <span className="bar-n">{n}</span>
        </div>
      ))}
    </div>
  );
}

export function AlertFeed({ alerts, onOpen, compact }) {
  if (!alerts.length) return <div className="muted pad">No alerts. Start the demo replay or post observations.</div>;
  return (
    <table className="table feed">
      <thead><tr><th>Time (obs)</th><th>Station</th><th>Parameter</th><th>Fault pattern</th><th>Severity</th><th>Probability</th>{!compact && <th>Explanation</th>}<th>State</th></tr></thead>
      <tbody>
        {alerts.map((a) => (
          <tr key={a.id} onClick={() => onOpen(a)} className={a.state === "INFO" ? "info-row" : ""} tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && onOpen(a)}>
            <td>{fmtTime(a.observation_time)}</td>
            <td><b>{a.station_id}</b></td>
            <td>{paramLabel(a.parameter)}</td>
            <td>{PATTERN[a.fault_pattern] || a.fault_pattern}</td>
            <td>{a.state === "INFO" ? <span className="chip info">INFO</span> : <SevChip sev={a.severity} />}</td>
            <td>{a.state === "INFO" ? "suppressed" : a.anomaly_probability == null ? "QC rule" : pct(a.anomaly_probability)}</td>
            {!compact && <td className="ellipsis">{a.explanation?.evidence?.[0] || "—"}</td>}
            <td><span className={`state s-${a.state}`}>{a.state}</span></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AlertDetail({ alert, onClose, onReview, error }) {
  if (!alert) return null;
  const e = alert.explanation || {};
  const unit = VARS.find((v) => v.k === alert.parameter)?.unit || "";
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Anomaly explanation" onClick={(ev) => ev.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">{alert.state === "INFO" ? "REGIONAL WEATHER EVENT — NOT TREATED AS SENSOR FAULT" : e.status || "ANOMALY"}</div>
            <h2>{alert.station_id} · {PATTERN[alert.fault_pattern] || alert.fault_pattern}</h2>
            <div className="muted">Observation time {fmtTime(alert.observation_time)} · detected {fmtTime(alert.detected_at)} · detector: {alert.detector_mode} · state <span className={`state s-${alert.state}`}>{alert.state}</span></div>
          </div>
          <button className="btn ghost" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="detail-grid">
          <div className="stat"><span>Probability</span><b>{alert.anomaly_probability == null ? "QC rule" : pct(alert.anomaly_probability)}</b>
            <small>{alert.anomaly_probability == null ? "deterministic check" : `isotonic-calibrated; alert ≥ ${pct(e.alert_threshold)}`}</small></div>
          <div className="stat"><span>Severity</span><b><SevChip sev={alert.severity} /></b><small>documented formula (README)</small></div>
          <div className="stat"><span>Raw value</span><b>{alert.raw_value == null ? "—" : `${num(alert.raw_value)} ${unit}`}</b><small>preserved, never overwritten</small></div>
          <div className="stat"><span>Estimated normal</span><b>{alert.estimated_value == null ? "—" : `${num(alert.estimated_value)} ${unit}`}</b><small>recommendation only</small></div>
        </div>
        <p className="pattern-line">{e.likely_fault_pattern}</p>
        <h3>Evidence</h3>
        <ul className="evidence">{(e.evidence || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
        {!!(e.feature_group_contributions || []).length && (
          <>
            <h3>Feature-group contribution <span className="muted">(drop in probability when the group is neutralised)</span></h3>
            {e.feature_group_contributions.map(([g, val]) => (
              <div className="bar-row" key={g}><span className="bar-label">{g.replace("_", " ")}</span>
                <span className="bar-track"><span className="bar" style={{ width: `${Math.min(100, 100 * val)}%` }} /></span><span className="bar-n">{(100 * val).toFixed(0)}pp</span></div>
            ))}
          </>
        )}
        {alert.estimated_value != null && <p className="recommend">Recommendation: use the estimated value downstream. Original observation: preserved.</p>}
        <p className="muted small">{e.note}</p>
        {alert.state !== "INFO" && (
          <div className="actions">
            <button className="btn" onClick={() => onReview(alert, "ACKNOWLEDGED")}>Acknowledge</button>
            <button className="btn warn" onClick={() => onReview(alert, "FALSE_ALARM")}>Mark false alarm</button>
            <button className="btn ghost" onClick={() => onReview(alert, "RESOLVED")}>Resolve</button>
            {error && <span className="err">{error}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

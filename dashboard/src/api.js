// Thin API client. Write calls send the operator token (kept in sessionStorage only) when set.
const BASE = "/api/v1";

export function getToken() {
  try { return sessionStorage.getItem("skyguard_token") || ""; } catch { return ""; }
}
export function setToken(t) {
  try { sessionStorage.setItem("skyguard_token", t); } catch { /* storage unavailable */ }
}

async function req(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  const tok = getToken();
  if (tok && opts.method && opts.method !== "GET") headers.Authorization = `Bearer ${tok}`;
  const r = await fetch(BASE + path, { ...opts, headers });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* non-JSON error */ }
    throw new Error(`${r.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return r.json();
}

export const api = {
  health: () => req("/health"),
  stations: () => req("/stations"),
  anomalies: (q = "") => req(`/anomalies?limit=300${q}`),
  history: (sid, limit = 240) => req(`/observations/history?station_id=${encodeURIComponent(sid)}&limit=${limit}`),
  metrics: () => req("/metrics"),
  scenarios: () => req("/replay/scenarios"),
  replayStatus: () => req("/replay/status"),
  replayStart: (scenario, speed) => req("/replay/start", { method: "POST", body: JSON.stringify({ scenario, speed, reset: true }) }),
  replayStop: () => req("/replay/stop", { method: "POST", body: "{}" }),
  review: (id, state) => req(`/anomalies/${id}`, { method: "PATCH", body: JSON.stringify({ state }) }),
};

// Server-Sent Events with automatic reconnect (EventSource retries on its own).
export function subscribe(handlers, onState) {
  const es = new EventSource(BASE + "/stream");
  es.onopen = () => onState("live");
  es.onerror = () => onState("reconnecting");
  for (const [ev, fn] of Object.entries(handlers)) es.addEventListener(ev, (e) => fn(JSON.parse(e.data)));
  return () => es.close();
}

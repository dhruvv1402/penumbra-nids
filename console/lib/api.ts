/**
 * API client for the Penumbra service.
 *
 * Requests go through Next's rewrite to /api/*, so the browser stays on one origin and the demo
 * works with the API bound to localhost only.
 *
 * Note what is absent: there is no `block()`, no `quarantine()`, no enforcement call of any kind.
 * The API has no such endpoint (ADR-0001) and the client could not reach one if it did.
 */

const BASE = "/api";

export type Verdict =
  | "KNOWN_ATTACK"
  | "SUSPECTED_NOVEL"
  | "UNCERTAIN"
  | "BENIGN"
  | "BENIGN_BY_POLICY";

export type Lane = "known_threat" | "hunting" | "review";
export type Severity = "Informational" | "Low" | "Medium" | "High";

export interface Contribution {
  feature: string;
  value: number | string | null;
  shap_value: number;
  direction: "toward_attack" | "toward_benign";
  narrative: string;
}

export interface AttackTechnique {
  tactic_id: string;
  tactic_name: string;
  technique_id: string;
  technique_name: string;
  confidence: "strong" | "good" | "medium" | "stretch";
  rationale: string;
}

export interface NetworkContext {
  src_ip: string | null;
  src_port: number | null;
  dst_ip: string | null;
  dst_port: number | null;
  protocol: string | null;
  src_bytes: number | null;
  dst_bytes: number | null;
  is_pseudonymised: boolean;
}

export interface Alert {
  alert_id: string;
  timestamp: string;
  verdict: Verdict;
  lane: Lane;
  severity: Severity;
  /** Calibrated P(attack). The only calibrated number the system emits. */
  p_attack: number;
  /** Percentile against known-benign traffic. Not a probability. */
  novelty_percentile: number;
  /** Triage sort key, 0-100. Explicitly NOT a probability. */
  priority: number;
  family: string | null;
  attack: AttackTechnique | null;
  network: NetworkContext;
  contributions: Contribution[];
  detector_agreement: number;
  suggested_action: string;
  requires_analyst_approval: boolean;
  incident_id: string | null;
}

export interface Incident {
  incident_id: string;
  opened_at: string;
  last_seen_at: string;
  title: string;
  severity: Severity;
  lane: Lane;
  priority: number;
  entity: string;
  family: string | null;
  attack: AttackTechnique | null;
  alert_ids: string[];
  event_count: number;
  distinct_destinations: number;
  distinct_ports: number;
  status: "open" | "triaging" | "closed" | "suppressed";
  analyst_verdict: string | null;
}

export interface Stats {
  alerts_total?: number;
  incidents_total?: number;
  compression_ratio?: number;
  alerts_known_threat?: number;
  alerts_hunting?: number;
  alerts_review?: number;
  pending_verdicts?: number;
  [key: string]: number | undefined;
}

export interface Session {
  token: string;
  role: string;
  segments: string[];
}

const TOKEN_KEY = "penumbra.session";

export function loadSession(): Session | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(TOKEN_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session | null): void {
  if (typeof window === "undefined") return;
  try {
    if (session) window.localStorage.setItem(TOKEN_KEY, JSON.stringify(session));
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private browsing; the session simply does not persist */
  }
}

async function request<T>(path: string, token: string | null, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }

  /** 403 is a permission boundary doing its job, not a bug. The UI says so. */
  get isPermissionDenied(): boolean {
    return this.status === 403;
  }
}

export async function login(username: string, password: string): Promise<Session> {
  const body = await request<{ access_token: string; role: string; segments: string[] }>(
    "/auth/login",
    null,
    { method: "POST", body: JSON.stringify({ username, password }) },
  );
  return { token: body.access_token, role: body.role, segments: body.segments };
}

export const getAlerts = (token: string, lane?: Lane, limit = 200) =>
  request<Alert[]>(`/alerts?limit=${limit}${lane ? `&lane=${lane}` : ""}`, token);

export const getIncidents = (token: string, limit = 100) =>
  request<Incident[]>(`/incidents?limit=${limit}`, token);

export const getIncident = (token: string, id: string) =>
  request<{ incident: Incident; alerts: Alert[] }>(`/incidents/${id}`, token);

export const getStats = (token: string) => request<Stats>("/stats", token);

export const recordVerdict = (token: string, incidentId: string, verdict: string, note = "") =>
  request<{ queued_for_training: boolean; promoted: boolean }>(
    `/incidents/${incidentId}/verdict`,
    token,
    { method: "POST", body: JSON.stringify({ verdict, note }) },
  );

export const promoteVerdicts = (token: string, incidentIds: string[]) =>
  request<{ promoted: number }>("/feedback/promote", token, {
    method: "POST",
    body: JSON.stringify({ incident_ids: incidentIds }),
  });

export const getAudit = (token: string, limit = 50) =>
  request<{ chain_intact: boolean; n_entries: number; entries: AuditEntry[] }>(
    `/audit?limit=${limit}`,
    token,
  );

export const getRbac = (token: string) =>
  request<{ matrix: string; roles: Record<string, string[]> }>("/governance/rbac", token);

export interface AuditEntry {
  sequence: number;
  timestamp: string;
  actor: string;
  role: string;
  action: string;
  target: string;
  detail: Record<string, unknown>;
  entry_hash: string;
}

/** Live alert stream. The token goes in the query string because a browser cannot set headers on a
 *  WebSocket handshake; the server still verifies it and refuses without one. */
export function openStream(token: string, onMessage: (msg: StreamMessage) => void): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/api/stream?token=${token}`);
  ws.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data) as StreamMessage);
    } catch {
      /* malformed frame; ignore rather than tearing down the stream */
    }
  };
  return ws;
}

export type StreamMessage =
  | { type: "connected"; user: string }
  | { type: "heartbeat" }
  | { type: "alert"; alert: Alert; at: string }
  | { type: "verdict"; incident_id: string; verdict: string };

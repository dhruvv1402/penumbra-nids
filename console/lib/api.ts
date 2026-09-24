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
  suppression_rule_id?: string | null;
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
  alerts_suppressed?: number;
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

export interface TriageNote {
  headline: string;
  what_fired: string[];
  what_this_might_be: string;
  what_to_check: string[];
  what_not_to_conclude: string[];
  citations: { technique_id: string; name: string; url: string; quoted: string }[];
  generated_by: string;
}

/** Offline BM25 copilot. Deterministic, every source cited; a novelty note names no technique. */
export const getTriage = (token: string, alertId: string) =>
  request<{ alert_id: string; note: TriageNote; text: string }>(`/alerts/${alertId}/triage`, token);

export const recordVerdict = (token: string, incidentId: string, verdict: string, note = "") =>
  request<{ queued_for_training: boolean; promoted: boolean }>(
    `/incidents/${incidentId}/verdict`,
    token,
    { method: "POST", body: JSON.stringify({ verdict, note }) },
  );

/** Verdict on a single alert. NSL-KDD and UNSW carry no IPs, so their alerts never correlate into
 *  incidents - without this route every verdict button on them was a dead end. */
export const recordAlertVerdict = (token: string, alertId: string, verdict: string, note = "") =>
  request<{
    queued_for_training: boolean;
    promoted: boolean;
    proposed_suppression: Record<string, string> | null;
  }>(`/alerts/${alertId}/verdict`, token, { method: "POST", body: JSON.stringify({ verdict, note }) });

/** `expected` pins the verdict the approver saw: if it changed since, nothing is promoted. */
export const promoteVerdicts = (token: string, ids: string[], expected?: Record<string, string>) =>
  request<{ promoted: number; refused_self_approval: string[] }>("/feedback/promote", token, {
    method: "POST",
    body: JSON.stringify({ incident_ids: ids, expected }),
  });

export interface PendingVerdict {
  target_id: string;
  target_kind: "alert" | "incident";
  verdict: string;
  actor: string;
  note: string | null;
  recorded_at: string;
  p_attack: number | null;
  family: string | null;
  flags: string[];
  flag_reasons: string[];
  self_approval: boolean;
}

export const getPending = (token: string) => request<PendingVerdict[]>("/feedback/pending", token);

export interface QueueItem {
  alert_id: string;
  lane: Lane;
  verdict: Verdict;
  p_attack: number;
  novelty_percentile: number;
  margin: number;
  why: string;
}

export const getLabellingQueue = (token: string, limit = 25) =>
  request<{ strategy: string; items: QueueItem[] }>(`/feedback/queue?limit=${limit}`, token);

export interface Suppression {
  rule_id: string;
  match: Record<string, string>;
  reason: string;
  created_by: string;
  created_at: string;
  expires_at: string;
  source_alert_id: string | null;
  active: boolean;
}

export const getSuppressions = (token: string) => request<Suppression[]>("/suppressions", token);

export const createSuppression = (
  token: string,
  body: { match: Record<string, string>; reason: string; days: number; source_alert_id?: string },
) => request<Suppression>("/suppressions", token, { method: "POST", body: JSON.stringify(body) });

export const getAudit = (token: string, limit = 50) =>
  request<{ chain_intact: boolean; n_entries: number; entries: AuditEntry[] }>(
    `/audit?limit=${limit}`,
    token,
  );

export interface ModelVersion {
  version: string;
  created_at: string;
  created_by: string;
  parent: string | null;
  feedback_rows: number | null;
  approvers: string[];
  gate: { passed: boolean; reasons: string[] } | null;
  shadow: { alert_volume_ratio: number | null; cohen_kappa: number | null; agreement: number | null; n_rows: number } | null;
  intact: boolean;
  champion: boolean;
}

export interface ModelRegistry {
  dataset: string;
  champion: string | null;
  history: { action: string; version: string; previous: string | null; by: string; at: string }[];
  versions: ModelVersion[];
}

export const getModels = (token: string, dataset = "nslkdd") =>
  request<ModelRegistry>(`/models/${dataset}`, token);

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

/**
 * Evaluation reports.
 *
 * Every number the evaluation page renders comes from one of these, and each one was written by a
 * CLI command whose name travels with it. Nothing on that page is hardcoded, so the console cannot
 * quietly disagree with the measurements.
 */
export interface ReportEntry {
  name: string;
  title: string;
  description: string;
  /** The command that produces this file. Shown in the UI so a number can always be traced. */
  command: string;
  available: boolean;
  generated_at: number | null;
  bytes: number;
}

export interface ReportPayload<T = unknown> {
  name: string;
  title: string;
  description: string;
  command: string;
  generated_at: number;
  data: T;
}

export const getReports = (token: string) =>
  request<{ reports: ReportEntry[] }>("/reports", token);

export const getReport = <T = unknown>(token: string, name: string) =>
  request<ReportPayload<T>>(`/reports/${name}`, token);

/** Shapes of the reports the evaluation page reads. Partial on purpose - the page renders what is
 *  present and says so when something is not, rather than throwing on an older report file. */
export interface SequenceReport {
  fpr_budget: number;
  n_train: number;
  n_test: number;
  window_length: number;
  arms: {
    name: string;
    roc_auc: number;
    recall_at_budget: number;
    realised_fpr: number;
    n_alerts: number;
    notes: string;
  }[];
  training?: { epochs_run: number; best_val_auc: number; collapsed_epochs?: number };
  per_family: Record<string, Record<string, number>>;
}

export interface MinedRulesReport {
  quarantined: string[];
  n_holdout: number;
  n_test: number;
  with_artifacts: RuleArm;
  without_artifacts: RuleArm;
}

export interface RuleArm {
  n_candidates: number;
  n_artifact_excluded: number;
  n_survivors: number;
  min_precision: number;
  holdout_coverage: { recall: number; precision: number; n_matched: number };
  test_coverage: { recall: number; precision: number; n_matched: number };
}

export interface ConformalControlEntry {
  condition: string;
  score_psi: number;
  coverage: number;
  gap: number;
  abstention: number;
  note: string;
}

export interface ConformalDriftReport {
  change_point_window: number;
  first_breach_window: number;
  points: {
    window: number;
    psi: number;
    coverage: number;
    nominal: number;
    gap: number;
    abstention_rate: number;
  }[];
}

export interface FeatureDriftReport {
  dataset: string;
  inject: string | null;
  n_reference: number;
  n_current: number;
  n_significant: number;
  n_moderate: number;
  n_ks_flagged: number;
  expected_false_flags: number;
  features: {
    feature: string;
    psi: number;
    verdict: string;
    ks_statistic: number;
    ks_pvalue: number;
    js_distance: number;
    ks_flagged: boolean;
  }[];
}

export interface EvasionReport {
  n_attacks: number;
  threshold: number;
  results: {
    name: string;
    constrained: boolean;
    points: {
      effort: number;
      detection_rate: number;
      n_evaded: number;
      mean_duration_multiple: number;
    }[];
  }[];
  per_family: Record<string, { baseline: number; attacked: number; delta: number; n: number }>;
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
  | { type: "verdict"; incident_id?: string; alert_id?: string; verdict: string }
  | { type: "suppression"; rule_id: string }
  | { type: "incidents"; count: number };

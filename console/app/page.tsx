"use client";

/**
 * The SOC console.
 *
 * Two lanes, side by side, because they are different jobs (ADR-0002):
 *
 *   KNOWN THREAT   high-precision supervised detections. SLA'd. Worked top to bottom.
 *   HUNTING        novelty findings, ranked, capped at a daily budget. Worked until the budget is
 *                  exhausted and then stopped - an unworked item at position 400 is the design
 *                  functioning, not an alert that was missed.
 *
 * The triage panel has Escalate and Dismiss and Benign-by-policy. It has no Block button, because
 * the API has no endpoint behind one.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  type Alert,
  type Incident,
  type Lane,
  type Session,
  type Stats,
  type TriageNote,
  getAlerts,
  getIncident,
  getIncidents,
  createSuppression,
  getStats,
  getTriage,
  loadSession,
  login,
  openStream,
  recordAlertVerdict,
  recordVerdict,
  saveSession,
} from "@/lib/api";
import {
  AttackChip,
  ContributionList,
  Empty,
  Panel,
  ScoreTriplet,
  SeverityDot,
  Stat,
  VerdictBadge,
} from "@/components/Primitives";

const HUNTING_BUDGET = 50;

export default function Console() {
  const [session, setSession] = useState<Session | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [stats, setStats] = useState<Stats>({});
  const [selected, setSelected] = useState<Alert | null>(null);
  const [lane, setLane] = useState<Lane | "incidents">("known_threat");
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [live, setLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => setSession(loadSession()), []);

  const refresh = useCallback(async (token: string) => {
    try {
      // One request per lane. A single top-N by priority starves the review lane: abstentions sit
      // near p = 0.5, below every known-threat alert, so "review 0" was showing over 1,046 items.
      const [known, hunting, review, s, inc] = await Promise.all([
        getAlerts(token, "known_threat", 300),
        getAlerts(token, "hunting", 300),
        getAlerts(token, "review", 300),
        getStats(token),
        getIncidents(token, 200),
      ]);
      setAlerts([...known, ...hunting, ...review]);
      // Largest first within priority: two incidents at equal priority are not equally urgent when
      // one is 41,866 flows and the other is one.
      setIncidents([...inc].sort((a, b) => b.priority - a.priority || b.event_count - a.event_count));
      setStats(s);
      setError(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        saveSession(null);
        setSession(null);
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    }
  }, []);

  useEffect(() => {
    if (!session) return;
    void refresh(session.token);

    const ws = openStream(session.token, (msg) => {
      if (msg.type === "connected") setLive(true);
      if (msg.type === "incidents") void refresh(session.token);
      if (msg.type === "alert") {
        // Prepend and cap, so a long demo does not grow the DOM without bound.
        setAlerts((prev) => [msg.alert, ...prev].slice(0, 1200));
        setStats((prev) => ({ ...prev, alerts_total: (prev.alerts_total ?? 0) + 1 }));
      }
    });
    ws.onclose = () => setLive(false);
    socket.current = ws;
    return () => ws.close();
  }, [session, refresh]);

  const lanes = useMemo(() => {
    const known = alerts
      .filter((a) => a.lane === "known_threat" && a.verdict !== "BENIGN" && a.verdict !== "BENIGN_BY_POLICY")
      .sort((a, b) => b.priority - a.priority);
    const hunting = alerts
      .filter((a) => a.lane === "hunting")
      .sort((a, b) => b.novelty_percentile - a.novelty_percentile);
    const review = alerts.filter((a) => a.lane === "review");
    return { known, hunting, review };
  }, [alerts]);

  const visible = lane === "known_threat" ? lanes.known : lane === "hunting" ? lanes.hunting : lane === "review" ? lanes.review : [];
  const overflow = lane === "hunting" ? Math.max(0, lanes.hunting.length - HUNTING_BUDGET) : 0;
  const shown = lane === "hunting" ? visible.slice(0, HUNTING_BUDGET) : visible;

  if (!session) return <LoginScreen onSession={(s) => { saveSession(s); setSession(s); }} />;

  return (
    <main className="h-screen flex flex-col">
      <Header session={session} stats={stats} live={live} onLogout={() => { saveSession(null); setSession(null); }} />

      {error && (
        <div className="px-3 py-1.5 bg-[color-mix(in_srgb,var(--color-sev-high)_15%,transparent)] border-b border-[var(--color-border)] text-[11px] text-[var(--color-sev-high)]">
          {error}
        </div>
      )}

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] gap-2 p-2 min-h-0">
        <Panel
          title="queue"
          right={
            <div className="flex gap-1">
              {(["incidents", "known_threat", "hunting", "review"] as const).map((l) => (
                <button
                  key={l}
                  onClick={() => setLane(l)}
                  className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                    lane === l
                      ? "border-[var(--color-ink-dim)] text-[var(--color-ink)]"
                      : "border-[var(--color-border)] text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)]"
                  }`}
                >
                  {l === "incidents"
                    ? `incidents ${incidents.length}`
                    : l === "known_threat"
                      ? `known ${lanes.known.length}`
                      : l === "hunting"
                        ? `hunting ${lanes.hunting.length}`
                        : `review ${lanes.review.length}`}
                </button>
              ))}
            </div>
          }
        >
          {lane === "incidents" ? (
            incidents.length === 0 ? (
              <Empty>
                No incidents. Correlation needs source IPs, which only CICIDS2017 has. Build a fixture
                with{" "}
                <code className="text-[var(--color-ink-dim)]">penumbra correlate --write-fixture incidents.json</code>{" "}
                then{" "}
                <code className="text-[var(--color-ink-dim)]">penumbra replay --from-fixture incidents.json --ingest</code>
              </Empty>
            ) : (
              <ul className="divide-y divide-[var(--color-border)]">
                {incidents.map((inc) => (
                  <li
                    key={inc.incident_id}
                    onClick={() => {
                      setSelectedIncident(inc);
                      setSelected(null);
                    }}
                    className={`px-3 py-2 cursor-pointer transition-colors ${
                      selectedIncident?.incident_id === inc.incident_id ? "bg-[var(--color-panel-2)]" : "hover:bg-[var(--color-panel-2)]/60"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <SeverityDot severity={inc.severity} />
                      <span className="text-[11px] text-[var(--color-ink)] truncate">{inc.title}</span>
                      <span className="ml-auto text-[11px] tabular-nums text-[var(--color-ink-faint)]">
                        {inc.event_count.toLocaleString()}
                      </span>
                    </div>
                    <div className="mt-1 text-[10px] text-[var(--color-ink-faint)]">
                      {inc.distinct_destinations.toLocaleString()} destinations · {inc.distinct_ports.toLocaleString()} ports ·{" "}
                      {inc.status}
                      {inc.analyst_verdict ? ` · ${inc.analyst_verdict.replaceAll("_", " ")}` : ""}
                    </div>
                  </li>
                ))}
              </ul>
            )
          ) : shown.length === 0 ? (
            <Empty>
              Nothing in this lane. Start the replay with{" "}
              <code className="text-[var(--color-ink-dim)]">penumbra replay</code> to stream alerts in.
            </Empty>
          ) : (
            <>
              {lane === "hunting" && (
                <p className="px-3 py-2 text-[10px] text-[var(--color-ink-faint)] border-b border-[var(--color-border)] leading-snug">
                  Ranked by novelty, capped at {HUNTING_BUDGET}/day. A fixed budget cannot cause alert
                  fatigue by construction — its metric is Precision@k, not FPR.
                  {overflow > 0 && ` ${overflow} below the cut are recorded but not queued.`}
                </p>
              )}
              <ul className="divide-y divide-[var(--color-border)]">
                {shown.map((a) => (
                  <AlertRow
                    key={a.alert_id}
                    alert={a}
                    active={selected?.alert_id === a.alert_id}
                    onSelect={() => {
                      setSelected(a);
                      setSelectedIncident(null);
                    }}
                  />
                ))}
              </ul>
            </>
          )}
        </Panel>

        {selectedIncident && !selected ? (
          <IncidentDetail
            incident={selectedIncident}
            token={session.token}
            role={session.role}
            onOpenAlert={(a) => setSelected(a)}
          />
        ) : (
          <Detail alert={selected} token={session.token} role={session.role} />
        )}
      </div>
    </main>
  );
}

function Header({
  session,
  stats,
  live,
  onLogout,
}: {
  session: Session;
  stats: Stats;
  live: boolean;
  onLogout: () => void;
}) {
  const compression = stats.compression_ratio ?? 0;
  return (
    <header className="flex items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-panel)] shrink-0">
      <div className="flex items-center gap-3 px-3 py-2">
        <span className="text-[15px] tracking-[0.2em] text-[var(--color-ink)]">PENUMBRA</span>
        <span className="text-[10px] text-[var(--color-ink-faint)] hidden md:inline">
          alerts a SOC · never blocks traffic
        </span>
        <nav className="flex items-center gap-3 ml-3">
          <Link
            href="/evaluation"
            className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
          >
            evaluation
          </Link>
          <Link
            href="/drift"
            className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
          >
            drift
          </Link>
          <Link
            href="/feedback"
            className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
          >
            feedback
          </Link>
          <Link
            href="/governance"
            className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
          >
            governance
          </Link>
        </nav>
      </div>

      <div className="flex items-stretch">
        <Stat label="alerts" value={String(stats.alerts_total ?? 0)} />
        <Stat label="incidents" value={String(stats.incidents_total ?? 0)} />
        <Stat
          label="events / incident"
          value={compression ? `${compression}x` : "—"}
          tone="var(--color-ok)"
        />
        <Stat label="pending verdicts" value={String(stats.pending_verdicts ?? 0)} />
        <Stat label="suppressed" value={String(stats.alerts_suppressed ?? 0)} />
        <div className="px-3 py-1.5 border-l border-[var(--color-border)] flex items-center gap-2">
          <span
            className={`w-1.5 h-1.5 rounded-full ${live ? "live-dot" : ""}`}
            style={{ background: live ? "var(--color-ok)" : "var(--color-ink-faint)" }}
          />
          <span className="text-[10px] text-[var(--color-ink-dim)]">{live ? "LIVE" : "offline"}</span>
        </div>
        <button
          onClick={onLogout}
          className="px-3 border-l border-[var(--color-border)] text-[10px] text-[var(--color-ink-faint)] hover:text-[var(--color-ink)]"
          title={`${session.role}${session.segments.length ? ` · ${session.segments.join(", ")}` : " · all segments"}`}
        >
          {session.role} ▾
        </button>
      </div>
    </header>
  );
}

function AlertRow({
  alert,
  active,
  onSelect,
}: {
  alert: Alert;
  active: boolean;
  onSelect: () => void;
}) {
  const net = alert.network;
  return (
    <li
      onClick={onSelect}
      className={`slide-in px-3 py-2 cursor-pointer transition-colors ${
        active ? "bg-[var(--color-panel-2)]" : "hover:bg-[var(--color-panel-2)]/60"
      }`}
    >
      <div className="flex items-center gap-2">
        <SeverityDot severity={alert.severity} />
        <VerdictBadge verdict={alert.verdict} />
        {alert.family && <span className="text-[11px] text-[var(--color-ink-dim)]">{alert.family}</span>}
        <span className="ml-auto text-[11px] tabular-nums text-[var(--color-ink-faint)]">
          {alert.priority}
        </span>
      </div>
      <div className="mt-1 text-[11px] text-[var(--color-ink-dim)] truncate">
        {net.src_ip ?? "—"}
        {net.dst_ip ? ` → ${net.dst_ip}` : ""}
        {net.dst_port ? `:${net.dst_port}` : ""}
        {net.protocol ? ` ${net.protocol}` : ""}
      </div>
    </li>
  );
}

function Detail({ alert, token, role }: { alert: Alert | null; token: string; role: string }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [proposal, setProposal] = useState<Record<string, string> | null>(null);
  const [triage, setTriage] = useState<TriageNote | null>(null);
  const [triageError, setTriageError] = useState<string | null>(null);

  // A new selection starts clean; a note about the previous alert is a note about the wrong alert.
  const alertId = alert?.alert_id;
  useEffect(() => {
    setNote(null);
    setProposal(null);
    setTriage(null);
    setTriageError(null);
    if (!alertId) return;
    let live = true;
    getTriage(token, alertId)
      .then((r) => live && setTriage(r.note))
      .catch((err) =>
        live && setTriageError(err instanceof ApiError && err.status === 503 ? "Copilot corpus not built: penumbra copilot build" : null),
      );
    return () => {
      live = false;
    };
  }, [alertId, token]);

  if (!alert) {
    return (
      <Panel title="detail">
        <Empty>Select an alert to see why it fired.</Empty>
      </Panel>
    );
  }

  const canSuppress = role === "senior" || role === "admin";

  const submit = async (verdict: string) => {
    setBusy(true);
    setProposal(null);
    try {
      // Correlated alerts are judged as part of their incident; uncorrelated ones on their own.
      // Both land in the same queue behind the same senior-only, two-person promotion gate.
      if (alert.incident_id) {
        await recordVerdict(token, alert.incident_id, verdict);
      } else {
        const res = await recordAlertVerdict(token, alert.alert_id, verdict);
        if (res.proposed_suppression) setProposal(res.proposed_suppression);
      }
      setNote(
        verdict === "benign_by_policy"
          ? canSuppress
            ? "Recorded. Benign-by-policy never trains the model - it becomes an expiring suppression rule below."
            : "Recorded. Benign-by-policy never trains the model; a senior analyst turns it into an expiring suppression rule."
          : "Verdict recorded and queued. A different senior analyst must promote it before it can train the model.",
      );
    } catch (err) {
      setNote(
        err instanceof ApiError && err.isPermissionDenied
          ? `Your role (${role}) cannot do that.`
          : err instanceof ApiError && err.status === 429
            ? "Verdict rate limit reached. Bulk relabelling is throttled and logged."
            : err instanceof Error
              ? err.message
              : String(err),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel
      title="detail"
      right={<span className="text-[10px] text-[var(--color-ink-faint)]">{alert.alert_id.slice(0, 8)}</span>}
    >
      <div className="p-3 space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <VerdictBadge verdict={alert.verdict} />
          <AttackChip attack={alert.attack} />
          {alert.detector_agreement > 0 && (
            <span className="text-[10px] text-[var(--color-ink-faint)]">
              {alert.detector_agreement}/3 novelty detectors agreed
            </span>
          )}
        </div>

        <ScoreTriplet
          pAttack={alert.p_attack}
          novelty={alert.novelty_percentile}
          priority={alert.priority}
        />

        <div className="bg-[var(--color-panel-2)] border border-[var(--color-border)] rounded px-3 py-2 text-[11px] text-[var(--color-ink-dim)] leading-relaxed">
          {alert.network.is_pseudonymised && (
            <span className="text-[9px] uppercase tracking-wider text-[var(--color-ink-faint)] block mb-1">
              addresses pseudonymised · re-identification is admin-only and audited
            </span>
          )}
          {alert.network.src_ip ?? "—"} → {alert.network.dst_ip ?? "—"}
          {alert.network.dst_port ? `:${alert.network.dst_port}` : ""}{" "}
          {alert.network.protocol?.toUpperCase()}
          {alert.network.src_bytes != null && (
            <span className="ml-2">
              {alert.network.src_bytes.toLocaleString()} ↑ /{" "}
              {(alert.network.dst_bytes ?? 0).toLocaleString()} ↓ bytes
            </span>
          )}
        </div>

        <div>
          <h3 className="text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)] mb-1">
            why it fired
          </h3>
          <div className="bg-[var(--color-panel-2)] border border-[var(--color-border)] rounded">
            <ContributionList contributions={alert.contributions} />
          </div>
        </div>

        {(triage || triageError) && (
          <div>
            <h3 className="text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)] mb-1">
              triage note · offline copilot, every source cited
            </h3>
            {triageError ? (
              <p className="text-[11px] text-[var(--color-ink-faint)]">{triageError}</p>
            ) : (
              triage && (
                <div className="bg-[var(--color-panel-2)] border border-[var(--color-border)] rounded px-3 py-2 space-y-2 text-[11px] leading-snug">
                  <p className="text-[var(--color-ink)]">{triage.headline}</p>
                  {triage.what_this_might_be && <p className="text-[var(--color-ink-dim)]">{triage.what_this_might_be}</p>}
                  {triage.what_to_check.length > 0 && (
                    <ol className="list-decimal ml-4 text-[var(--color-ink-dim)] space-y-0.5">
                      {triage.what_to_check.slice(0, 4).map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ol>
                  )}
                  {triage.what_not_to_conclude.length > 0 && (
                    <p className="text-[10px] text-[var(--color-sev-medium)]">
                      does not establish: {triage.what_not_to_conclude[0]}
                    </p>
                  )}
                  {triage.citations.length > 0 && (
                    <div className="flex flex-wrap gap-2 pt-1 border-t border-[var(--color-border)]">
                      {triage.citations.map((c) => (
                        <a
                          key={c.technique_id}
                          href={c.url}
                          target="_blank"
                          rel="noreferrer"
                          title={c.quoted}
                          className="text-[10px] text-[var(--color-ink-dim)] underline hover:text-[var(--color-ink)]"
                        >
                          {c.technique_id} {c.name}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        )}

        {alert.suggested_action && (
          <div className="border border-dashed border-[var(--color-border)] rounded px-3 py-2">
            <div className="text-[9px] uppercase tracking-wider text-[var(--color-sev-medium)] mb-1">
              suggested action · requires analyst approval
            </div>
            <p className="text-[11px] text-[var(--color-ink-dim)] leading-snug">
              {alert.suggested_action}
            </p>
          </div>
        )}

        <div className="flex gap-2 flex-wrap">
          <VerdictButton label="True positive" onClick={() => submit("true_positive")} disabled={busy} tone="var(--color-sev-high)" />
          <VerdictButton label="False positive" onClick={() => submit("false_positive")} disabled={busy} tone="var(--color-ok)" />
          <VerdictButton label="Benign by policy" onClick={() => submit("benign_by_policy")} disabled={busy} tone="var(--color-sev-low)" />
        </div>

        {note && <p className="text-[11px] text-[var(--color-ink-dim)] leading-snug">{note}</p>}

        {proposal && canSuppress && (
          <SuppressionForm
            token={token}
            alertId={alert.alert_id}
            proposal={proposal}
            onDone={(msg) => {
              setProposal(null);
              setNote(msg);
            }}
          />
        )}

        {alert.suppression_rule_id && (
          <p className="text-[10px] text-[var(--color-sev-low)] leading-snug">
            Suppressed by {alert.suppression_rule_id}. Stored and counted, not queued.
          </p>
        )}

        <p className="text-[10px] text-[var(--color-ink-faint)] leading-snug border-t border-[var(--color-border)] pt-2">
          There is no block action here, and no endpoint behind one. Penumbra surfaces; a human
          decides. See ADR-0001.
        </p>
      </div>
    </Panel>
  );
}

/**
 * Benign-by-policy becomes a rule, not a label. The form is pre-filled with the narrowest match the
 * alert supports; the analyst can remove fields, but the API refuses a rule scoped to a family alone,
 * a wildcard, or one that outlives 90 days.
 */
function SuppressionForm({
  token,
  alertId,
  proposal,
  onDone,
}: {
  token: string;
  alertId: string;
  proposal: Record<string, string>;
  onDone: (message: string) => void;
}) {
  const [fields, setFields] = useState<Record<string, boolean>>(
    Object.fromEntries(Object.keys(proposal).map((k) => [k, true])),
  );
  const [reason, setReason] = useState("");
  const [days, setDays] = useState(30);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const create = async () => {
    setBusy(true);
    setError(null);
    try {
      const match = Object.fromEntries(Object.entries(proposal).filter(([k]) => fields[k]));
      const rule = await createSuppression(token, { match, reason, days, source_alert_id: alertId });
      onDone(`Suppression ${rule.rule_id} active until ${rule.expires_at.slice(0, 10)}. It is in the audit log.`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border border-[var(--color-border)] rounded px-3 py-2 space-y-2">
      <div className="text-[9px] uppercase tracking-wider text-[var(--color-sev-low)]">
        create suppression · always scoped · always expires
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(proposal).map(([k, v]) => (
          <label key={k} className="flex items-center gap-1 text-[11px] text-[var(--color-ink-dim)]">
            <input
              type="checkbox"
              checked={fields[k] ?? false}
              onChange={(e) => setFields((f) => ({ ...f, [k]: e.target.checked }))}
            />
            {k}={v}
          </label>
        ))}
      </div>
      <input
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder="why this is policy, not a threat (e.g. nightly vulnerability scanner)"
        className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-[11px] text-[var(--color-ink)] focus:outline-none"
      />
      <div className="flex items-center gap-2">
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-1 py-0.5 text-[11px] text-[var(--color-ink)]"
        >
          {[7, 14, 30, 60, 90].map((d) => (
            <option key={d} value={d}>
              expires in {d} days
            </option>
          ))}
        </select>
        <button
          onClick={create}
          disabled={busy || reason.trim().length < 10}
          className="text-[11px] px-2.5 py-1 rounded border border-[var(--color-sev-low)] text-[var(--color-sev-low)] disabled:opacity-40"
        >
          Create rule
        </button>
      </div>
      {error && <p className="text-[10px] text-[var(--color-sev-high)] break-words">{error}</p>}
    </div>
  );
}

function VerdictButton({
  label,
  onClick,
  disabled,
  tone,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  tone: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="text-[11px] px-2.5 py-1 rounded border transition-colors disabled:opacity-40"
      style={{ borderColor: tone, color: tone }}
    >
      {label}
    </button>
  );
}

function LoginScreen({ onSession }: { onSession: (s: Session) => void }) {
  const [username, setUsername] = useState("analyst");
  const [password, setPassword] = useState("analyst");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      onSession(await login(username, password));
    } catch {
      setError("Invalid credentials, or the API is not running on :8000.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="h-screen grid place-items-center">
      <form
        onSubmit={submit}
        className="w-[320px] bg-[var(--color-panel)] border border-[var(--color-border)] rounded p-5 space-y-3"
      >
        <div>
          <h1 className="text-[18px] tracking-[0.2em]">PENUMBRA</h1>
          <p className="text-[10px] text-[var(--color-ink-faint)] mt-1">SOC console</p>
        </div>
        <Field label="user" value={username} onChange={setUsername} />
        <Field label="password" value={password} onChange={setPassword} type="password" />
        {error && <p className="text-[11px] text-[var(--color-sev-high)]">{error}</p>}
        <button
          type="submit"
          disabled={busy}
          className="w-full text-[12px] py-1.5 rounded border border-[var(--color-ink-dim)] text-[var(--color-ink)] hover:bg-[var(--color-panel-2)] disabled:opacity-40"
        >
          {busy ? "…" : "Sign in"}
        </button>
        <p className="text-[9px] text-[var(--color-ink-faint)] leading-snug">
          Demo users (analyst / senior / admin) exist only when the API runs with
          PENUMBRA_ALLOW_DEMO_USERS set. Roles differ: an analyst sees only its own segments and
          cannot promote a verdict into training.
        </p>
      </form>
    </main>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
}) {
  return (
    <label className="block">
      <span className="text-[9px] uppercase tracking-wider text-[var(--color-ink-faint)]">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full mt-0.5 bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-[12px] text-[var(--color-ink)] focus:outline-none focus:border-[var(--color-ink-dim)]"
      />
    </label>
  );
}

/**
 * One incident: the unit an analyst actually works. The event count is the whole point - a 41,866-
 * flow scan is one row here, not 41,866 tickets. Its verdict applies to every alert in it.
 */
function IncidentDetail({
  incident,
  token,
  role,
  onOpenAlert,
}: {
  incident: Incident;
  token: string;
  role: string;
  onOpenAlert: (a: Alert) => void;
}) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setNote(null);
    getIncident(token, incident.incident_id)
      .then((r) => live && setAlerts(r.alerts))
      .catch(() => live && setAlerts([]));
    return () => {
      live = false;
    };
  }, [incident.incident_id, token]);

  const submit = async (verdict: string) => {
    try {
      await recordVerdict(token, incident.incident_id, verdict);
      setNote("Verdict recorded on the whole incident and queued. A different senior must promote it.");
    } catch (err) {
      setNote(
        err instanceof ApiError && err.isPermissionDenied
          ? `Your role (${role}) cannot do that.`
          : err instanceof ApiError && err.status === 409
            ? "This incident's verdict is already promoted into training; it cannot be changed."
            : err instanceof Error
              ? err.message
              : String(err),
      );
    }
  };

  return (
    <Panel title="incident" right={<span className="text-[10px] text-[var(--color-ink-faint)]">{incident.incident_id.slice(0, 8)}</span>}>
      <div className="p-3 space-y-3">
        <p className="text-[12px] text-[var(--color-ink)]">{incident.title}</p>
        <div className="grid grid-cols-3 border border-[var(--color-border)] rounded">
          <Stat label="events" value={incident.event_count.toLocaleString()} />
          <Stat label="destinations" value={incident.distinct_destinations.toLocaleString()} />
          <Stat label="ports" value={incident.distinct_ports.toLocaleString()} />
        </div>
        <p className="text-[11px] text-[var(--color-ink-dim)] leading-snug">
          {incident.family
            ? `The model names this ${incident.family}. That is its closest trained family, not an identification; the fan-out above is measured.`
            : "The model could not name a family. The grouping is by source and time window; the fan-out above is measured."}{" "}
          {new Date(incident.opened_at).toLocaleString()} → {new Date(incident.last_seen_at).toLocaleString()}
        </p>
        <div className="flex gap-2 flex-wrap">
          <VerdictButton label="True positive" onClick={() => submit("true_positive")} disabled={false} tone="var(--color-sev-high)" />
          <VerdictButton label="False positive" onClick={() => submit("false_positive")} disabled={false} tone="var(--color-ok)" />
          <VerdictButton label="Benign by policy" onClick={() => submit("benign_by_policy")} disabled={false} tone="var(--color-sev-low)" />
        </div>
        {note && <p className="text-[11px] text-[var(--color-ink-dim)]">{note}</p>}
        <div>
          <h3 className="text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)] mb-1">
            sample of its alerts ({alerts.length} of {incident.event_count.toLocaleString()})
          </h3>
          <ul className="divide-y divide-[var(--color-border)] border border-[var(--color-border)] rounded">
            {alerts.map((a) => (
              <AlertRow key={a.alert_id} alert={a} active={false} onSelect={() => onOpenAlert(a)} />
            ))}
          </ul>
        </div>
      </div>
    </Panel>
  );
}

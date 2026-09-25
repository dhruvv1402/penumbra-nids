"use client";

/**
 * The feedback page: where an analyst verdict becomes (or does not become) training data.
 *
 * The human-in-the-loop is also a poisoning vector we built ourselves (THREAT_MODEL T1). A stolen
 * analyst session that marks its own traffic `false_positive` teaches the next model to ignore it.
 * This page is where that is stopped, and each control is visible rather than described:
 *
 *   PROMOTION     senior only, and never your own verdict (two-person rule, enforced in SQL).
 *   FLAGS         each pending verdict carries the reasons it deserves a second look.
 *   LABEL NEXT    uncertainty sampling - the alerts whose labels the model cannot produce itself.
 *   SUPPRESSIONS  benign-by-policy lives here, with an owner and an expiry, never in the weights.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  type PendingVerdict,
  type QueueItem,
  type Session,
  type Suppression,
  getLabellingQueue,
  getPending,
  getReport,
  getSuppressions,
  loadSession,
  promoteVerdicts,
  revokeSuppression,
  saveSession,
} from "@/lib/api";
import { Empty, Panel, Stat } from "@/components/Primitives";

export default function Feedback() {
  const [session, setSession] = useState<Session | null>(null);
  const [pending, setPending] = useState<PendingVerdict[] | null>(null);
  const [pendingDenied, setPendingDenied] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [rules, setRules] = useState<Suppression[]>([]);
  const [drill, setDrill] = useState<PoisoningReport | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setSession(loadSession()), []);

  const load = useCallback(async (token: string) => {
    try {
      const [q, r] = await Promise.all([getLabellingQueue(token, 20), getSuppressions(token)]);
      setQueue(q.items);
      setRules(r);
      try {
        setDrill((await getReport<PoisoningReport>(token, "poisoning")).data);
      } catch {
        setDrill(null);
      }
      setError(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        saveSession(null);
        setSession(null);
        return;
      }
      setError(err instanceof Error ? err.message : String(err));
    }
    try {
      setPending(await getPending(token));
      setPendingDenied(false);
    } catch (err) {
      if (err instanceof ApiError && err.isPermissionDenied) setPendingDenied(true);
      else setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    if (session) void load(session.token);
  }, [session, load]);

  const promote = async (ids: string[]) => {
    if (!session) return;
    try {
      const seen = Object.fromEntries((pending ?? []).filter((p) => ids.includes(p.target_id)).map((p) => [p.target_id, p.verdict]));
      const res = await promoteVerdicts(session.token, ids, seen);
      const stale = ids.length - res.promoted - res.refused_self_approval.length;
      setNote(
        res.refused_self_approval.length
          ? `${res.promoted} promoted. ${res.refused_self_approval.length} refused: you recorded them, so someone else must approve them.`
          : stale > 0
            ? `${stale} not promoted: the verdict changed after this page loaded. Review it again.`
            : `${res.promoted} promoted into the training pool. Recorded in the audit log.`,
      );
      await load(session.token);
    } catch (err) {
      setNote(err instanceof Error ? err.message : String(err));
    }
  };

  const revoke = async (ruleId: string) => {
    if (!session) return;
    try {
      await revokeSuppression(session.token, ruleId);
      setNote(`Suppression ${ruleId} ended now. It stays on record; the revocation is in the audit log.`);
      await load(session.token);
    } catch (err) {
      setNote(err instanceof ApiError && err.isPermissionDenied ? "Only a senior can end a suppression." : String(err));
    }
  };

  if (!session) {
    return (
      <main className="h-screen grid place-items-center">
        <p className="text-[12px] text-[var(--color-ink-dim)]">
          <Link href="/" className="underline">
            Sign in on the queue page
          </Link>{" "}
          to view feedback.
        </p>
      </main>
    );
  }

  const flagged = (pending ?? []).filter((p) => p.flags.length > 0).length;

  return (
    <main className="min-h-screen flex flex-col">
      <header className="flex items-center gap-4 px-3 py-2 border-b border-[var(--color-border)] shrink-0">
        <Link
          href="/"
          className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
        >
          ← queue
        </Link>
        <Link
          href="/governance"
          className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
        >
          governance
        </Link>
        <h1 className="text-[11px] tracking-[0.14em] uppercase">feedback</h1>
        <p className="text-[11px] text-[var(--color-ink-dim)] ml-auto">signed in as {session.role}</p>
      </header>

      {(error || note) && (
        <div
          className={`px-3 py-1.5 border-b border-[var(--color-border)] text-[11px] ${
            error ? "text-[var(--color-sev-high)]" : "text-[var(--color-ink-dim)]"
          }`}
        >
          {error ?? note}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-2 p-2">
        <Panel
          title="promotion queue — verdicts waiting to become training data"
          right={
            <div className="flex">
              <Stat label="pending" value={String(pending?.length ?? 0)} />
              <Stat label="flagged" value={String(flagged)} tone={flagged ? "var(--color-sev-medium)" : undefined} />
            </div>
          }
        >
          {pendingDenied ? (
            <Empty>
              Your role ({session.role}) cannot promote verdicts. That is the control: an analyst account
              alone cannot teach the model anything.
            </Empty>
          ) : !pending || pending.length === 0 ? (
            <Empty>No verdicts waiting. Record one from the queue page.</Empty>
          ) : (
            <ul className="divide-y divide-[var(--color-border)]">
              {pending.map((v) => (
                <li key={v.target_id} className="px-3 py-2 text-[11px] space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[var(--color-ink)]">{v.verdict.replaceAll("_", " ")}</span>
                    <span className="text-[var(--color-ink-faint)]">
                      {v.target_kind} {v.target_id.slice(0, 8)} · by {v.actor}
                      {v.family ? ` · ${v.family}` : ""}
                      {v.p_attack != null ? ` · model said ${v.p_attack.toFixed(2)}` : ""}
                    </span>
                    <button
                      onClick={() => promote([v.target_id])}
                      disabled={v.self_approval || v.verdict === "benign_by_policy"}
                      title={
                        v.self_approval
                          ? "You recorded this verdict; a different senior must promote it."
                          : v.verdict === "benign_by_policy"
                            ? "Benign-by-policy is a suppression rule, never a training label."
                            : "Promote into the training pool"
                      }
                      className="ml-auto text-[10px] px-2 py-0.5 rounded border border-[var(--color-border)] text-[var(--color-ink-dim)] hover:text-[var(--color-ink)] disabled:opacity-30"
                    >
                      {v.self_approval ? "yours" : "promote"}
                    </button>
                  </div>
                  {v.flag_reasons.map((r) => (
                    <div key={r} className="text-[10px] text-[var(--color-sev-medium)]">
                      ⚑ {r}
                    </div>
                  ))}
                </li>
              ))}
            </ul>
          )}
          <Caption>
            Two-person rule: the approver can never be the analyst who recorded the verdict. It is
            enforced in the repository&apos;s UPDATE statement, so no route can skip it. Flags never
            drop a verdict — a silent filter would be a second, unaudited decision-maker. They tell
            the approver where to look.
          </Caption>
        </Panel>

        <Panel title="label next — uncertainty sampling" right={<Stat label="shown" value={String(queue.length)} />}>
          {queue.length === 0 ? (
            <Empty>Nothing to label. Stream alerts in with penumbra replay.</Empty>
          ) : (
            <ul className="divide-y divide-[var(--color-border)]">
              {queue.map((q) => (
                <li key={q.alert_id} className="px-3 py-1.5 text-[11px] flex items-center gap-2">
                  <span className="tabular-nums text-[var(--color-ink)] w-10">{q.p_attack.toFixed(2)}</span>
                  <span className="text-[var(--color-ink-faint)] w-16">{q.lane}</span>
                  <span className="text-[var(--color-ink-dim)] truncate">{q.why}</span>
                </li>
              ))}
            </ul>
          )}
          <Caption>
            An analyst&apos;s hour is the scarcest thing in a SOC. Confirming detections the model is
            already sure of teaches it nothing; the label worth having is the one it could not
            produce itself. Abstentions first, then smallest margin from the decision boundary.
          </Caption>
        </Panel>

        <Panel title="suppression rules — benign by policy" right={<Stat label="active" value={String(rules.filter((r) => r.active).length)} />}>
          {rules.length === 0 ? (
            <Empty>No suppression rules. A senior creates one from a benign-by-policy verdict.</Empty>
          ) : (
            <table className="w-full text-[11px]">
              <thead className="text-[var(--color-ink-dim)]">
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left font-normal px-3 py-1.5">matches</th>
                  <th className="text-left font-normal px-3 py-1.5">reason</th>
                  <th className="text-left font-normal px-3 py-1.5">owner</th>
                  <th className="text-left font-normal px-3 py-1.5">expires</th>
                  <th className="px-3 py-1.5" />
                </tr>
              </thead>
              <tbody>
                {rules.map((r) => (
                  <tr key={r.rule_id} className={`border-b border-[var(--color-border)] ${r.active ? "" : "opacity-40"}`}>
                    <td className="px-3 py-1.5 font-mono text-[10px]">
                      {Object.entries(r.match)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(" ∧ ")}
                    </td>
                    <td className="px-3 py-1.5 text-[var(--color-ink-dim)]">{r.reason}</td>
                    <td className="px-3 py-1.5">{r.created_by}</td>
                    <td className="px-3 py-1.5 tabular-nums">{r.expires_at.slice(0, 10)}</td>
                    <td className="px-3 py-1.5 text-right">
                      {r.active && (
                        <button
                          onClick={() => revoke(r.rule_id)}
                          className="text-[10px] px-2 py-0.5 rounded border border-[var(--color-border)] text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
                        >
                          end now
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <Caption>
            Every rule is scoped narrower than a family, expires within 90 days, and names its owner.
            Matching alerts are reclassified at ingest and still stored and counted — suppression
            removes them from the queue, not from the record.
          </Caption>
        </Panel>

        <DrillPanel report={drill} />
      </div>
    </main>
  );
}

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3 py-2 text-[11px] leading-relaxed text-[var(--color-ink-dim)] border-t border-[var(--color-border)]">
      {children}
    </p>
  );
}

interface PoisoningArm {
  dose?: number;
  n_flipped?: number;
  evaluation: { recall: number; fpr: number; target_recall: number; target_recall_ci95: [number, number] };
  gate?: { passed: boolean; reasons: string[] };
  integrity?: { poisoned_flagged_any: number; honest_accounts_clearances_flagged_any?: number };
}

interface PoisoningReport {
  target: string;
  target_train_support: number;
  arms: Record<string, PoisoningArm>;
}

/**
 * E7, measured: what the controls on this page actually catch. Pre-registered in EXPERIMENTS.md.
 * The page shows the drill next to the controls because a control without a measurement is a claim.
 */
function DrillPanel({ report }: { report: PoisoningReport | null }) {
  if (!report) {
    return (
      <Panel title="poisoning drill (E7)">
        <Empty>
          Not generated yet. <code>penumbra poison-drill</code> (about 13 minutes).
        </Empty>
      </Panel>
    );
  }
  return (
    <Panel
      title={`poisoning drill (E7) — one account flips its own ${report.target} alerts`}
      right={<Stat label="training rows" value={String(report.target_train_support)} />}
    >
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left font-normal px-3 py-1.5">arm</th>
            <th className="text-right font-normal px-3 py-1.5">flipped</th>
            <th className="text-right font-normal px-3 py-1.5">target recall</th>
            <th className="text-right font-normal px-3 py-1.5">FPR</th>
            <th className="text-center font-normal px-3 py-1.5">gate</th>
            <th className="text-right font-normal px-3 py-1.5">flips flagged</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(report.arms).map(([name, arm]) => {
            const [lo, hi] = arm.evaluation.target_recall_ci95;
            const flagged = arm.integrity?.poisoned_flagged_any;
            return (
              <tr key={name} className="border-b border-[var(--color-border)]">
                <td className="px-3 py-1.5">{name.replaceAll("_", " ")}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{arm.n_flipped ?? "—"}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {arm.evaluation.target_recall.toFixed(3)}{" "}
                  <span className="text-[var(--color-ink-faint)]">
                    [{lo.toFixed(2)}, {hi.toFixed(2)}]
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">{arm.evaluation.fpr.toFixed(4)}</td>
                <td
                  className={`px-3 py-1.5 text-center ${
                    arm.gate ? (arm.gate.passed ? "text-[var(--color-ok)]" : "text-[var(--color-sev-high)]") : ""
                  }`}
                >
                  {arm.gate ? (arm.gate.passed ? "pass" : "FAIL") : "—"}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {flagged == null || Number.isNaN(flagged) ? "—" : `${(flagged * 100).toFixed(0)}%`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Caption>
        The attack works only at full strength: while some honest verdicts on the same attack type
        remain, they outvote the flipped ones. The clearance-rate outlier flag caught every flip at
        every dose that did damage and never flagged an honest account. The canary gate failed the
        damaging model by name, and passed the honest retrain, which cut false positives by more
        than half.
      </Caption>
    </Panel>
  );
}

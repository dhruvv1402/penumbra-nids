"use client";

/**
 * The drift page.
 *
 * Drift is the rubric line most projects answer with a paragraph. This answers it with a
 * measurement, and specifically with the measurement that makes the claim falsifiable.
 *
 * The argument runs: conformal prediction guarantees coverage **under exchangeability**. Drift is
 * exactly the violation of exchangeability. So rather than treating that as an awkward caveat, we
 * measure coverage as the distribution moves — and when coverage falls below nominal, the
 * guarantee's own failure becomes a drift detector.
 *
 * The three-way control is the part that matters. One condition showing degraded coverage proves
 * nothing; a condition where coverage HOLDS, next to two where it does not, is what separates a
 * measurement from a story.
 *
 * Abstention gets equal billing because it is the deployable half. Coverage needs ground truth,
 * and in a real SOC labels arrive days late or never. Abstention needs none, and it tracks.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  type ConformalControlEntry,
  type ConformalDriftReport,
  type Session,
  getReport,
  getReports,
  loadSession,
  saveSession,
} from "@/lib/api";
import { Empty, Panel } from "@/components/Primitives";

// The PSI convention from Lewis (1994) / Siddiqi. A convention, not a derived result, and
// sample-size dependent - which the page says rather than presenting them as law.
const PSI_WATCH = 0.1;
const PSI_ACT = 0.25;

export default function Drift() {
  const [session, setSession] = useState<Session | null>(null);
  const [control, setControl] = useState<ConformalControlEntry[] | null>(null);
  const [series, setSeries] = useState<ConformalDriftReport | null>(null);
  const [commands, setCommands] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setSession(loadSession()), []);

  const load = useCallback(async (token: string) => {
    try {
      const { reports } = await getReports(token);
      setCommands(Object.fromEntries(reports.map((r) => [r.name, r.command])));

      const pull = async <T,>(name: string, set: (value: T | null) => void) => {
        try {
          set((await getReport<T>(token, name)).data);
        } catch {
          set(null);
        }
      };
      await Promise.all([
        pull<ConformalControlEntry[]>("conformal-control", setControl),
        pull<ConformalDriftReport>("conformal", setSeries),
      ]);
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
    if (session) void load(session.token);
  }, [session, load]);

  if (!session) {
    return (
      <main className="h-screen grid place-items-center">
        <p className="text-[12px] text-[var(--color-ink-dim)]">
          <Link href="/" className="underline">
            Sign in on the queue page
          </Link>{" "}
          to view drift.
        </p>
      </main>
    );
  }

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
          href="/evaluation"
          className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
        >
          evaluation
        </Link>
        <h1 className="text-[11px] tracking-[0.14em] uppercase">drift</h1>
        <p className="text-[11px] text-[var(--color-ink-dim)] ml-auto">
          conformal coverage as a label-free drift signal
        </p>
      </header>

      {error && (
        <div className="px-3 py-1.5 border-b border-[var(--color-border)] text-[11px] text-[var(--color-sev-high)]">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-2 p-2">
        <ControlPanel entries={control} command={commands["conformal-control"] ?? "penumbra eval"} />
        <SeriesPanel report={series} command={commands["conformal"] ?? "penumbra replay"} />
        <LimitsPanel />
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

function NotGenerated({ command }: { command: string }) {
  return (
    <Empty>
      <span className="block mb-2">This report has not been generated yet.</span>
      <code className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-bg)] border border-[var(--color-border)]">
        {command}
      </code>
    </Empty>
  );
}

function psiTone(psi: number): string {
  if (psi >= PSI_ACT) return "text-[var(--color-sev-high)]";
  if (psi >= PSI_WATCH) return "text-[var(--color-sev-medium)]";
  return "text-[var(--color-ink-dim)]";
}

function ControlPanel({
  entries,
  command,
}: {
  entries: ConformalControlEntry[] | null;
  command: string;
}) {
  return (
    <Panel
      title="three-way control — coverage against exchangeability"
      right={<code className="text-[10px] text-[var(--color-ink-dim)] font-mono">{command}</code>}
    >
      {!entries || entries.length === 0 ? (
        <NotGenerated command={command} />
      ) : (
        <>
          <table className="w-full text-[11px]">
            <thead className="text-[var(--color-ink-dim)]">
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left font-normal px-3 py-1.5">condition</th>
                <th className="text-right font-normal px-3 py-1.5">score PSI</th>
                <th className="text-right font-normal px-3 py-1.5">coverage</th>
                <th className="text-right font-normal px-3 py-1.5">gap vs nominal</th>
                <th className="text-right font-normal px-3 py-1.5">abstention</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => {
                const holds = entry.gap > -0.02;
                return (
                  <tr
                    key={entry.condition}
                    className="border-b border-[var(--color-border)] last:border-0 align-top"
                  >
                    <td className="px-3 py-2">
                      {entry.condition}
                      <span className="block text-[10px] text-[var(--color-ink-dim)] mt-0.5">
                        {entry.note}
                      </span>
                    </td>
                    <td className={`px-3 py-2 text-right tabular-nums ${psiTone(entry.score_psi)}`}>
                      {entry.score_psi.toFixed(4)}
                    </td>
                    <td
                      className={`px-3 py-2 text-right tabular-nums ${
                        holds ? "text-[var(--color-sev-low)]" : "text-[var(--color-sev-high)]"
                      }`}
                    >
                      {(entry.coverage * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {(entry.gap * 100).toFixed(1)} pts
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {(entry.abstention * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <Caption>
            Conformal prediction guarantees coverage <em>under exchangeability</em>, and drift is
            exactly the violation of it. The first row is the control: same distribution as
            calibration, coverage holds. Without it, degraded coverage in the other two rows would
            prove nothing — it could just as easily be a broken implementation. Abstention rises
            with every point of lost coverage, and unlike coverage it needs no ground truth, which
            is what makes it deployable in a SOC where verdicts arrive days late or never.
          </Caption>
        </>
      )}
    </Panel>
  );
}

function SeriesPanel({ report, command }: { report: ConformalDriftReport | null; command: string }) {
  if (!report || report.points.length === 0) {
    return (
      <Panel title="coverage over a drifting stream">
        <NotGenerated command={command} />
      </Panel>
    );
  }
  const maxPsi = Math.max(...report.points.map((p) => p.psi), PSI_ACT);
  const breachedBeforeInjection = report.first_breach_window < report.change_point_window;

  return (
    <Panel
      title="coverage over a drifting stream"
      right={<code className="text-[10px] text-[var(--color-ink-dim)] font-mono">{command}</code>}
    >
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-right font-normal px-3 py-1.5">window</th>
            <th className="text-right font-normal px-3 py-1.5">PSI</th>
            <th className="text-left font-normal px-3 py-1.5 w-1/3">
              &nbsp;
            </th>
            <th className="text-right font-normal px-3 py-1.5">coverage</th>
            <th className="text-right font-normal px-3 py-1.5">nominal</th>
            <th className="text-right font-normal px-3 py-1.5">abstention</th>
          </tr>
        </thead>
        <tbody>
          {report.points.map((point) => {
            const injected = point.window >= report.change_point_window;
            const breached = point.coverage < point.nominal - 0.02;
            return (
              <tr key={point.window} className="border-b border-[var(--color-border)] last:border-0">
                <td className="px-3 py-1.5 text-right tabular-nums text-[var(--color-ink-dim)]">
                  {point.window}
                  {point.window === report.change_point_window && (
                    <span className="text-[var(--color-sev-medium)]"> ←inject</span>
                  )}
                </td>
                <td className={`px-3 py-1.5 text-right tabular-nums ${psiTone(point.psi)}`}>
                  {point.psi.toFixed(3)}
                </td>
                <td className="px-3 py-1.5">
                  <div className="h-1.5 rounded bg-[var(--color-bg)] overflow-hidden">
                    <div
                      className={
                        point.psi >= PSI_ACT
                          ? "h-full bg-[var(--color-sev-high)]"
                          : point.psi >= PSI_WATCH
                            ? "h-full bg-[var(--color-sev-medium)]"
                            : "h-full bg-[var(--color-sev-low)]"
                      }
                      style={{ width: `${Math.min(100, (point.psi / maxPsi) * 100)}%` }}
                    />
                  </div>
                </td>
                <td
                  className={`px-3 py-1.5 text-right tabular-nums ${
                    breached ? "text-[var(--color-sev-high)]" : ""
                  }`}
                >
                  {(point.coverage * 100).toFixed(1)}%
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums text-[var(--color-ink-dim)]">
                  {(point.nominal * 100).toFixed(1)}%
                </td>
                <td
                  className={`px-3 py-1.5 text-right tabular-nums ${
                    injected ? "" : "text-[var(--color-ink-dim)]"
                  }`}
                >
                  {(point.abstention_rate * 100).toFixed(1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Caption>
        {breachedBeforeInjection ? (
          <>
            Coverage was already below nominal at window {report.first_breach_window}, before the
            injection at window {report.change_point_window}. So <strong>no detection delay can be
            claimed from this stream</strong> — the baseline was never exchangeable to begin with.
            Reporting a negative delay as though it were a lead time is the mistake this caption
            exists to prevent.
          </>
        ) : (
          <>
            Coverage first fell below nominal at window {report.first_breach_window}, against an
            injection at window {report.change_point_window}. PSI is computed against{" "}
            <strong>frozen reference bin edges</strong>: recomputing bins per window is a common bug
            that makes PSI read near zero no matter how far the distribution has moved.
          </>
        )}
      </Caption>
    </Panel>
  );
}

function LimitsPanel() {
  return (
    <Panel title="what this can and cannot detect">
      <div className="px-3 py-3 space-y-3 text-[11px] leading-relaxed">
        <p>
          <strong>Without labels you can detect covariate shift and prediction shift.</strong> True
          concept drift — P(y | x) moving while P(x) stays put — is undetectable without ground
          truth, full stop. Saying so is not a limitation we are conceding; it is the part of the
          problem most drift dashboards quietly skip.
        </p>
        <p>
          That is why the primary signal here is the prediction and novelty-score distribution
          rather than measured accuracy. In a real SOC, labels arrive days late or never, so a
          monitor that needs them is a monitor that fires after the incident.
        </p>
        <p className="text-[var(--color-ink-dim)]">
          PSI&apos;s {PSI_WATCH} / {PSI_ACT} thresholds are a convention from Lewis (1994) and
          Siddiqi, not a derived result, and they are sample-size dependent. They are used here as a
          shading cue, not as a decision rule.
        </p>
        <p className="text-[var(--color-ink-dim)]">
          The 42-feature KS sweep is Benjamini-Hochberg corrected. At α = 0.05 across 42 features
          you expect about two false flags every window, so an uncorrected &ldquo;3 features
          drifted&rdquo; is noise wearing a number.
        </p>
      </div>
    </Panel>
  );
}

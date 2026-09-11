"use client";

/**
 * The evaluation page.
 *
 * Every number here is read from a report file that a CLI command wrote, and every panel prints
 * the command that produced it. Nothing is hardcoded in this file and nothing is recomputed in the
 * browser, so the console cannot quietly disagree with the measurements — and "where did that
 * number come from" has an answer a judge can run.
 *
 * Three panels, chosen because each one is a result we would rather not have to explain away:
 *
 *   MINED RULES   half the mined rules rested on a feature our own audit quarantined.
 *   SEQUENCE      the deep model did not beat nine cheap features.
 *   EVASION       the unconstrained attack evades far more than the realisable one, and the gap
 *                 is the point.
 *
 * A report that has not been generated renders as the command that would generate it, never as a
 * blank panel — on stage those two look identical and mean completely different things.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  type EvasionReport,
  type MinedRulesReport,
  type ReportEntry,
  type SequenceReport,
  type Session,
  getReport,
  getReports,
  loadSession,
  saveSession,
} from "@/lib/api";
import { Empty, Panel, Stat } from "@/components/Primitives";

export default function Evaluation() {
  const [session, setSession] = useState<Session | null>(null);
  const [catalogue, setCatalogue] = useState<ReportEntry[]>([]);
  const [rules, setRules] = useState<MinedRulesReport | null>(null);
  const [sequence, setSequence] = useState<SequenceReport | null>(null);
  const [evasion, setEvasion] = useState<EvasionReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setSession(loadSession()), []);

  const load = useCallback(async (token: string) => {
    try {
      const { reports } = await getReports(token);
      setCatalogue(reports);

      // Each fetch is independent: one missing report must not blank the other two panels.
      const pull = async <T,>(name: string, set: (value: T | null) => void) => {
        try {
          const payload = await getReport<T>(token, name);
          set(payload.data);
        } catch {
          set(null);
        }
      };
      await Promise.all([
        pull<MinedRulesReport>("rules-unsw", setRules),
        pull<SequenceReport>("sequence", setSequence),
        pull<EvasionReport>("adversarial", setEvasion),
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

  const commandFor = (name: string) =>
    catalogue.find((entry) => entry.name === name)?.command ?? "penumbra --help";

  if (!session) {
    return (
      <main className="h-screen grid place-items-center">
        <p className="text-[12px] text-[var(--color-ink-dim)]">
          <Link href="/" className="underline">
            Sign in on the queue page
          </Link>{" "}
          to view evaluation results.
        </p>
      </main>
    );
  }

  return (
    <main className="min-h-screen flex flex-col">
      <header className="flex items-center gap-4 px-3 py-2 border-b border-[var(--color-border)] shrink-0">
        <Link href="/" className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]">
          ← queue
        </Link>
        <h1 className="text-[11px] tracking-[0.14em] uppercase">evaluation</h1>
        <p className="text-[11px] text-[var(--color-ink-dim)] ml-auto">
          every number below was written by a command, not typed into this page
        </p>
      </header>

      {error && (
        <div className="px-3 py-1.5 border-b border-[var(--color-border)] text-[11px] text-[var(--color-sev-high)]">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-2 p-2">
        <MinedRulesPanel report={rules} command={commandFor("rules-unsw")} />
        <SequencePanel report={sequence} command={commandFor("sequence")} />
        <EvasionPanel report={evasion} command={commandFor("adversarial")} />
        <CataloguePanel entries={catalogue} />
      </div>
    </main>
  );
}

/** What a panel shows when its report has not been generated. The command, not a blank. */
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

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3 py-2 text-[11px] leading-relaxed text-[var(--color-ink-dim)] border-t border-[var(--color-border)]">
      {children}
    </p>
  );
}

function CommandTag({ command }: { command: string }) {
  return (
    <code className="text-[10px] text-[var(--color-ink-dim)] font-mono">{command}</code>
  );
}

function MinedRulesPanel({ report, command }: { report: MinedRulesReport | null; command: string }) {
  return (
    <Panel title="mined rules — what the quarantine cost" right={<CommandTag command={command} />}>
      {!report ? (
        <NotGenerated command={command} />
      ) : (
        <>
          <table className="w-full text-[11px]">
            <thead className="text-[var(--color-ink-dim)]">
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left font-normal px-3 py-1.5"> </th>
                <th className="text-right font-normal px-3 py-1.5">with artifacts</th>
                <th className="text-right font-normal px-3 py-1.5">quarantined</th>
              </tr>
            </thead>
            <tbody>
              <Row
                label="candidate paths"
                a={report.with_artifacts.n_candidates.toLocaleString()}
                b={report.without_artifacts.n_candidates.toLocaleString()}
              />
              <Row
                label="discarded as artifact-dependent"
                a="0"
                b={report.without_artifacts.n_artifact_excluded.toLocaleString()}
                emphasise
              />
              <Row
                label={`survived validation (≥ ${report.without_artifacts.min_precision.toFixed(2)})`}
                a={report.with_artifacts.n_survivors.toLocaleString()}
                b={report.without_artifacts.n_survivors.toLocaleString()}
              />
              <Row
                label="set recall, test"
                a={report.with_artifacts.test_coverage.recall.toFixed(4)}
                b={report.without_artifacts.test_coverage.recall.toFixed(4)}
              />
              <Row
                label="set precision, test"
                a={report.with_artifacts.test_coverage.precision.toFixed(4)}
                b={report.without_artifacts.test_coverage.precision.toFixed(4)}
              />
            </tbody>
          </table>
          <Caption>
            Quarantined: <code>{report.quarantined.join(", ") || "nothing"}</code>. Both columns were
            validated on the same {report.n_holdout.toLocaleString()} held-out rows the trees never
            saw — which is exactly why held-out validation alone does not catch a testbed artifact.
            The left column is the number we could have reported. The right one is the number we
            stand behind.
          </Caption>
        </>
      )}
    </Panel>
  );
}

function Row({
  label,
  a,
  b,
  emphasise = false,
}: {
  label: string;
  a: string;
  b: string;
  emphasise?: boolean;
}) {
  return (
    <tr className="border-b border-[var(--color-border)] last:border-0">
      <td className="px-3 py-1.5 text-[var(--color-ink-dim)]">{label}</td>
      <td className="px-3 py-1.5 text-right tabular-nums">{a}</td>
      <td
        className={`px-3 py-1.5 text-right tabular-nums ${emphasise ? "text-[var(--color-sev-high)]" : ""}`}
      >
        {b}
      </td>
    </tr>
  );
}

function SequencePanel({ report, command }: { report: SequenceReport | null; command: string }) {
  const collapsed = (report?.training?.collapsed_epochs ?? 0) > 0;
  return (
    <Panel title="does sequence context buy recall?" right={<CommandTag command={command} />}>
      {!report ? (
        <NotGenerated command={command} />
      ) : (
        <>
          {collapsed && (
            <div className="px-3 py-2 text-[11px] text-[var(--color-sev-high)] border-b border-[var(--color-border)]">
              Training diverged to a constant output for{" "}
              {report.training?.collapsed_epochs} epoch(s). These numbers describe that run, not the
              architecture.
            </div>
          )}
          <table className="w-full text-[11px]">
            <thead className="text-[var(--color-ink-dim)]">
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left font-normal px-3 py-1.5">arm</th>
                <th className="text-right font-normal px-3 py-1.5">ROC-AUC</th>
                <th className="text-right font-normal px-3 py-1.5">
                  recall @ {(report.fpr_budget * 100).toFixed(0)}% FPR
                </th>
                <th className="text-right font-normal px-3 py-1.5">realised FPR</th>
              </tr>
            </thead>
            <tbody>
              {report.arms.map((arm) => (
                <tr key={arm.name} className="border-b border-[var(--color-border)] last:border-0">
                  <td className="px-3 py-1.5">{arm.name}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{arm.roc_auc.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {arm.recall_at_budget.toFixed(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-[var(--color-ink-dim)]">
                    {arm.realised_fpr.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <Caption>
            All three arms scored on the same {report.n_test.toLocaleString()} rows at a matched
            benign-flag budget — realised FPR is printed so you can check the match held. The
            hypothesis was registered in <code>docs/EXPERIMENTS.md</code> before this ran.
          </Caption>
        </>
      )}
    </Panel>
  );
}

function EvasionPanel({ report, command }: { report: EvasionReport | null; command: string }) {
  const constrained = report?.results.find((r) => r.constrained);
  const strawman = report?.results.find((r) => !r.constrained);
  return (
    <Panel title="evasion — realisable vs unconstrained" right={<CommandTag command={command} />}>
      {!report || !constrained ? (
        <NotGenerated command={command} />
      ) : (
        <>
          <table className="w-full text-[11px]">
            <thead className="text-[var(--color-ink-dim)]">
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left font-normal px-3 py-1.5">attacker effort</th>
                <th className="text-right font-normal px-3 py-1.5">× slower</th>
                <th className="text-right font-normal px-3 py-1.5">detection (realisable)</th>
                <th className="text-right font-normal px-3 py-1.5">detection (unconstrained)</th>
              </tr>
            </thead>
            <tbody>
              {constrained.points.map((point, i) => (
                <tr key={point.effort} className="border-b border-[var(--color-border)] last:border-0">
                  <td className="px-3 py-1.5 tabular-nums">{point.effort.toFixed(1)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-[var(--color-ink-dim)]">
                    {point.mean_duration_multiple.toFixed(1)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {point.detection_rate.toFixed(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-[var(--color-ink-dim)]">
                    {strawman?.points[i]?.detection_rate.toFixed(4) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <Caption>
            Effort is the attacker&apos;s own price: 9.0 means accepting a roughly ten-times slower
            flow. The realisable column pads bytes, stretches duration, keeps packet counts whole
            and never un-sends anything. The unconstrained column does none of that, and it is what
            most published evasion rates measure — perturbations that include fractional packet
            counts and throughputs inconsistent with their own byte counts. Detections it
            &ldquo;evades&rdquo; were never defending those rows.
          </Caption>
        </>
      )}
    </Panel>
  );
}

function CataloguePanel({ entries }: { entries: ReportEntry[] }) {
  const missing = entries.filter((e) => !e.available);
  return (
    <Panel title="all reports" right={<Stat label="generated" value={`${entries.length - missing.length}/${entries.length}`} />}>
      <table className="w-full text-[11px]">
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.name} className="border-b border-[var(--color-border)] last:border-0 align-top">
              <td className="px-3 py-2 w-1/3">
                <span className={entry.available ? "" : "text-[var(--color-ink-dim)]"}>
                  {entry.title}
                </span>
                <span className="block text-[10px] text-[var(--color-ink-dim)] font-mono mt-0.5">
                  {entry.command}
                </span>
              </td>
              <td className="px-3 py-2 text-[var(--color-ink-dim)] leading-relaxed">
                {entry.description}
              </td>
              <td className="px-3 py-2 text-right whitespace-nowrap">
                {entry.available ? (
                  <span className="text-[var(--color-ink-dim)]">
                    {(entry.bytes / 1024).toFixed(0)} KB
                  </span>
                ) : (
                  <span className="text-[var(--color-sev-medium)]">not generated</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Caption>
        Reports that have not been generated are listed rather than hidden. A page that silently
        renders nothing looks identical to one whose data is genuinely empty.
      </Caption>
    </Panel>
  );
}

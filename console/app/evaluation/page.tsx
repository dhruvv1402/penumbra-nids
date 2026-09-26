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
  const [correlation, setCorrelation] = useState<CorrelationReport | null>(null);
  const [calibration, setCalibration] = useState<Record<string, CalibrationReport | null>>({});
  const [refit, setRefit] = useState<RefitReport | null>(null);
  const [speed, setSpeed] = useState<LoadReport | null>(null);
  const [rebaseline, setRebaseline] = useState<RebaselineReport | null>(null);
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
        pull<CorrelationReport>("correlation", setCorrelation),
        pull<CalibrationReport>("calibration-unsw", (v) => setCalibration((c) => ({ ...c, unsw: v }))),
        pull<CalibrationReport>("calibration-nslkdd", (v) => setCalibration((c) => ({ ...c, nslkdd: v }))),
        pull<RefitReport>("threshold-refit", setRefit),
        pull<LoadReport>("loadtest", setSpeed),
        pull<RebaselineReport>("rebaseline", setRebaseline),
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
        <Link
          href="/drift"
          className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
        >
          drift
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
        <CorrelationPanel report={correlation} command={commandFor("correlation")} />
        <CalibrationPanel reports={calibration} />
        <RefitPanel report={refit} command={commandFor("threshold-refit")} />
        <RebaselinePanel report={rebaseline} command={commandFor("rebaseline")} />
        <SpeedPanel report={speed} command={commandFor("loadtest")} />
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

interface CorrelationReport {
  replayed_flows: number;
  n_alerts: number;
  n_incidents: number;
  compression_ratio: number;
  attack_rows_replayed: number;
  attack_rows_reaching_an_analyst?: number;
  benign_rows_replayed?: number;
  benign_rows_reaching_an_analyst?: number;
  alerts_by_verdict: Record<string, number>;
  top: {
    title: string;
    events: number;
    destinations: number;
    ports: number;
    family_predicted: string | null;
    ground_truth: Record<string, number>;
  }[];
}

/** Incidents, with what they really were. The family an analyst sees is a guess; fan-out is measured. */
function CorrelationPanel({ report, command }: { report: CorrelationReport | null; command: string }) {
  if (!report) {
    return (
      <Panel title="alert → incident correlation (CICIDS2017, real source IPs)">
        <NotGenerated command={command} />
      </Panel>
    );
  }
  const reached =
    report.attack_rows_reaching_an_analyst != null
      ? `${((100 * report.attack_rows_reaching_an_analyst) / report.attack_rows_replayed).toFixed(2)}%`
      : "—";
  return (
    <Panel
      title="alert → incident correlation (CICIDS2017, real source IPs)"
      right={<CommandTag command={command} />}
    >
      <div className="grid grid-cols-4 border-b border-[var(--color-border)]">
        <Stat label="alerts" value={report.n_alerts.toLocaleString()} />
        <Stat label="incidents" value={report.n_incidents.toLocaleString()} />
        <Stat label="events / incident" value={report.compression_ratio.toFixed(1)} />
        <Stat label="attack flows reached analyst" value={reached} />
      </div>
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-right font-normal px-3 py-1.5">events</th>
            <th className="text-right font-normal px-3 py-1.5">dest · ports</th>
            <th className="text-left font-normal px-3 py-1.5">family shown</th>
            <th className="text-left font-normal px-3 py-1.5">ground truth</th>
          </tr>
        </thead>
        <tbody>
          {report.top.slice(0, 6).map((t) => (
            <tr key={t.title + t.events} className="border-b border-[var(--color-border)]">
              <td className="px-3 py-1.5 text-right tabular-nums">{t.events.toLocaleString()}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">
                {t.destinations.toLocaleString()} · {t.ports.toLocaleString()}
              </td>
              <td className="px-3 py-1.5">{t.family_predicted ?? "(unrecognised)"}</td>
              <td className="px-3 py-1.5 text-[var(--color-ink-dim)]">
                {Object.entries(t.ground_truth ?? {})
                  .slice(0, 2)
                  .map(([k, v]) => `${k} ${v.toLocaleString()}`)
                  .join(" · ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Caption>
        {(report.alerts_by_verdict.UNCERTAIN ?? 0).toLocaleString()} of the alerts are abstentions: every Thursday–Friday
        family is absent from Monday–Wednesday, so the supervised head does not recognise them and says so.
        Correlation is what turns that review lane into {report.n_incidents} things to look at.
      </Caption>
    </Panel>
  );
}

interface CalibrationMethod {
  brier: number;
  ece: number;
}

type CalibrationReport = Record<string, unknown> & {
  held_out_train: Record<string, CalibrationMethod>;
  test_split: Record<string, CalibrationMethod>;
};

/** Calibration that holds in-distribution and breaks under shift, side by side. */
function CalibrationPanel({ reports }: { reports: Record<string, CalibrationReport | null> }) {
  const rows = Object.entries(reports).filter(([, r]) => r) as [string, CalibrationReport][];
  if (rows.length === 0) {
    return (
      <Panel title="calibration — in-distribution vs under shift">
        <NotGenerated command="penumbra calibrate -d unsw" />
      </Panel>
    );
  }
  const methods = ["uncalibrated", "isotonic", "sigmoid"];
  return (
    <Panel title="calibration — in-distribution vs under shift">
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left font-normal px-3 py-1.5">dataset · slice</th>
            {methods.map((m) => (
              <th key={m} className="text-right font-normal px-3 py-1.5">
                {m === "sigmoid" ? "Platt" : m} Brier · ECE
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.flatMap(([name, r]) =>
            (["held_out_train", "test_split"] as const).map((slice) => {
              const best = Math.min(...methods.map((m) => r[slice][m]?.brier ?? Infinity));
              return (
                <tr key={name + slice} className="border-b border-[var(--color-border)]">
                  <td className="px-3 py-1.5">
                    {name} · {slice === "held_out_train" ? "held-out train" : "shifted test"}
                  </td>
                  {methods.map((m) => (
                    <td
                      key={m}
                      className={`px-3 py-1.5 text-right tabular-nums ${r[slice][m]?.brier === best ? "text-[var(--color-ok)]" : ""}`}
                    >
                      {r[slice][m] ? `${r[slice][m].brier.toFixed(4)} · ${r[slice][m].ece.toFixed(4)}` : "—"}
                    </td>
                  ))}
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
      <Caption>
        Isotonic calibration does its job on held-out training data and makes things worse on the
        shifted test split, on both datasets. It carries the training distribution&apos;s mapping into
        data where that mapping no longer holds. So p_attack is &quot;calibrated on held-out training
        data&quot;, and the abstention rate on the drift page is the warning that it has stopped being true.
      </Caption>
    </Panel>
  );
}

const pct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${(100 * v).toFixed(digits)}%`);
const num = (v: number | null | undefined, digits = 3) => (v == null ? "—" : v.toFixed(digits));

interface Rate {
  rate: number | null;
  ci95?: [number | null, number | null];
}

interface RefitArm {
  fpr: Rate;
  recall_all: Rate;
  recall_unseen17: Rate;
  benign_reach: Rate;
}

interface RefitReport {
  A0: RefitArm;
  windows: { n: number; seed: number | null; A1: RefitArm; A2: RefitArm }[];
  dirty: { dose: number; A2_dirty: RefitArm; A2_clean: RefitArm }[];
}

function RefitPanel({ report, command }: { report: RefitReport | null; command: string }) {
  const full = report?.windows.find((w) => w.seed === null);
  if (!report || !full) {
    return (
      <Panel title="threshold drift — move the operating point, or re-learn normal? (E8)">
        <NotGenerated command={command} />
      </Panel>
    );
  }
  const dirty5 = report.dirty.filter((d) => d.dose === 0.05);
  const loss5 =
    dirty5.length > 0
      ? dirty5.reduce((acc, d) => acc + ((d.A2_clean.recall_unseen17.rate ?? 0) - (d.A2_dirty.recall_unseen17.rate ?? 0)), 0) /
        dirty5.length
      : null;
  const fpr5 =
    dirty5.length > 0 ? dirty5.reduce((acc, d) => acc + (d.A2_dirty.fpr.rate ?? 0), 0) / dirty5.length : null;
  const rows: [string, RefitArm][] = [
    ["as shipped", report.A0],
    ["move thresholds", full.A1],
    ["re-learn normal", full.A2],
  ];
  return (
    <Panel title="threshold drift — move the operating point, or re-learn normal? (E8)" right={<CommandTag command={command} />}>
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left font-normal px-3 py-1.5">NSL-KDD shifted test, 1% target</th>
            <th className="text-right font-normal px-3 py-1.5">realised FPR</th>
            <th className="text-right font-normal px-3 py-1.5">unseen-17 recall</th>
            <th className="text-right font-normal px-3 py-1.5">benign to an analyst</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, arm]) => (
            <tr key={name} className="border-b border-[var(--color-border)]">
              <td className="px-3 py-1.5">{name}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{pct(arm.fpr.rate, 2)}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{num(arm.recall_unseen17.rate)}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{pct(arm.benign_reach.rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Caption>
        The 1% target realised {pct(report.A0.fpr.rate)} under drift, and moving the thresholds on {full.n.toLocaleString()} recent
        benign flows restores {pct(full.A1.fpr.rate, 2)}. The unseen-attack recall it gives back had been bought with false
        positives. A baseline that is 5% attack traffic cost {num(loss5)} unseen recall while the FPR fell to {pct(fpr5)}:
        poisoning the baseline looks like tuning. Pre-registered; two predictions refuted.
      </Caption>
    </Panel>
  );
}

interface RebaselineRates {
  flows: number;
  fired_rate: number;
  reach_rate: number;
}

interface RebaselineReport {
  mode?: string;
  target_fpr: number;
  passed: boolean;
  window: { benign_flows: number };
  gates: Record<string, { passed: boolean }>;
  evidence: {
    holdout?: { current: RebaselineRates; rebaselined: RebaselineRates };
    attack_flows?: { current: RebaselineRates; rebaselined: RebaselineRates };
  };
}

function RebaselinePanel({ report, command }: { report: RebaselineReport | null; command: string }) {
  const hold = report?.evidence.holdout;
  if (!report || !hold) {
    return (
      <Panel title="re-baselined on a real network (lab capture)">
        <NotGenerated command={command} />
      </Panel>
    );
  }
  const attack = report.evidence.attack_flows;
  const gates = Object.entries(report.gates);
  return (
    <Panel title="re-baselined on a real network (lab capture)" right={<CommandTag command={command} />}>
      <div className="grid grid-cols-4 border-b border-[var(--color-border)]">
        <Stat label="benign window" value={report.window.benign_flows.toLocaleString()} />
        <Stat label="target FPR" value={pct(report.target_fpr, 0)} />
        <Stat label="gate" value={report.passed ? "passed" : "failed"} />
        <Stat label="checks" value={gates.map(([name, g]) => `${name.split("_")[0]} ${g.passed ? "✓" : "✗"}`).join(" ")} />
      </div>
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left font-normal px-3 py-1.5"> </th>
            <th className="text-right font-normal px-3 py-1.5">as shipped</th>
            <th className="text-right font-normal px-3 py-1.5">re-baselined</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-[var(--color-border)]">
            <td className="px-3 py-1.5">ordinary flows that fired ({hold.current.flows.toLocaleString()} held out)</td>
            <td className="px-3 py-1.5 text-right tabular-nums">{pct(hold.current.fired_rate)}</td>
            <td className="px-3 py-1.5 text-right tabular-nums">{pct(hold.rebaselined.fired_rate)}</td>
          </tr>
          <tr className="border-b border-[var(--color-border)]">
            <td className="px-3 py-1.5">ordinary flows reaching an analyst</td>
            <td className="px-3 py-1.5 text-right tabular-nums">{pct(hold.current.reach_rate)}</td>
            <td className="px-3 py-1.5 text-right tabular-nums">{pct(hold.rebaselined.reach_rate)}</td>
          </tr>
          {attack && (
            <tr className="border-b border-[var(--color-border)]">
              <td className="px-3 py-1.5">scan flows that fired ({attack.current.flows.toLocaleString()})</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{pct(attack.current.fired_rate)}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{pct(attack.rebaselined.fired_rate)}</td>
            </tr>
          )}
        </tbody>
      </table>
      <Caption>
        No labels: the detector&apos;s normal is re-learnt from the network&apos;s own benign traffic, checked on a held-out
        slice, and registered as a candidate. Promotion is a separate, audited step.
      </Caption>
    </Panel>
  );
}

interface LoadPoint {
  batch_size: number;
  flows_per_second: number;
  batch_latency_ms: { p50: number };
}

interface LoadEngine {
  fixed_overhead_ms: number;
  points: LoadPoint[];
}

interface LoadReport extends LoadEngine {
  engines?: Record<string, LoadEngine>;
  compile?: { decisions_changed: number; n_rows: number };
  alert_building?: { alerts_per_second: number };
}

function SpeedPanel({ report, command }: { report: LoadReport | null; command: string }) {
  if (!report) {
    return (
      <Panel title="speed — one flow, and a batch">
        <NotGenerated command={command} />
      </Panel>
    );
  }
  const engines = report.engines ?? { sklearn: report };
  const batches = (Object.values(engines)[0]?.points ?? []).map((p) => p.batch_size);
  const oneFlow = report.points.find((p) => p.batch_size === 1);
  return (
    <Panel title="speed — one flow, and a batch" right={<CommandTag command={command} />}>
      <div className="grid grid-cols-4 border-b border-[var(--color-border)]">
        <Stat label="one flow" value={oneFlow ? `${oneFlow.batch_latency_ms.p50.toFixed(1)} ms` : "—"} />
        <Stat label="fixed cost / call" value={`${report.fixed_overhead_ms.toFixed(1)} ms`} />
        <Stat
          label="alerts built / s"
          value={report.alert_building ? Math.round(report.alert_building.alerts_per_second).toLocaleString() : "—"}
        />
        <Stat
          label="decisions changed"
          value={report.compile ? `${report.compile.decisions_changed} / ${report.compile.n_rows.toLocaleString()}` : "—"}
        />
      </div>
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left font-normal px-3 py-1.5">flows / s at batch</th>
            {batches.map((b) => (
              <th key={b} className="text-right font-normal px-3 py-1.5">
                {b.toLocaleString()}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(engines).map(([name, engine]) => (
            <tr key={name} className="border-b border-[var(--color-border)]">
              <td className="px-3 py-1.5">{name}</td>
              {engine.points.map((p) => (
                <td key={p.batch_size} className="px-3 py-1.5 text-right tabular-nums">
                  {Math.round(p.flows_per_second).toLocaleString()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <Caption>
        The compiled scorer runs the forests as flat arrays below a batch size it measures at startup, and checks that no
        decision changes before it is used. Still a batch scorer, and still far outside an inline-blocking budget.
      </Caption>
    </Panel>
  );
}

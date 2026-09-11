"use client";

/**
 * Shared display primitives.
 *
 * Two of these carry design decisions rather than styling:
 *
 * `ScoreTriplet` renders p_attack, novelty_percentile and priority as three visibly different
 * things. They are different kinds of number - a calibrated probability, a percentile rank, and a
 * sort key - and a console that shows them as three interchangeable percentages teaches the analyst
 * something false (ADR-0002).
 *
 * `AttackChip` always shows the mapping's confidence tier. There is no authoritative mapping from
 * these dataset taxonomies to ATT&CK, so a technique ID presented without its confidence overstates
 * what we know.
 */

import type { AttackTechnique, Contribution, Severity, Verdict } from "@/lib/api";

export const SEVERITY_COLOR: Record<Severity, string> = {
  High: "var(--color-sev-high)",
  Medium: "var(--color-sev-medium)",
  Low: "var(--color-sev-low)",
  Informational: "var(--color-sev-info)",
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  KNOWN_ATTACK: "KNOWN ATTACK",
  SUSPECTED_NOVEL: "SUSPECTED NOVEL",
  UNCERTAIN: "UNCERTAIN",
  BENIGN: "BENIGN",
  BENIGN_BY_POLICY: "BENIGN BY POLICY",
};

export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`bg-[var(--color-panel)] border border-[var(--color-border)] rounded flex flex-col min-h-0 ${className}`}
    >
      {title && (
        <header className="flex items-center justify-between px-3 py-2 border-b border-[var(--color-border)] shrink-0">
          <h2 className="text-[11px] tracking-[0.14em] text-[var(--color-ink-dim)] uppercase">
            {title}
          </h2>
          {right}
        </header>
      )}
      <div className="flex-1 min-h-0 overflow-auto">{children}</div>
    </section>
  );
}

export function SeverityDot({ severity }: { severity: Severity }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full shrink-0"
      style={{ background: SEVERITY_COLOR[severity] }}
      aria-label={severity}
    />
  );
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const novel = verdict === "SUSPECTED_NOVEL";
  const known = verdict === "KNOWN_ATTACK";
  const color = novel
    ? "var(--color-novel)"
    : known
      ? "var(--color-sev-high)"
      : "var(--color-ink-faint)";
  return (
    <span
      className="text-[10px] tracking-wider px-1.5 py-0.5 rounded border"
      style={{ color, borderColor: color, background: `color-mix(in srgb, ${color} 12%, transparent)` }}
    >
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

/**
 * The three numbers, rendered as three different things on purpose.
 */
export function ScoreTriplet({
  pAttack,
  novelty,
  priority,
}: {
  pAttack: number;
  novelty: number;
  priority: number;
}) {
  return (
    <div className="grid grid-cols-3 gap-2 text-[11px]">
      <Metric
        label="p(attack)"
        value={pAttack.toFixed(3)}
        hint="calibrated probability"
        bar={pAttack}
        color="var(--color-sev-high)"
      />
      <Metric
        label="novelty"
        value={`${(novelty * 100).toFixed(1)}%`}
        hint="percentile vs benign"
        bar={novelty}
        color="var(--color-novel)"
      />
      <Metric
        label="priority"
        value={String(priority)}
        hint="sort key, not a probability"
        bar={priority / 100}
        color="var(--color-sev-low)"
      />
    </div>
  );
}

function Metric({
  label,
  value,
  hint,
  bar,
  color,
}: {
  label: string;
  value: string;
  hint: string;
  bar: number;
  color: string;
}) {
  return (
    <div className="bg-[var(--color-panel-2)] border border-[var(--color-border)] rounded px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-[var(--color-ink-faint)]">{label}</div>
      <div className="text-[15px] mt-0.5" style={{ color }}>
        {value}
      </div>
      <div className="h-[3px] bg-[var(--color-border)] rounded mt-1 overflow-hidden">
        <div
          className="h-full rounded transition-[width] duration-300"
          style={{ width: `${Math.min(100, Math.max(0, bar * 100))}%`, background: color }}
        />
      </div>
      <div className="text-[9px] text-[var(--color-ink-faint)] mt-1 leading-tight">{hint}</div>
    </div>
  );
}

const CONFIDENCE_COLOR: Record<AttackTechnique["confidence"], string> = {
  strong: "var(--color-ok)",
  good: "var(--color-sev-low)",
  medium: "var(--color-sev-medium)",
  stretch: "var(--color-sev-high)",
};

export function AttackChip({ attack }: { attack: AttackTechnique | null }) {
  if (!attack) {
    return (
      <span className="text-[10px] text-[var(--color-ink-faint)] border border-dashed border-[var(--color-border)] rounded px-1.5 py-0.5">
        no defensible ATT&amp;CK mapping
      </span>
    );
  }
  const color = CONFIDENCE_COLOR[attack.confidence];
  return (
    <a
      href={`https://attack.mitre.org/techniques/${attack.technique_id.split(".")[0]}/`}
      target="_blank"
      rel="noreferrer"
      title={attack.rationale}
      className="inline-flex items-center gap-1.5 text-[10px] border rounded px-1.5 py-0.5 hover:brightness-125"
      style={{ borderColor: "var(--color-border)" }}
    >
      <span className="text-[var(--color-ink-dim)]">{attack.tactic_id}</span>
      <span className="text-[var(--color-ink)]">{attack.technique_id}</span>
      <span className="text-[var(--color-ink-dim)] hidden md:inline">{attack.technique_name}</span>
      <span style={{ color }} className="uppercase tracking-wider">
        {attack.confidence}
      </span>
    </a>
  );
}

/**
 * SHAP contributions rendered as plain English first, numbers second.
 *
 * "sload = 1.4e6" means nothing on a triage queue. "outbound throughput 1.4 MB/s, typical for this
 * host is 12 KB/s" is actionable by a tier-1 analyst who is not an ML engineer, which is the whole
 * point of attaching them.
 */
export function ContributionList({ contributions }: { contributions: Contribution[] }) {
  if (!contributions.length) {
    return <p className="text-[var(--color-ink-faint)] text-[11px] px-3 py-2">No attributions.</p>;
  }
  const max = Math.max(...contributions.map((c) => Math.abs(c.shap_value)), 0.0001);
  return (
    <ul className="divide-y divide-[var(--color-border)]">
      {contributions.map((c) => {
        const toward = c.direction === "toward_attack";
        const color = toward ? "var(--color-sev-high)" : "var(--color-ok)";
        return (
          <li key={c.feature} className="px-3 py-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-[12px] text-[var(--color-ink)]">{c.feature}</span>
              <span className="text-[11px] tabular-nums" style={{ color }}>
                {c.shap_value >= 0 ? "+" : ""}
                {c.shap_value.toFixed(3)}
              </span>
            </div>
            <div className="h-[3px] bg-[var(--color-border)] rounded mt-1 overflow-hidden">
              <div
                className="h-full rounded"
                style={{ width: `${(Math.abs(c.shap_value) / max) * 100}%`, background: color }}
              />
            </div>
            {c.narrative && (
              <p className="text-[11px] text-[var(--color-ink-dim)] mt-1 leading-snug">{c.narrative}</p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="px-3 py-1.5 border-l border-[var(--color-border)] first:border-l-0">
      <div className="text-[9px] uppercase tracking-wider text-[var(--color-ink-faint)]">{label}</div>
      <div className="text-[15px] tabular-nums" style={{ color: tone ?? "var(--color-ink)" }}>
        {value}
      </div>
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-full grid place-items-center p-8 text-center">
      <p className="text-[var(--color-ink-faint)] text-[12px] max-w-sm leading-relaxed">{children}</p>
    </div>
  );
}

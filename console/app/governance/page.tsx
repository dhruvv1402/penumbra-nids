"use client";

/**
 * The governance page.
 *
 * Three claims this project makes about itself, each rendered from live state rather than from a
 * paragraph in a README:
 *
 *   NO BLOCKING     the API has no enforcement endpoint, and CI fails the build if one appears.
 *   RBAC            who may do what, read from the server's own permission table.
 *   AUDIT           an append-only hash chain, with its integrity verified on every load.
 *
 * The audit panel is the one worth watching. It does not display "tamper-evident" as a label; it
 * displays the result of actually re-walking the chain, and it turns red at the exact entry where
 * the chain breaks. A claim you can watch fail is worth more than one you have to believe.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  type AuditEntry,
  type Session,
  getAudit,
  getRbac,
  loadSession,
  saveSession,
} from "@/lib/api";
import { Empty, Panel, Stat } from "@/components/Primitives";

interface AuditState {
  chain_intact: boolean;
  broken_at?: number | null;
  n_entries: number;
  entries: AuditEntry[];
}

export default function Governance() {
  const [session, setSession] = useState<Session | null>(null);
  const [audit, setAudit] = useState<AuditState | null>(null);
  const [rbac, setRbac] = useState<{ matrix: string; roles: Record<string, string[]> } | null>(null);
  const [denied, setDenied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setSession(loadSession()), []);

  const load = useCallback(async (token: string) => {
    try {
      setRbac(await getRbac(token));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        saveSession(null);
        setSession(null);
        return;
      }
      setError(err instanceof Error ? err.message : String(err));
    }
    try {
      setAudit((await getAudit(token, 100)) as AuditState);
      setDenied(false);
    } catch (err) {
      // 403 here is the permission boundary working, not a failure. Say which one it is.
      if (err instanceof ApiError && err.isPermissionDenied) setDenied(true);
      else setError(err instanceof Error ? err.message : String(err));
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
          to view governance.
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
        <Link
          href="/drift"
          className="text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
        >
          drift
        </Link>
        <h1 className="text-[11px] tracking-[0.14em] uppercase">governance</h1>
        <p className="text-[11px] text-[var(--color-ink-dim)] ml-auto">
          signed in as {session.role}
        </p>
      </header>

      {error && (
        <div className="px-3 py-1.5 border-b border-[var(--color-border)] text-[11px] text-[var(--color-sev-high)]">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-2 p-2">
        <NoBlockPanel />
        <RbacPanel roles={rbac?.roles ?? null} role={session.role} />
        <AuditPanel audit={audit} denied={denied} role={session.role} />
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

function NoBlockPanel() {
  return (
    <Panel title="ADR-0001 — alert, never block">
      <div className="px-3 py-3 space-y-3 text-[11px] leading-relaxed">
        <p>
          Penumbra has no blocking code path. Not a disabled one, not a feature-flagged one — the
          API exposes no enforcement endpoint and the console could not call one if it did.
        </p>
        <div className="font-mono text-[10px] bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-2 space-y-0.5">
          <div>
            <span className="text-[var(--color-ink-dim)]">DvcAction &nbsp;&nbsp;&nbsp;&nbsp;=</span>{" "}
            &quot;Allow&quot; <span className="text-[var(--color-ink-dim)]">← we observed and did not act</span>
          </div>
          <div>
            <span className="text-[var(--color-ink-dim)]">ThreatConfidence =</span> 93{" "}
            <span className="text-[var(--color-ink-dim)]">← and we were confident</span>
          </div>
          <div>
            <span className="text-[var(--color-ink-dim)]">EventSeverity &nbsp;&nbsp;=</span>{" "}
            &quot;High&quot;
          </div>
        </div>
        <p>
          That is the decision expressed in Microsoft Sentinel&apos;s own ASIM vocabulary rather
          than in a README sentence — a reviewer can read it straight off the payload.
        </p>
        <p>
          It is enforced rather than promised: CI greps the whole tree for any enforcement call
          path and fails the build if one appears. The blast-radius argument is the reason — at a
          1% false-positive rate on a busy link, auto-blocking takes the business offline faster
          than any attacker would.
        </p>
        <p className="text-[var(--color-ink-dim)]">
          EU AI Act Article 14 requires high-risk systems to be designed so a person can oversee,
          interpret, override and halt them. NIST AI RMF 1.0 says the same under GOVERN 3.2. We
          alert; a human decides.
        </p>
      </div>
    </Panel>
  );
}

function RbacPanel({ roles, role }: { roles: Record<string, string[]> | null; role: string }) {
  if (!roles) {
    return (
      <Panel title="RBAC">
        <Empty>Permission table unavailable.</Empty>
      </Panel>
    );
  }
  const names = Object.keys(roles);
  const permissions = Array.from(new Set(Object.values(roles).flat())).sort();

  return (
    <Panel title="RBAC — who may do what" right={<Stat label="you" value={role} />}>
      <table className="w-full text-[11px]">
        <thead className="text-[var(--color-ink-dim)]">
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left font-normal px-3 py-1.5">permission</th>
            {names.map((name) => (
              <th
                key={name}
                className={`text-center font-normal px-3 py-1.5 ${name === role ? "text-[var(--color-ink)]" : ""}`}
              >
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {permissions.map((permission) => (
            <tr key={permission} className="border-b border-[var(--color-border)] last:border-0">
              <td className="px-3 py-1.5 font-mono text-[10px]">{permission}</td>
              {names.map((name) => (
                <td key={name} className="text-center px-3 py-1.5">
                  {roles[name].includes(permission) ? (
                    <span className="text-[var(--color-sev-low)]">●</span>
                  ) : (
                    <span className="text-[var(--color-ink-dim)] opacity-40">·</span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <Caption>
        <code>verdict:record</code> and <code>verdict:promote</code> are deliberately separate
        permissions. The analyst feedback loop is also a poisoning vector — a compromised analyst
        account could otherwise teach the model that its own traffic is benign — so a verdict needs
        a second, more privileged person before it can enter the training pool. MITRE ATLAS calls
        this one out; splitting the permission is the mitigation.
      </Caption>
    </Panel>
  );
}

function AuditPanel({
  audit,
  denied,
  role,
}: {
  audit: AuditState | null;
  denied: boolean;
  role: string;
}) {
  if (denied) {
    return (
      <Panel title="audit log" className="xl:col-span-2">
        <Empty>
          <span className="block mb-1">
            The <code>{role}</code> role cannot read the audit log.
          </span>
          <span className="text-[var(--color-ink-dim)]">
            This is the permission boundary working, not an error. Sign in as a role holding{" "}
            <code>audit:read</code>.
          </span>
        </Empty>
      </Panel>
    );
  }
  if (!audit) {
    return (
      <Panel title="audit log" className="xl:col-span-2">
        <Empty>No audit entries yet. Record a verdict on the queue page to create one.</Empty>
      </Panel>
    );
  }

  return (
    <Panel
      title="audit log — hash chain"
      className="xl:col-span-2"
      right={
        <span
          className={`text-[11px] ${audit.chain_intact ? "text-[var(--color-sev-low)]" : "text-[var(--color-sev-high)]"}`}
        >
          {audit.chain_intact
            ? `chain verified · ${audit.n_entries.toLocaleString()} entries`
            : `CHAIN BROKEN at entry ${audit.broken_at}`}
        </span>
      }
    >
      {audit.entries.length === 0 ? (
        <Empty>No entries yet.</Empty>
      ) : (
        <table className="w-full text-[11px]">
          <thead className="text-[var(--color-ink-dim)]">
            <tr className="border-b border-[var(--color-border)]">
              <th className="text-right font-normal px-3 py-1.5">#</th>
              <th className="text-left font-normal px-3 py-1.5">when</th>
              <th className="text-left font-normal px-3 py-1.5">actor</th>
              <th className="text-left font-normal px-3 py-1.5">action</th>
              <th className="text-left font-normal px-3 py-1.5">target</th>
              <th className="text-left font-normal px-3 py-1.5">digest</th>
            </tr>
          </thead>
          <tbody>
            {audit.entries.map((entry) => {
              const broken = audit.broken_at != null && entry.sequence >= audit.broken_at;
              return (
                <tr
                  key={entry.sequence}
                  className={`border-b border-[var(--color-border)] last:border-0 ${
                    broken ? "text-[var(--color-sev-high)]" : ""
                  }`}
                >
                  <td className="px-3 py-1.5 text-right tabular-nums text-[var(--color-ink-dim)]">
                    {entry.sequence}
                  </td>
                  <td className="px-3 py-1.5 whitespace-nowrap text-[var(--color-ink-dim)]">
                    {entry.timestamp.replace("T", " ").slice(0, 19)}
                  </td>
                  <td className="px-3 py-1.5">
                    {entry.actor}
                    <span className="text-[var(--color-ink-dim)]"> ({entry.role})</span>
                  </td>
                  <td className="px-3 py-1.5 font-mono text-[10px]">{entry.action}</td>
                  <td className="px-3 py-1.5 font-mono text-[10px] text-[var(--color-ink-dim)]">
                    {entry.target}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-[10px] text-[var(--color-ink-dim)]">
                    {entry.entry_hash.slice(0, 12)}…
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <Caption>
        Each entry commits to the digest of the one before it, so altering any past entry breaks
        every digest after it. The badge above is not a label — it is the result of re-walking the
        chain on this request, and it names the entry where verification failed. Every
        re-identification of a pseudonymised address and every suppression rule lands here with who
        and why.
      </Caption>
    </Panel>
  );
}

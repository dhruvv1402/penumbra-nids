"use client";

/**
 * Try the model: the deployed champions, run live (ADR-0006).
 *
 * Three ways in. Held-out test flows are the primary one: real flows the model never trained on,
 * drawn from a bundled pool, with the true label beside each verdict, so a visitor sees both what
 * it catches and what it gets wrong. A CSV in a model's schema and a pcap capture are the other
 * two. Every result is an alert, never a block; nothing here is stored, so the recorded demo
 * queue is untouched by anything a visitor uploads.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  type ScoreCatalogue,
  type ScoreResult,
  type ScoredRow,
  type Session,
  downloadTemplate,
  getScoreModels,
  guestLogin,
  loadSession,
  saveSession,
  scoreSample,
  scoreUpload,
} from "@/lib/api";
import { Empty, Panel, Stat, VerdictBadge } from "@/components/Primitives";

type Mode = "sample" | "csv" | "pcap";

const NAV = "text-[11px] tracking-[0.14em] uppercase text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]";
const pct = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(0)}%` : "—");

export default function TryTheModel() {
  const [session, setSession] = useState<Session | null>(null);
  const [catalogue, setCatalogue] = useState<ScoreCatalogue | null>(null);
  const [dataset, setDataset] = useState("unsw");
  const [mode, setMode] = useState<Mode>("sample");
  const [n, setN] = useState(200);
  const [share, setShare] = useState(0.1);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setSession(loadSession()), []);

  const fail = useCallback((err: unknown) => {
    if (err instanceof ApiError && err.status === 401) {
      saveSession(null);
      setSession(null);
      return;
    }
    let message = err instanceof Error ? err.message : String(err);
    try {
      message = JSON.parse(message).detail ?? message;
    } catch {
      /* plain text */
    }
    setError(message);
  }, []);

  useEffect(() => {
    if (!session) return;
    getScoreModels(session.token).then(setCatalogue).catch(fail);
  }, [session, fail]);

  const model = catalogue?.models.find((m) => m.dataset === (mode === "pcap" ? catalogue.pcap_dataset : dataset));

  const run = async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      if (mode === "sample") setResult(await scoreSample(session.token, dataset, n, share));
      else if (file) setResult(await scoreUpload(session.token, mode, file, dataset));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  if (!session) {
    return (
      <main className="h-screen grid place-items-center">
        <div className="text-center space-y-3">
          <p className="text-[12px] text-[var(--color-ink-dim)]">Run Penumbra&apos;s trained models on real traffic.</p>
          <button
            className="text-[12px] px-3 py-1.5 rounded border border-[var(--color-border)] hover:bg-[var(--color-panel-2)]"
            onClick={() =>
              guestLogin()
                .then((s) => {
                  saveSession(s);
                  setSession(s);
                })
                .catch(fail)
            }
          >
            View as guest
          </button>
          <p className="text-[11px] text-[var(--color-ink-faint)]">
            or{" "}
            <Link href="/console" className="underline">
              sign in on the queue page
            </Link>
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen flex flex-col">
      <header className="flex items-center gap-4 px-3 py-2 border-b border-[var(--color-border)] shrink-0">
        <Link href="/console" className={NAV}>
          ← queue
        </Link>
        <Link href="/evaluation" className={NAV}>
          evaluation
        </Link>
        <Link href="/governance" className={NAV}>
          governance
        </Link>
        <h1 className="text-[11px] tracking-[0.14em] uppercase">try the model</h1>
        <p className="text-[11px] text-[var(--color-ink-dim)] ml-auto">
          the deployed champions, live · alerts only, never a block · nothing you send is stored
        </p>
      </header>

      {error && (
        <div className="px-3 py-1.5 border-b border-[var(--color-border)] text-[11px] text-[var(--color-sev-high)]">{error}</div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[380px_1fr] gap-2 p-2">
        <Panel title="input">
          <div className="p-3 space-y-4 text-[12px]">
            <Segmented
              value={mode}
              onChange={(v) => {
                setMode(v as Mode);
                setResult(null);
                setFile(null);
              }}
              options={[
                ["sample", "held-out flows"],
                ["csv", "upload CSV"],
                ["pcap", "upload pcap"],
              ]}
            />

            {mode !== "pcap" && (
              <Field label="model">
                <Segmented
                  value={dataset}
                  onChange={(v) => {
                    setDataset(v);
                    setResult(null);
                  }}
                  options={(catalogue?.models ?? []).map((m) => [m.dataset, m.title] as [string, string])}
                />
              </Field>
            )}

            {model && (
              <p className="text-[11px] leading-relaxed text-[var(--color-ink-dim)]">
                {model.title} champion <code>{model.registry_version ?? "v001"}</code>: a random forest plus a benign-only
                novelty head, trained on {model.trained_rows?.toLocaleString()} flows,{" "}
                {model.calibration === "held_out"
                  ? "thresholds fitted on held-out rows"
                  : "thresholds fitted in-sample (the original fit; its held-out challenger was refused by the gate)"}
                , target{" "}
                {(model.target_fpr * 100).toFixed(0)}% false alarms.
                {mode === "sample" && <> Draws from: {model.test_pool}.</>}
              </p>
            )}

            {mode === "sample" && (
              <>
                <Field label="flows">
                  <Segmented
                    value={String(n)}
                    onChange={(v) => setN(Number(v))}
                    options={[50, 200, 500].map((x) => [String(x), String(x)] as [string, string])}
                  />
                </Field>
                <Field label="share that are attacks">
                  <Segmented
                    value={String(share)}
                    onChange={(v) => setShare(Number(v))}
                    options={(catalogue?.attack_shares ?? [0.05, 0.1, 0.5]).map(
                      (x) => [String(x), `${Math.round(x * 100)}%`] as [string, string],
                    )}
                  />
                </Field>
              </>
            )}

            {mode === "csv" && (
              <>
                <p className="text-[11px] text-[var(--color-ink-dim)] leading-relaxed">
                  One flow per row, with {model ? `the ${model.features.length} columns` : "the columns"} this model reads;
                  extra columns are ignored. Up to {catalogue?.max_csv_mb ?? 5} MB.
                </p>
                <button
                  className="text-[11px] underline text-[var(--color-ink-dim)]"
                  onClick={() => downloadTemplate(session.token, dataset).catch(fail)}
                >
                  download a template with real rows
                </button>
                <FileInput accept=".csv,text/csv" onFile={setFile} file={file} />
              </>
            )}

            {mode === "pcap" && (
              <>
                <p className="text-[11px] text-[var(--color-ink-dim)] leading-relaxed">
                  A .pcap or .pcapng up to {catalogue?.max_pcap_mb ?? 8} MB, assembled into flows and scored by the UNSW
                  champion; addresses come back pseudonymised. <strong>Only capture traffic you own.</strong> Out of the box a
                  network the model was not trained on mostly alerts - that is what <code>penumbra rebaseline</code> is for
                  (our own lab: ordinary flows alerting 100% → 4.9% after re-baselining).
                </p>
                <FileInput accept=".pcap,.pcapng,.cap" onFile={setFile} file={file} />
              </>
            )}

            <button
              disabled={busy || (mode !== "sample" && !file) || !catalogue}
              onClick={run}
              className="w-full text-[12px] py-2 rounded border border-[var(--color-ok)] text-[var(--color-ok)] disabled:opacity-40 hover:bg-[color-mix(in_srgb,var(--color-ok)_10%,transparent)]"
            >
              {busy ? "scoring…" : "score"}
            </button>
            {catalogue && (
              <p className="text-[10px] text-[var(--color-ink-faint)] leading-relaxed">
                Limits for your role: {catalogue.limits.calls_per_hour} calls an hour, {catalogue.limits.max_rows} flows a call.
                Scores are rounded and every call is audited - a public scorer is an oracle (ADR-0006).
              </p>
            )}
          </div>
        </Panel>

        <Results result={result} mode={mode} />
      </div>
    </main>
  );
}

function Results({ result, mode }: { result: ScoreResult | null; mode: Mode }) {
  const [open, setOpen] = useState<number | null>(null);
  const s = result?.summary;
  const rows = useMemo(() => result?.rows ?? [], [result]);
  if (!result || !s) {
    return (
      <Panel title="result">
        <Empty>Choose an input and press score. The verdicts appear here, with the truth beside them when it is known.</Empty>
      </Panel>
    );
  }
  const t = s.truth;
  return (
    <Panel title={`result · model ${s.model ?? "—"}`}>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--color-border)] border-b border-[var(--color-border)]">
        <Tile label="flows scored" value={s.flows.toLocaleString()} />
        <Tile label="reached an analyst" value={`${s.alerts.toLocaleString()} (${pct(s.alerts, s.flows)})`} />
        {t ? (
          <>
            <Tile label="attacks caught" value={`${t.attacks_reaching_an_analyst} / ${t.attacks} (${pct(t.attacks_reaching_an_analyst, t.attacks)})`} />
            <Tile label="benign flagged" value={`${t.benign_reaching_an_analyst} / ${t.benign} (${pct(t.benign_reaching_an_analyst, t.benign)})`} />
          </>
        ) : (
          <>
            <Tile label="known attack" value={String(s.by_verdict.KNOWN_ATTACK ?? 0)} />
            <Tile label="suspected novel" value={String(s.by_verdict.SUSPECTED_NOVEL ?? 0)} />
          </>
        )}
      </div>
      {t?.unseen_attacks != null && (
        <p className="px-3 py-2 text-[11px] border-b border-[var(--color-border)]">
          Attack types <strong>never seen in training</strong>: {t.unseen_reaching_an_analyst} of {t.unseen_attacks} reached an
          analyst.
        </p>
      )}
      <p className="px-3 py-1.5 text-[10px] text-[var(--color-ink-faint)] border-b border-[var(--color-border)]">
        {Object.entries(s.by_verdict)
          .map(([v, c]) => `${v.replace("_", " ").toLowerCase()} ${c}`)
          .join(" · ")}
        {s.rows_returned < s.flows && ` · showing ${s.rows_returned} of ${s.flows}, alerts first`}
        {" · "}&quot;reached an analyst&quot; counts uncertain verdicts too: the model declining to commit is routed to a human.
      </p>
      <div className="overflow-auto max-h-[70vh]">
        <table className="w-full text-[11px]">
          <thead className="text-[var(--color-ink-dim)] sticky top-0 bg-[var(--color-panel)]">
            <tr className="border-b border-[var(--color-border)]">
              <th className="text-left font-normal px-3 py-1.5">verdict</th>
              <th className="text-right font-normal px-2 py-1.5">p(attack)</th>
              <th className="text-right font-normal px-2 py-1.5">novelty</th>
              <th className="text-left font-normal px-2 py-1.5">head</th>
              <th className="text-left font-normal px-2 py-1.5">family</th>
              {mode === "pcap" ? (
                <th className="text-left font-normal px-2 py-1.5">src → dst</th>
              ) : (
                rows[0]?.truth && <th className="text-left font-normal px-2 py-1.5">truth</th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <Row key={`${r.row}-${i}`} r={r} open={open === i} onToggle={() => setOpen(open === i ? null : i)} />
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function Row({ r, open, onToggle }: { r: ScoredRow; open: boolean; onToggle: () => void }) {
  const alerted = r.verdict !== "BENIGN";
  const correct = r.truth ? (r.truth.label === 1) === alerted : null;
  return (
    <>
      <tr onClick={onToggle} className="border-b border-[var(--color-border)] cursor-pointer hover:bg-[var(--color-panel-2)]">
        <td className="px-3 py-1.5">
          <VerdictBadge verdict={r.verdict} />
        </td>
        <td className="px-2 py-1.5 text-right tabular-nums">{r.p_attack.toFixed(2)}</td>
        <td className="px-2 py-1.5 text-right tabular-nums">{r.novelty_percentile.toFixed(2)}</td>
        <td className="px-2 py-1.5 text-[var(--color-ink-dim)]">{r.head}</td>
        <td className="px-2 py-1.5">
          {r.family ?? <span className="text-[var(--color-ink-faint)]">—</span>}
          {r.attack_technique && <span className="ml-1 text-[10px] text-[var(--color-ink-dim)]">{r.attack_technique}</span>}
        </td>
        {r.src ? (
          <td className="px-2 py-1.5 font-mono text-[10px] text-[var(--color-ink-dim)]">
            {r.src} → {r.dst}
          </td>
        ) : (
          r.truth && (
            <td className="px-2 py-1.5">
              <span className={correct ? "text-[var(--color-ok)]" : "text-[var(--color-sev-high)]"}>{correct ? "✓" : "✗"}</span>{" "}
              {r.truth.label ? r.truth.family : "benign"}
              {r.truth.unseen_in_training && <span className="ml-1 text-[10px] text-[var(--color-novel)]">never in training</span>}
            </td>
          )
        )}
      </tr>
      {open && (
        <tr className="border-b border-[var(--color-border)] bg-[var(--color-panel-2)]">
          <td colSpan={6} className="px-3 py-2 text-[11px] text-[var(--color-ink-dim)]">
            {r.why.length ? (
              <span>
                Most influential features (global importance weighted by this flow):{" "}
                {r.why.map((w) => `${w.feature}${w.narrative ? ` (${w.narrative})` : ""}`).join(" · ")}
              </span>
            ) : (
              "No attribution for this flow."
            )}
            {r.conformal_set.length > 0 && <span className="block mt-1">Conformal prediction set: {r.conformal_set.join(", ")}</span>}
          </td>
        </tr>
      )}
    </>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[var(--color-panel)] px-3 py-2">
      <Stat label={label} value={value} />
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="text-[10px] tracking-[0.12em] uppercase text-[var(--color-ink-faint)]">{label}</p>
      {children}
    </div>
  );
}

function Segmented({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map(([v, label]) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className={`text-[11px] px-2 py-1 rounded border ${
            v === value
              ? "border-[var(--color-ink)] text-[var(--color-ink)]"
              : "border-[var(--color-border)] text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function FileInput({ accept, file, onFile }: { accept: string; file: File | null; onFile: (f: File | null) => void }) {
  return (
    <label className="block border border-dashed border-[var(--color-border)] rounded px-3 py-3 text-[11px] text-[var(--color-ink-dim)] cursor-pointer hover:border-[var(--color-ink-dim)]">
      <input type="file" accept={accept} className="hidden" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
      {file ? `${file.name} · ${(file.size / 2 ** 20).toFixed(1)} MB` : "choose a file"}
    </label>
  );
}

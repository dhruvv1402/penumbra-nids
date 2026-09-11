"""Command line entry point.

Deliberately the only entry point. There is no Makefile: this machine has no `make`, and a Typer
app declared in `[project.scripts]` works identically on every platform the team uses.

Every number that appears in the report or the slide deck is produced by a command here, so that
"regenerate everything" is one invocation rather than a remembered sequence of notebook cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from penumbra import __version__
from penumbra.config import settings
from penumbra.seeds import seed_everything

app = typer.Typer(
    name="penumbra",
    help="ML-native network detection and response. Alerts a SOC; never blocks traffic.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(name="data", help="Fetch and inspect datasets.", no_args_is_help=True)
app.add_typer(data_app)

console = Console()

DatasetName = Annotated[str, typer.Option("--dataset", "-d", help="unsw | nslkdd | cicids | attack")]


def _load(dataset: str, *, drop_artifacts: bool = False):
    """Resolve a dataset name to a loaded Dataset."""
    key = dataset.lower().replace("-", "").replace("_", "")
    if key in {"unsw", "unswnb15"}:
        from penumbra.data.loaders import unsw

        return unsw.load(drop_artifacts=drop_artifacts)
    if key in {"nslkdd", "nsl", "kdd"}:
        from penumbra.data.loaders import nsl_kdd

        return nsl_kdd.load(drop_artifacts=drop_artifacts)
    raise typer.BadParameter(f"unknown dataset {dataset!r}; expected unsw or nslkdd")


@app.command()
def version() -> None:
    """Print version and resolved paths."""
    s = settings()
    console.print(f"[bold]penumbra[/bold] {__version__}")
    console.print(f"  data      {s.data_root}")
    console.print(f"  artifacts {s.artifact_root}")
    console.print(f"  seed      {s.seed}")


# =================================================================================================
# data
# =================================================================================================


@data_app.command("fetch")
def data_fetch(
    dataset: Annotated[
        str, typer.Option("--dataset", "-d", help="unsw | nslkdd | cicids | attack | all")
    ] = "all",
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
) -> None:
    """Download datasets and verify them against data/manifest.json.

    Size is checked on download and SHA256 is pinned after the first fetch, because the failure
    mode that matters here is silent: CICIDS2017's official URL answers 200 and serves an HTML
    landing page, and one popular UNSW mirror has train and test swapped.
    """
    from penumbra.data import download, manifest

    if dataset == "all":
        specs = manifest.ALL_FILES
    else:
        specs = manifest.BY_DATASET.get(dataset.lower(), [])
        if not specs:
            raise typer.BadParameter(
                f"unknown dataset {dataset!r}; expected one of {sorted(manifest.BY_DATASET)} or 'all'"
            )

    def report(result: download.FetchResult) -> None:
        colour = {"downloaded": "green", "cached": "dim", "repaired": "yellow", "failed": "red"}[
            result.status
        ]
        console.print(f"[{colour}]{result.status:<11}[/{colour}] {result.spec.filename:<40} {result.detail}")

    results = download.fetch_all(specs, force=force, on_result=report)
    failed = [r for r in results if not r.ok]
    if failed:
        console.print(f"\n[red]{len(failed)} file(s) failed.[/red]")
        raise typer.Exit(1)
    console.print(f"\n[green]{len(results)} file(s) verified.[/green]")


@data_app.command("info")
def data_info(dataset: DatasetName = "unsw") -> None:
    """Show shape, prevalence and class balance for a dataset."""
    ds = _load(dataset)
    console.print(ds.describe())

    table = Table(title=f"{ds.name} - class balance", show_edge=False)
    table.add_column("family")
    table.add_column("train", justify="right")
    table.add_column("test", justify="right")
    tr, te = ds.family_counts("train"), ds.family_counts("test")
    for fam in tr.index:
        table.add_row(str(fam), f"{tr.get(fam, 0):,}", f"{te.get(fam, 0):,}")
    console.print(table)


# =================================================================================================
# audit
# =================================================================================================


@app.command()
def audit(
    dataset: DatasetName = "unsw",
    save: Annotated[bool, typer.Option("--save/--no-save", help="Write JSON to artifacts/reports/.")] = True,
) -> None:
    """Audit a dataset for leakage before any model is trained.

    Four checks: single-feature AUC, leaky values, train/test overlap, and negative controls.

    This runs first, and deliberately so - it constrains what we are allowed to claim afterwards.
    A model reporting 0.98 AUC on a dataset where one feature value fingerprints 22% of the benign
    class has not learned what its author thinks it has.
    """
    seed_everything()
    from penumbra.data import audit as audit_mod

    ds = _load(dataset)
    console.print(f"[dim]auditing {ds.name} ({len(ds.X_train):,} train rows)...[/dim]\n")
    report = audit_mod.run_audit(ds)
    console.print(report.summary())

    if save:
        settings().ensure_dirs()
        out = settings().report_dir / f"audit_{dataset.lower()}.json"
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        console.print(f"\n[dim]written to {out}[/dim]")


# =================================================================================================
# placeholders - implemented in later phases, declared here so `--help` shows the intended shape
# =================================================================================================


@app.command()
def train(
    dataset: DatasetName = "unsw",
    model: Annotated[str, typer.Option("--model", "-m", help="logreg | rf | xgb")] = "xgb",
    drop_artifacts: Annotated[
        bool, typer.Option("--drop-artifacts", help="Exclude features the audit quarantined.")
    ] = False,
) -> None:
    """Train one known-threat model and print its honest metrics."""
    seed_everything()
    from penumbra.eval import runner

    ds = _load(dataset, drop_artifacts=drop_artifacts)
    console.print(f"[dim]training {model} on {ds.name} ({len(ds.X_train):,} rows)...[/dim]")
    result = runner.train_and_score(model, ds)
    console.print(f"\n[bold]{model}[/bold]  fit in {result.train_seconds:.1f}s")
    console.print(result.metrics.summary())
    console.print(f"\n  ROC-AUC 95% CI  {result.roc_auc_ci}")
    console.print(f"  recall  95% CI  {result.recall_ci}")


@app.command("eval")
def eval_cmd(
    dataset: DatasetName = "unsw",
    models: Annotated[str, typer.Option("--models", help="Comma-separated.")] = "logreg,rf,xgb",
    boot: Annotated[int, typer.Option("--boot", help="Bootstrap resamples for CIs.")] = 500,
    per_family: Annotated[bool, typer.Option("--per-family/--no-per-family")] = True,
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Full honest evaluation: every model, with and without quarantined features.

    Reports the prevalence-invariant pair (TPR/FPR) as primary, annotates PR-AUC with the
    prevalence it was computed at, attaches stratified bootstrap CIs, and projects the measured
    rates onto realistic deployment base rates.
    """
    seed_everything()
    from penumbra.eval import runner

    names = tuple(m.strip() for m in models.split(",") if m.strip())
    result = runner.evaluate_dataset(
        dataset,
        models=names,
        n_boot=boot,
        on_progress=lambda msg: console.print(f"[dim]  {msg}[/dim]"),
    )
    console.print(runner.format_result(result))

    ds = _load(dataset)
    console.print(runner.compare_models(result, ds.y_test.to_numpy()))
    if per_family:
        console.print(runner.per_family_report(ds))

    if save:
        out = runner.save(result)
        console.print(f"\n[dim]written to {out}[/dim]")


@app.command()
def ablate(
    dataset: DatasetName = "unsw",
    model: Annotated[str, typer.Option("--model", "-m")] = "rf",
    leakage_demo: Annotated[
        bool, typer.Option("--leakage-demo", help="Only run the SMOTE-before-split demonstration.")
    ] = False,
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Class-imbalance ablation, evaluated on the natural distribution.

    With --leakage-demo, runs the same pipeline correctly and leakily on a rare family and prints
    both numbers. The leaky version does not error; it reports a better result, which is why the
    comparison is worth printing.
    """
    seed_everything()
    import json as _json

    from penumbra.imbalance import ablation

    ds = _load(dataset)

    settings().ensure_dirs()

    if leakage_demo:
        leak = ablation.leakage_demonstration(ds, model=model)
        console.print(leak.summary())
        if save:
            out = settings().report_dir / "smote_leakage.json"
            out.write_text(_json.dumps(leak.to_dict(), indent=2), encoding="utf-8")
            console.print(f"[dim]written to {out}[/dim]")
        return

    console.print(f"[dim]ablating {ds.name} with {model}...[/dim]")
    table = ablation.run(ds, model=model, on_progress=lambda n: console.print(f"[dim]  {n}[/dim]"))
    console.print(table.summary())
    if save:
        out = settings().report_dir / f"ablation_{dataset.lower()}.json"
        out.write_text(_json.dumps(table.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")


@app.command()
def loafo(dataset: DatasetName = "unsw") -> None:
    """Leave-One-Attack-Family-Out at a matched false-positive budget.

    On nslkdd this runs the natural experiment instead: the 17 attack types present in KDDTest+ but
    absent from KDDTrain+, swept across operating points.
    """
    seed_everything()
    import json as _json

    from penumbra.eval import loafo as loafo_mod

    settings().ensure_dirs()

    if dataset.lower().startswith("nsl"):
        curve = loafo_mod.unseen_curve()
        console.print(curve.summary())
        out = settings().report_dir / "unseen17_curve.json"
        out.write_text(_json.dumps(curve.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")
        return

    ds = _load(dataset)
    matrix = loafo_mod.run(
        ds, budget_fpr=0.02, on_progress=lambda f: console.print(f"[dim]  holding out {f}[/dim]")
    )
    console.print(matrix.summary())
    out = settings().report_dir / f"loafo_{dataset.lower()}.json"
    out.write_text(_json.dumps(matrix.to_dict(), indent=2), encoding="utf-8")
    console.print(f"[dim]written to {out}[/dim]")


@app.command()
def fit(
    dataset: DatasetName = "unsw",
    model: Annotated[str, typer.Option("--model", "-m")] = "rf",
    target_fpr: Annotated[float, typer.Option("--fpr", help="Target false-positive rate.")] = 0.01,
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Fit the full two-head detector and save it.

    Thresholds are fitted on held-out benign traffic, never on test labels, so the saved operating
    point is one that could actually be chosen at deployment time.
    """
    seed_everything()
    from penumbra.models.detector import PenumbraDetector

    ds = _load(dataset)
    console.print(f"[dim]fitting on {ds.name} ({len(ds.X_train):,} rows)...[/dim]")
    det = PenumbraDetector(target_fpr=target_fpr).fit(
        ds, model_name=model, on_progress=lambda m: console.print(f"[dim]  {m}[/dim]")
    )
    path = det.save(out or settings().model_dir / dataset.lower())
    console.print(f"\n[green]saved[/green] {path}")
    if det.metadata:
        console.print(
            f"  supervised threshold {det.metadata.supervised_threshold:.4f}  "
            f"novelty threshold {det.metadata.novelty_threshold:.4f}"
        )


@app.command()
def replay(
    dataset: DatasetName = "unsw",
    rows: Annotated[int, typer.Option("--rows", help="Cap the stream length.")] = 5000,
    delay: Annotated[float, typer.Option("--delay", help="Seconds between batches.")] = 0.0,
    ingest: Annotated[bool, typer.Option("--ingest", help="POST to a running API.")] = False,
    api: Annotated[str, typer.Option("--api")] = "http://127.0.0.1:8000",
    inject_drift: Annotated[
        str | None, typer.Option("--inject-drift", help="abrupt | gradual | seasonal | evasion")
    ] = None,
    fixture_out: Annotated[Path | None, typer.Option("--write-fixture")] = None,
) -> None:
    """Stream a dataset through the detector as if it were live traffic."""
    seed_everything()
    from penumbra.models.detector import PenumbraDetector
    from penumbra.replay import engine

    model_dir = settings().model_dir / dataset.lower()
    if not (model_dir / "detector.joblib").exists():
        console.print()
        console.print(f"[red]No detector at {model_dir}.[/red] Run `penumbra fit -d {dataset}` first.")
        raise typer.Exit(1)

    det = PenumbraDetector.load(model_dir)
    ds = _load(dataset)
    X = ds.X_test

    if inject_drift:
        from penumbra.drift import injector

        plan = injector.scenario(inject_drift, min(rows, len(X)))
        X = injector.inject(X.head(rows), plan, numeric_features=list(X.select_dtypes("number").columns))
        console.print(f"[yellow]drift injected[/yellow] scenario={inject_drift} at row {plan.change_point:,}")

    client = None
    if ingest:
        client = engine.IngestClient(api)
        if not client.login("senior", "senior"):
            console.print(
                "[red]Could not authenticate to the API.[/red] Start it with PENUMBRA_ALLOW_DEMO_USERS=1."
            )
            raise typer.Exit(1)

    def on_batch(start: int, alerts: list, stats) -> None:
        if client and alerts:
            client.send(alerts)
        console.print(
            f"[dim]  {start + len(alerts):>7,} scored  {stats.alerts_emitted:>6,} alerts[/dim]",
            end="\n",
        )

    alerts, stats = engine.replay(
        det,
        X,
        config=engine.ReplayConfig(delay=delay, max_rows=rows),
        dataset=dataset,
        on_batch=on_batch,
    )

    console.print()
    incidents, skipped = engine.correlate_if_possible(alerts, dataset=dataset)
    stats.incidents = len(incidents)
    console.print(stats.summary())
    if skipped:
        console.print(f"\n[yellow]correlation skipped:[/yellow] {skipped}")

    if fixture_out:
        path = engine.write_fixture(alerts, incidents, fixture_out)
        console.print(f"\n[dim]fixture written to {path}[/dim]")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    demo_users: Annotated[bool, typer.Option("--demo-users/--no-demo-users")] = True,
) -> None:
    """Run the API.

    The console proxies /api/* here, so `penumbra serve` plus `npm run dev` in console/ is the whole
    demo stack.
    """
    import os

    import uvicorn

    if demo_users:
        os.environ["PENUMBRA_ALLOW_DEMO_USERS"] = "1"
    console.print(f"[dim]API on http://{host}:{port}  docs at /docs[/dim]")
    uvicorn.run("penumbra.api.app:app", host=host, port=port, log_level="info")


@app.command()
def rules(
    dataset: DatasetName = "unsw",
    min_precision: Annotated[
        float, typer.Option("--min-precision", help="Held-out precision floor a rule must clear.")
    ] = 0.98,
    min_support: Annotated[int, typer.Option("--min-support", help="Minimum leaf size.")] = 50,
    max_conditions: Annotated[
        int, typer.Option("--max-conditions", help="Readability cap on rule length.")
    ] = 4,
    sigma: Annotated[bool, typer.Option("--sigma/--no-sigma", help="Write Sigma files too.")] = True,
) -> None:
    """Mine KQL and Sigma detection rules out of a forest, validated on held-out data.

    The model mines the signature; the SIEM enforces it. Rules run without Python, without the
    model and without a GPU, which is what makes them adoptable by a team that already has a SIEM.

    Two arms are mined and both are reported: one with the quarantined testbed features available
    and one without. The gap between them is how much of a "validated detection" was the testbed.
    """
    seed_everything()
    from penumbra.rules import runner

    ds = _load(dataset)
    key = dataset.lower().replace("-", "").replace("_", "")
    report = runner.run(
        ds,
        dataset_key=key,
        min_support=min_support,
        min_purity=0.98,
        max_conditions=max_conditions,
        min_precision=min_precision,
        on_progress=lambda m: console.print(f"[dim]  {m}[/dim]"),
    )

    console.print()
    console.print(report.summary())

    paths = runner.write_artifacts(
        report,
        report_path=settings().report_dir / f"mined_rules_{key}.json",
        kql_path=Path("sentinel") / "Analytic Rules" / f"PenumbraMinedRules_{key}.kql",
        sigma_dir=(Path("sentinel") / "Sigma" / key) if sigma else None,
    )
    console.print(f"[green]wrote[/green] {paths['report']}")
    console.print(f"[green]wrote[/green] {paths['kql']}")
    if paths["sigma_written"]:
        console.print(f"[green]wrote[/green] {len(paths['sigma_written'])} Sigma rules")
    console.print(
        f"[dim]{paths['sigma_skipped']} rules were not emitted as Sigma: their conditions "
        f"reference flow features Sigma's log-based taxonomy does not define.[/dim]"
    )


@app.command()
def sequence(
    fpr: Annotated[float, typer.Option("--fpr", help="Matched false-positive budget.")] = 0.01,
    window: Annotated[int, typer.Option("--window", "-k", help="Flows per causal window.")] = 16,
    epochs: Annotated[int, typer.Option("--epochs")] = 12,
    rows_per_day: Annotated[
        int | None,
        typer.Option(
            "--rows-per-day",
            help="Cap rows read per CICIDS day. Smoke tests only - it reads each day's HEAD, "
            "which on Monday is 371k flows with zero attacks.",
        ),
    ] = None,
    train_stride: Annotated[
        int, typer.Option("--train-stride", help="Materialise every Nth training window.")
    ] = 4,
    test_stride: Annotated[int, typer.Option("--test-stride")] = 3,
) -> None:
    """Run E6: does sequence context buy recall per-flow features cannot?

    Three arms on identical rows at a matched benign-flag budget - per-flow, per-flow plus causal
    entity-graph features, and the CNN/BiGRU sequence head. The hypothesis is registered in
    docs/EXPERIMENTS.md and was committed before this ever ran.

    CICIDS2017 only: it is the one dataset here carrying source IPs and timestamps.
    """
    seed_everything()
    from penumbra.data.loaders import cicids
    from penumbra.eval import sequence_experiment

    console.print("[dim]loading CICIDS2017 (temporal split, Mon-Wed / Thu-Fri)...[/dim]")
    ds = cicids.load(nrows_per_day=rows_per_day)
    console.print(f"[dim]  {len(ds.X_train):,} train / {len(ds.X_test):,} test flows[/dim]")

    result = sequence_experiment.run(
        ds,
        fpr_budget=fpr,
        window_length=window,
        train_stride=train_stride,
        test_stride=test_stride,
        epochs=epochs,
        on_progress=lambda m: console.print(f"[dim]  {m}[/dim]"),
    )

    console.print()
    console.print(result.summary())
    out = sequence_experiment.write_report(result, settings().report_dir / "sequence_cicids.json")
    console.print(f"[green]wrote[/green] {out}")


@app.command()
def adversarial(
    dataset: DatasetName = "unsw",
    efforts: Annotated[
        str, typer.Option("--efforts", help="Comma-separated attacker effort levels.")
    ] = "0,0.5,1,2,4,9",
    rows: Annotated[int, typer.Option("--rows", help="Attack flows to perturb.")] = 20_000,
) -> None:
    """Measure detection decay under problem-space constrained evasion.

    Two attacks are run on the same flows: one that respects what an attacker can physically do
    (pad bytes up, stretch duration, never un-send a packet, recompute every derived feature), and
    the unconstrained feature-space attack most of the literature reports. The gap between them is
    the result.

    Needs a fitted detector: run `penumbra fit -d <dataset>` first.
    """
    seed_everything()
    from penumbra.adversarial import evasion
    from penumbra.models.detector import PenumbraDetector

    model_dir = settings().model_dir / dataset.lower()
    if not (model_dir / "detector.joblib").exists():
        console.print()
        console.print(f"[red]No detector at {model_dir}.[/red] Run `penumbra fit -d {dataset}` first.")
        raise typer.Exit(1)

    det = PenumbraDetector.load(model_dir)
    ds = _load(dataset)
    key = dataset.lower().replace("-", "").replace("_", "")

    y_test = np.asarray(ds.y_test).astype(int)
    attack_rows = np.flatnonzero(y_test == 1)[:rows]
    X_attacks = ds.X_test.iloc[attack_rows].reset_index(drop=True)
    families = pd.Series(ds.fam_test).reset_index(drop=True).iloc[attack_rows].reset_index(drop=True)
    benign = ds.X_test.iloc[np.flatnonzero(y_test == 0)[:rows]].reset_index(drop=True)

    threshold = float(det.metadata.supervised_threshold) if det.metadata else 0.5
    console.print(
        f"[dim]{len(X_attacks):,} attack flows, detector threshold {threshold:.4f} "
        f"(fitted on held-out benign, not on these rows)[/dim]"
    )

    levels = tuple(float(v) for v in efforts.split(",") if v.strip())
    report = evasion.evaluate(
        lambda frame: det.score(frame)["p_attack"].to_numpy(),
        X_attacks,
        dataset=key,
        threshold=threshold,
        benign_reference=benign,
        families=families,
        efforts=levels,
        on_progress=lambda m: console.print(f"[dim]  {m}[/dim]"),
    )

    console.print()
    console.print(report.summary())
    out = evasion.write_report(report, settings().report_dir / f"adversarial_{key}.json")
    console.print(f"[green]wrote[/green] {out}")


@app.command()
def pcap(
    capture: Annotated[Path, typer.Argument(help="Path to a .pcap or .pcapng file.")],
    score: Annotated[
        bool, typer.Option("--score/--no-score", help="Run the detector over the assembled flows.")
    ] = True,
    model: Annotated[str, typer.Option("--model", help="Which fitted detector to score with.")] = "unsw",
    out: Annotated[Path | None, typer.Option("--out", help="Write the assembled flows as CSV.")] = None,
) -> None:
    """Assemble a packet capture into flow features, and optionally score it.

    This is what removes the "it only works on a CSV someone else prepared" objection: point it at
    a capture and the detector runs on traffic.

    Only capture from a network you own. See docs/ETHICS_SCOPE.md - this package reads capture
    files and never transmits a packet.
    """
    seed_everything()
    from penumbra.pcap import assemble as assembler

    if not capture.exists():
        console.print(f"[red]No such capture:[/red] {capture}")
        raise typer.Exit(1)

    console.print(f"[dim]assembling flows from {capture.name}...[/dim]")
    flows = assembler.assemble(capture)
    console.print()
    console.print(assembler.summary(flows))

    if out is not None and not flows.empty:
        out.parent.mkdir(parents=True, exist_ok=True)
        flows.to_csv(out, index=False)
        console.print()
        console.print(f"[green]wrote[/green] {out}")

    if not score or flows.empty:
        return

    model_dir = settings().model_dir / model
    if not (model_dir / "detector.joblib").exists():
        console.print()
        console.print(
            f"[yellow]No detector at {model_dir}[/yellow] - flows assembled but not scored. "
            f"Run `penumbra fit -d {model}` first."
        )
        return

    from penumbra.alerts.builder import alerts_from_scores
    from penumbra.models.detector import PenumbraDetector

    det = PenumbraDetector.load(model_dir)
    scored = det.score(flows)
    alerts = alerts_from_scores(det, flows, scored, dataset=model)
    console.print()
    console.print(
        f"  {len(flows):,} flows scored -> [bold]{len(alerts):,} alerts[/bold] "
        f"({len(alerts) / max(len(flows), 1):.1%})"
    )
    for alert in alerts[:5]:
        console.print(f"    {alert.verdict:18} p_attack={alert.p_attack:.3f} priority={alert.priority}")


@app.command("loadtest")
def loadtest(
    dataset: DatasetName = "unsw",
    rows: Annotated[int, typer.Option("--rows", help="Flows to score.")] = 20_000,
    api: Annotated[
        str | None, typer.Option("--api", help="Also measure round-trip latency against a running API.")
    ] = None,
) -> None:
    """Measure scoring throughput and latency, and convert it honestly.

    Reports flows/s and p99 per batch across several batch sizes, then converts the best figure to
    monitored Mbps with the assumption stated. Flows per second is not link speed and the output
    says so - a link carrying long-lived connections produces far fewer flows per second than one
    carrying a scan.
    """
    seed_everything()
    from penumbra.eval import loadtest as lt
    from penumbra.models.detector import PenumbraDetector

    model_dir = settings().model_dir / dataset.lower()
    if not (model_dir / "detector.joblib").exists():
        console.print(f"[red]No detector at {model_dir}.[/red] Run `penumbra fit -d {dataset}` first.")
        raise typer.Exit(1)

    det = PenumbraDetector.load(model_dir)
    ds = _load(dataset)
    mean_bytes, median_bytes = lt.estimate_flow_bytes(ds.X_test)
    report = lt.measure_scoring(
        det,
        ds.X_test,
        max_rows=rows,
        dataset=ds.name,
        mean_flow_bytes=mean_bytes,
        on_progress=lambda m: console.print(f"[dim]  {m}[/dim]"),
    )
    report.median_flow_bytes = median_bytes

    console.print()
    console.print(report.summary())

    if api:
        from penumbra.eval.loadtest import measure_api

        console.print(f"[dim]measuring API round trip against {api}...[/dim]")
        try:
            import httpx

            resp = httpx.post(
                f"{api}/auth/login", json={"username": "analyst", "password": "analyst"}, timeout=10.0
            )
            resp.raise_for_status()
            latency = measure_api(api, resp.json()["access_token"])
            console.print(
                f"  API GET /alerts  p50 {latency.p50:.1f} ms  p95 {latency.p95:.1f} ms  "
                f"p99 {latency.p99:.1f} ms"
            )
            report.notes.append(
                f"API round trip p50/p95/p99 = {latency.p50:.1f}/{latency.p95:.1f}/{latency.p99:.1f} ms, "
                "measured sequentially against a single-process uvicorn."
            )
        except Exception as exc:  # noqa: BLE001 - the API being down must not lose the scoring result
            console.print(f"[yellow]API measurement skipped:[/yellow] {type(exc).__name__}: {exc}")

    out = lt.write_report(report, settings().report_dir / f"loadtest_{dataset.lower()}.json")
    console.print(f"[green]wrote[/green] {out}")


@app.command("reproduce-all")
def reproduce_all(
    skip_slow: Annotated[
        bool, typer.Option("--skip-slow", help="Omit the steps measured in tens of minutes.")
    ] = False,
    stop_on_error: Annotated[bool, typer.Option("--stop-on-error/--keep-going")] = False,
) -> None:
    """Regenerate every number in docs/EVALUATION.md from scratch.

    Each step is a real invocation of the same code path the documented run used - nothing here
    re-reads a cached report. A step whose inputs are absent is reported as SKIPPED with the reason,
    never silently passed: "12 of 14 regenerated, CICIDS2017 not downloaded" is a useful answer and
    a green tick that hid two missing datasets is not.
    """
    import time

    from penumbra.data import manifest

    def have(dataset: str) -> bool:
        """Is the dataset actually on disk? A missing one is SKIPPED, never quietly OK."""
        specs = manifest.BY_DATASET.get(dataset, [])
        root = settings().raw_dir
        return bool(specs) and all((root / s.dataset / s.filename).exists() for s in specs)

    def fitted(dataset: str) -> bool:
        return (settings().model_dir / dataset / "detector.joblib").exists()

    def have_keras() -> bool:
        try:
            import keras  # noqa: F401
        except ImportError:
            return False
        return True

    # (label, callable, slow?, guard) - the guard returns None to run or a reason to skip.
    steps: list[tuple[str, Any, bool, Any]] = [
        # The audits are 3-9 minutes each: they fit a one-feature stump per column and sweep every
        # value of every categorical. Slow enough to belong behind --skip-slow.
        (
            "audit unsw",
            lambda: audit("unsw", save=True),
            True,
            lambda: None if have("unsw") else "unsw not fetched",
        ),
        (
            "audit nslkdd",
            lambda: audit("nslkdd", save=True),
            True,
            lambda: None if have("nslkdd") else "nslkdd not fetched",
        ),
        (
            "eval unsw",
            lambda: eval_cmd("unsw", save=True),
            True,
            lambda: None if have("unsw") else "unsw not fetched",
        ),
        (
            "eval nslkdd",
            lambda: eval_cmd("nslkdd", save=True),
            True,
            lambda: None if have("nslkdd") else "nslkdd not fetched",
        ),
        (
            "ablate unsw",
            lambda: ablate("unsw", save=True),
            True,
            lambda: None if have("unsw") else "unsw not fetched",
        ),
        ("loafo unsw", lambda: loafo("unsw"), True, lambda: None if have("unsw") else "unsw not fetched"),
        (
            "loafo nslkdd",
            lambda: loafo("nslkdd"),
            True,
            lambda: None if have("nslkdd") else "nslkdd not fetched",
        ),
        ("rules unsw", lambda: rules("unsw"), True, lambda: None if have("unsw") else "unsw not fetched"),
        (
            "rules nslkdd",
            lambda: rules("nslkdd"),
            True,
            lambda: None if have("nslkdd") else "nslkdd not fetched",
        ),
        ("fit unsw", lambda: fit("unsw"), True, lambda: None if have("unsw") else "unsw not fetched"),
        (
            "adversarial unsw",
            lambda: adversarial("unsw"),
            False,
            # Depends on `fit unsw` above. Guarded on the artifact rather than assumed, so a
            # --skip-slow run reports SKIPPED with the reason instead of FAILED with an exit code.
            lambda: (
                None
                if have("unsw") and fitted("unsw")
                else (
                    "unsw not fetched"
                    if not have("unsw")
                    else "no fitted detector - run `penumbra fit -d unsw`"
                )
            ),
        ),
        (
            "sequence cicids",
            lambda: sequence(),
            True,
            lambda: (
                None
                if have("cicids") and have_keras()
                else ("cicids not fetched" if not have("cicids") else "the `dl` extra is not installed")
            ),
        ),
    ]

    outcomes: list[tuple[str, str, float, str]] = []
    for label, run_step, slow, guard in steps:
        if slow and skip_slow:
            outcomes.append((label, "SKIPPED", 0.0, "--skip-slow"))
            continue
        reason = guard()
        if reason:
            outcomes.append((label, "SKIPPED", 0.0, reason))
            continue

        console.rule(f"[bold]{label}")
        started = time.monotonic()
        try:
            run_step()
            outcomes.append((label, "OK", time.monotonic() - started, ""))
        except Exception as exc:  # noqa: BLE001 - one broken step must not hide the other eleven
            # typer.Exit carries only a status code, so its repr is "Exit: 1" - useless in a table
            # whose whole job is saying why something did not run.
            note = (
                f"exited {getattr(exc, 'exit_code', 1)} - see the step's own output"
                if isinstance(exc, typer.Exit)
                else f"{type(exc).__name__}: {exc}"
            )
            outcomes.append((label, "FAILED", time.monotonic() - started, note))
            if stop_on_error:
                break

    console.print()
    table = Table(title="reproduce-all", show_lines=False)
    table.add_column("step")
    table.add_column("result")
    table.add_column("seconds", justify="right")
    table.add_column("note")
    for label, status, seconds, note in outcomes:
        colour = {"OK": "green", "FAILED": "red", "SKIPPED": "yellow"}[status]
        table.add_row(label, f"[{colour}]{status}[/{colour}]", f"{seconds:.1f}" if seconds else "", note)
    console.print(table)

    failed = [o for o in outcomes if o[1] == "FAILED"]
    skipped = [o for o in outcomes if o[1] == "SKIPPED"]
    console.print()
    console.print(
        f"{len(outcomes) - len(failed) - len(skipped)} regenerated, "
        f"{len(skipped)} skipped, {len(failed)} failed."
    )
    console.print(f"[dim]reports in {settings().report_dir}[/dim]")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

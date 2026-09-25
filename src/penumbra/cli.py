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
    from_fixture: Annotated[
        Path | None,
        typer.Option("--from-fixture", help="Replay pre-scored alerts. No model, no dataset."),
    ] = None,
) -> None:
    """Stream a dataset through the detector as if it were live traffic."""
    seed_everything()
    from penumbra.replay import engine

    if from_fixture is not None:
        _replay_fixture(from_fixture, ingest=ingest, api=api, delay=delay, rows=rows)
        return

    det = _deployed_detector(dataset)
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


def _registry(dataset: str):
    from penumbra.models.registry import ModelRegistry

    return ModelRegistry(settings().artifact_root / "registry", dataset)


def _deployed_detector(dataset: str):
    """The registry's champion if there is one (hash-verified before unpickling), else the model
    `penumbra fit` wrote."""
    from penumbra.models.detector import PenumbraDetector
    from penumbra.models.registry import TamperedArtifact

    reg = _registry(dataset)
    try:
        champion = reg.champion()
        det = reg.load() if champion else None
    except TamperedArtifact as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if det is not None:
        console.print(f"[dim]champion {reg.champion()} (hashes verified)[/dim]")
        return det

    model_dir = settings().model_dir / dataset.lower()
    if not (model_dir / "detector.joblib").exists():
        console.print()
        console.print(f"[red]No detector at {model_dir}.[/red] Run `penumbra fit -d {dataset}` first.")
        raise typer.Exit(1)
    return PenumbraDetector.load(model_dir)


def _replay_fixture(path: Path, *, ingest: bool, api: str, delay: float, rows: int) -> None:
    """The demo-day fallback: push a frozen, already-scored run into the API.

    Nothing here imports the detector or a loader, so it works with no trained model, no dataset on
    disk and no network beyond loopback. The only moving part left is the API itself.
    """
    import time

    from penumbra.replay import engine

    if not path.exists():
        console.print(f"[red]No fixture at {path}.[/red] Write one with `penumbra replay --write-fixture`.")
        raise typer.Exit(1)

    alerts, incidents = engine.read_fixture(path)
    alerts = alerts[:rows]
    console.print(f"[dim]fixture {path}: {len(alerts):,} alerts, {len(incidents):,} incidents[/dim]")
    if not ingest:
        console.print("Nothing sent. Add --ingest to push into a running API.")
        return

    client = engine.IngestClient(api)
    if not client.login("senior", "senior"):
        console.print(
            "[red]Could not authenticate to the API.[/red] Start it with PENUMBRA_ALLOW_DEMO_USERS=1."
        )
        raise typer.Exit(1)

    sent = 0
    for start in range(0, len(alerts), 250):
        sent += client.send(alerts[start : start + 250])
        console.print(f"[dim]  {sent:>7,} ingested[/dim]")
        if delay:
            time.sleep(delay)
    if sent < len(alerts):
        # Incidents derive their segment from their stored alerts; posting them without those
        # alerts would store them unscoped. Better none than wrongly visible.
        console.print(
            f"[yellow]{len(alerts) - sent:,} alerts were not accepted by the API; incidents not sent.[/yellow]"
        )
        raise typer.Exit(1)
    shipped = {a.alert_id for a in alerts}
    ready = [i for i in incidents if set(i.get("alert_ids", [])) & shipped]
    n_inc = client.send_incidents(ready)
    console.print(f"{sent:,} alerts and {n_inc:,} incidents ingested from fixture.")
    if len(ready) < len(incidents):
        console.print(
            f"[dim]{len(incidents) - len(ready):,} incidents skipped: none of their alerts were within --rows.[/dim]"
        )


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

    def have_onnx() -> bool:
        import importlib.util

        return (
            importlib.util.find_spec("skl2onnx") is not None
            and importlib.util.find_spec("onnxruntime") is not None
        )

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
            "loadtest unsw",
            lambda: loadtest("unsw"),
            False,
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
        (
            "calibrate unsw",
            lambda: calibrate_cmd("unsw"),
            False,
            lambda: None if have("unsw") else "unsw not fetched",
        ),
        (
            "calibrate nslkdd",
            lambda: calibrate_cmd("nslkdd"),
            False,
            lambda: None if have("nslkdd") else "nslkdd not fetched",
        ),
        (
            "drift nslkdd",
            lambda: drift_cmd("nslkdd"),
            False,
            lambda: None if have("nslkdd") else "nslkdd not fetched",
        ),
        (
            "correlate cicids",
            lambda: correlate_cmd(),
            True,  # fit on 400k flows + replay 240k, ~12 minutes
            lambda: None if have("cicids") else "cicids not fetched",
        ),
        (
            "export-onnx nslkdd",
            lambda: export_onnx("nslkdd"),
            False,
            lambda: (
                None
                if have("nslkdd") and fitted("nslkdd") and have_onnx()
                else ("the `onnx` extra is not installed" if not have_onnx() else "no fitted nslkdd detector")
            ),
        ),
        (
            "poison-drill nslkdd",
            lambda: poison_drill(),
            True,  # six full two-head fits, ~13 minutes
            lambda: None if have("nslkdd") else "nslkdd not fetched",
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


@app.command("drift")
def drift_cmd(
    dataset: DatasetName = "nslkdd",
    inject: Annotated[
        str | None, typer.Option("--inject", help="abrupt | gradual | seasonal: perturb the current window")
    ] = None,
    rows: Annotated[int, typer.Option("--rows", help="Rows in the current window.")] = 20_000,
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Rank features by drift between training (reference) and test (current).

    PSI with reference-frozen bins, KS with Benjamini-Hochberg across the whole feature set, and
    the count of KS flags expected by chance printed next to the observed count. Without --inject
    this measures the dataset's own train/test shift, which on NSL-KDD is large and real.
    """
    seed_everything()
    from penumbra.drift import detectors

    ds = _load(dataset)
    reference = ds.X_train
    current = ds.X_test.sample(n=min(rows, len(ds.X_test)), random_state=0)
    if inject:
        from penumbra.drift import injector

        numeric = list(current.select_dtypes("number").columns)
        plan = injector.scenario(inject, len(current))
        current = injector.inject(current, plan, numeric_features=numeric)
        # Only the rows after the change point are the drifted window.
        current = current.iloc[plan.change_point :]

    bins = detectors.fit_bins(reference)
    report = detectors.compare(reference, current, bins=bins)
    console.print(report.summary())

    if save:
        out = settings().report_dir / f"drift_features_{dataset.lower()}.json"
        payload = {"dataset": dataset.lower(), "inject": inject, **report.to_dict()}
        out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")


@app.command("gate")
def gate_cmd(
    baseline: Annotated[Path, typer.Option("--baseline")] = Path("ci/baseline_nslkdd.json"),
    write: Annotated[
        bool, typer.Option("--write", help="Record the current numbers as the baseline.")
    ] = False,
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    """ML regression gate: fit the supervised head on NSL-KDD and compare to the committed baseline.

    Exit 1 on a regression beyond tolerance, and also on an improvement beyond tolerance - an
    improvement must be recorded in the baseline in the same change, so it gets reviewed.
    """
    seed_everything()
    from penumbra.data.loaders import nsl_kdd
    from penumbra.eval import regression
    from penumbra.models import supervised

    ds, _, fine_test = nsl_kdd.load_with_fine_labels()
    model = supervised.build("rf", ds, n_classes=2, balanced=True)
    model.fit(ds.X_train, ds.y_train)
    scores = supervised.attack_scores(model, ds.X_test)
    current = regression.measure(scores, ds.y_test.to_numpy(), nsl_kdd.unseen_mask(fine_test).to_numpy())

    if write:
        regression.write_baseline(baseline, current, note=note or "recorded with `penumbra gate --write`")
        console.print(f"[green]baseline written[/green] {baseline}")
        for m in regression.METRICS:
            console.print(f"  {m:<24} {current[m]:.4f}")
        return

    if not baseline.exists():
        console.print(f"[red]No baseline at {baseline}.[/red] Record one with --write.")
        raise typer.Exit(1)
    outcome = regression.compare(current, json.loads(baseline.read_text(encoding="utf-8")))
    console.print(outcome.summary())
    if not outcome.passed:
        raise typer.Exit(1)


@app.command("correlate")
def correlate_cmd(
    train_rows: Annotated[
        int, typer.Option("--train-rows", help="Mon-Wed rows to fit on (strided).")
    ] = 400_000,
    rows: Annotated[int, typer.Option("--rows", help="Thu-Fri flows to replay (strided).")] = 240_000,
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
    fixture_out: Annotated[
        Path | None,
        typer.Option("--write-fixture", help="Freeze every incident, with sample alerts, for the console."),
    ] = None,
) -> None:
    """Alert-to-incident correlation on CICIDS2017's real source IPs, end to end.

    Fit the two-head detector on Mon-Wed, replay Thu-Fri with the capture's own timestamps and
    pseudonymised source addresses, and group alerts by (source, family, 15-minute window).

    Both sides are thinned by STRIDE over time order, not by taking the first N rows: the first N
    rows of Monday contain no attacks at all, and the first N of Thursday would be one morning.
    """
    seed_everything()
    from dataclasses import replace

    from penumbra.alerts.correlate import Correlator
    from penumbra.data.loaders import cicids
    from penumbra.models.detector import PenumbraDetector

    ds = cicids.load()
    ents, stamps = cicids.entities(ds, "test"), cicids.timestamps(ds, "test")

    def stride(n_total: int, n_keep: int) -> np.ndarray:
        return np.unique(np.linspace(0, n_total - 1, min(n_keep, n_total)).astype(int))

    tr = stride(len(ds.X_train), train_rows)
    fit_ds = replace(
        ds,
        X_train=ds.X_train.iloc[tr].reset_index(drop=True),
        y_train=ds.y_train.iloc[tr].reset_index(drop=True),
        fam_train=ds.fam_train.iloc[tr].reset_index(drop=True),
    )
    console.print(f"[dim]fitting on {len(fit_ds.X_train):,} Mon-Wed flows (strided)...[/dim]")
    det = PenumbraDetector(target_fpr=0.01).fit(
        fit_ds, on_progress=lambda m: console.print(f"[dim]  {m}[/dim]")
    )

    # Time order first, then stride, so the replayed stream spans both days.
    order = np.argsort(pd.to_datetime(pd.Series(stamps)).to_numpy(), kind="stable")
    keep = order[stride(len(order), rows)]
    X = ds.X_test.iloc[keep].reset_index(drop=True)
    X.attrs.clear()  # the entity frame rides in attrs and pandas deep-copies it on every slice
    kept_ents = [ents[i] for i in keep]
    # Destination address and port travel as alert context, never as model features (the port is a
    # quarantined leak column). Without them every incident title read "0 destinations, 0 ports",
    # which hides the one thing that makes a scan a scan: fan-out.
    dests = cicids.destinations(ds, "test")
    kept_dests = [dests[i] for i in keep] if dests else []
    kept_stamps = [pd.Timestamp(stamps[i]).tz_localize("UTC").to_pydatetime() for i in keep]

    console.print(f"[dim]replaying {len(X):,} Thu-Fri flows...[/dim]")
    from penumbra.alerts.builder import alerts_with_positions

    alerts, row_of = [], {}
    for start in range(0, len(X), 5000):
        chunk = X.iloc[start : start + 5000]
        for pos, alert in alerts_with_positions(
            det, chunk, dataset="cicids", entities=kept_ents[start : start + 5000]
        ):
            alert.timestamp = kept_stamps[start + pos]
            row_of[alert.alert_id] = start + pos
            if kept_dests:
                alert.network.dst_ip, alert.network.dst_port = kept_dests[start + pos]
            alerts.append(alert)
    result = Correlator().correlate(alerts, dataset="cicids")
    console.print(result.summary())

    # The family on an incident is the model's PREDICTION. Ground truth is attached alongside, so
    # a mislabelled incident (e.g. Friday's DDoS called by a Wednesday class name) is visible.
    truth = ds.fam_test.iloc[keep].reset_index(drop=True)
    y_kept = ds.y_test.iloc[keep].to_numpy()

    def composition(inc) -> dict[str, int]:
        counts: dict[str, int] = {}
        for aid in inc.alert_ids:
            label = str(truth.iloc[row_of[aid]])
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    top = [
        {
            "title": inc.title,
            "events": inc.event_count,
            "destinations": inc.distinct_destinations,
            "ports": inc.distinct_ports,
            "severity": inc.severity.value,
            "family_predicted": inc.family,
            "ground_truth": composition(inc),
        }
        for inc in sorted(result.incidents, key=lambda i: i.event_count, reverse=True)[:25]
    ]

    biggest = max(result.incidents, key=lambda i: i.event_count) if result.incidents else None
    report = {
        "dataset": "cicids2017",
        "split": "train Mon-Wed, replay Thu-Fri",
        "train_rows": int(len(fit_ds.X_train)),
        "replayed_flows": int(len(X)),
        "window_minutes": 15,
        "group_by": ["source (pseudonymised)", "predicted family", "15-minute window"],
        "n_alerts": result.n_alerts,
        "n_incidents": len(result.incidents),
        "compression_ratio": result.compression_ratio,
        "largest": None
        if biggest is None
        else {
            "title": biggest.title,
            "events": biggest.event_count,
            "destinations": biggest.distinct_destinations,
            "ports": biggest.distinct_ports,
            "family_predicted": biggest.family,
            "ground_truth": composition(biggest),
        },
        "attack_rows_replayed": int(ds.y_test.iloc[keep].sum()),
        "alerts_by_verdict": {
            v: sum(a.verdict.value == v for a in alerts) for v in {a.verdict.value for a in alerts}
        },
        # What each lane actually received, against ground truth. The lane an attack lands in is a
        # routing decision; whether it reached a human at all is the detection question.
        "verdict_truth": {
            v: {
                "attack": sum(
                    1 for a in alerts if a.verdict.value == v and int(y_kept[row_of[a.alert_id]]) == 1
                ),
                "benign": sum(
                    1 for a in alerts if a.verdict.value == v and int(y_kept[row_of[a.alert_id]]) == 0
                ),
            }
            for v in {a.verdict.value for a in alerts}
        },
        "attack_rows_reaching_an_analyst": sum(1 for a in alerts if int(y_kept[row_of[a.alert_id]]) == 1),
        "benign_rows_reaching_an_analyst": sum(1 for a in alerts if int(y_kept[row_of[a.alert_id]]) == 0),
        "benign_rows_replayed": int((ds.y_test.iloc[keep] == 0).sum()),
        "top": top,
    }
    if fixture_out is not None:
        from penumbra.replay import engine

        # Every incident, so the console's events-per-incident matches the measured figure (a top-N
        # slice overstates it), with a sample of up to 8 alerts each. event_count keeps the true size;
        # alert_ids is trimmed to what ships, so every id in the fixture resolves.
        by_id = {a.alert_id: a for a in alerts}
        chosen = sorted(result.incidents, key=lambda i: i.event_count, reverse=True)
        kept_incidents, kept_alerts = [], []
        for inc in chosen:
            sample = [by_id[a] for a in inc.alert_ids[:8] if a in by_id]
            kept_alerts.extend(sample)
            kept_incidents.append(inc.model_copy(update={"alert_ids": [a.alert_id for a in sample]}))
        path = engine.write_fixture(kept_alerts, kept_incidents, fixture_out)
        console.print(
            f"[dim]fixture: {len(kept_incidents)} incidents, {len(kept_alerts):,} alerts -> {path}[/dim]"
        )

    if save:
        out = settings().report_dir / "correlation_cicids.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")


@app.command("export-onnx")
def export_onnx(
    dataset: DatasetName = "nslkdd",
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Export the supervised head to ONNX and measure parity against scikit-learn on every test row.

    Needs the `onnx` extra. The .onnx file is written next to the detector with its SHA-256.
    """
    try:
        from penumbra.models import onnx_export
    except ImportError as exc:  # pragma: no cover - depends on the extra
        console.print("[red]Install the extra first:[/red] uv sync --extra onnx")
        raise typer.Exit(1) from exc

    det = _deployed_detector(dataset)
    ds = _load(dataset)
    blob = onnx_export.export(det.supervised_model, ds.X_test.head(10), ds.categorical)
    scorer = onnx_export.OnnxScorer(
        blob, ds.categorical, onnx_export.category_map(det.supervised_model, ds.categorical)
    )
    report = onnx_export.parity(
        det.supervised_model, scorer, ds.X_test, threshold=det.gate.supervised_threshold if det.gate else 0.5
    )

    import hashlib

    out = settings().model_dir / dataset.lower() / "supervised.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    report["bytes"] = len(blob)
    report["sha256"] = hashlib.sha256(blob).hexdigest()
    report["path"] = str(out)

    console.print(
        f"ONNX {len(blob) / 1e6:.1f} MB  max |dp| {report['max_abs_diff']:.2e}  "
        f"decisions flipped {report['decisions_flipped']} of {report['n_rows']:,}  "
        f"latency {report['sklearn_ms_per_batch']:.1f} -> {report['onnx_ms_per_batch']:.1f} ms per "
        f"{report['latency_batch']:,} rows (x{report['speedup']:.1f})"
    )
    if save:
        rp = settings().report_dir / f"onnx_{dataset.lower()}.json"
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"[dim]written to {rp}[/dim]")


@app.command("calibrate")
def calibrate_cmd(
    dataset: DatasetName = "unsw",
    model: Annotated[str, typer.Option("--model", "-m")] = "rf",
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Brier and reliability before and after calibration, in-distribution AND under shift.

    Train is split into fit / calibration / held-out slices. The held-out slice is exchangeable with
    the calibration data, so it shows what calibration can achieve; the test split is the dataset's
    own shifted split, which shows what survives. Reporting only the first is how a calibration
    claim gets made that deployment does not honour.
    """
    seed_everything()
    from sklearn.model_selection import train_test_split

    from penumbra.models import calibration, supervised
    from penumbra.seeds import SEED

    ds = _load(dataset, drop_artifacts=True)
    X_rest, X_hold, y_rest, y_hold = train_test_split(
        ds.X_train, ds.y_train, test_size=0.15, stratify=ds.y_train, random_state=SEED
    )
    X_fit, X_cal, y_fit, y_cal = calibration.split_for_calibration(X_rest, y_rest.to_numpy())
    est = supervised.build(model, ds, n_classes=2, balanced=True)
    est.fit(X_fit, y_fit)

    report: dict[str, Any] = {"dataset": ds.name, "model": model, "artifacts_quarantined": True}
    for label, X_eval, y_eval in (
        ("held_out_train", X_hold, y_hold.to_numpy()),
        ("test_split", ds.X_test, ds.y_test.to_numpy()),
    ):
        methods = calibration.compare_methods(est, X_cal, y_cal, X_eval, y_eval)
        report[label] = {
            name: {
                **rep.to_dict(),
                "decomposition": calibration.brier_decomposition(
                    y_eval,
                    (
                        est
                        if name == "uncalibrated"
                        else calibration.calibrate(est, X_cal, y_cal, method=name)
                    ).predict_proba(X_eval)[:, 1],
                ),
            }
            for name, rep in methods.items()
        }
        console.print(f"\n[bold]{label}[/bold]  (n={len(y_eval):,}, prevalence {float(np.mean(y_eval)):.3f})")
        for name, rep in methods.items():
            console.print(
                f"  {name:<13} Brier {rep.brier:.5f}   ECE {rep.expected_calibration_error:.5f}   MCE {rep.max_calibration_error:.5f}"
            )

    if save:
        out = settings().report_dir / f"calibration_{dataset.lower()}.json"
        out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")


@app.command("lab")
def lab_cmd(
    capture: Annotated[Path, typer.Argument(help="A capture from hardware you own.")],
    attacker: Annotated[str, typer.Option("--attacker", help="IP that ran the attacks.")],
    target: Annotated[str, typer.Option("--target", help="IP that was attacked.")],
    model: Annotated[str, typer.Option("--model")] = "unsw",
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Score a real lab capture against known ground truth, then re-baseline the novelty head on it.

    Ground truth is the address pair: flows between --attacker and --target are attack, the rest
    are not. The saved report holds counts and scores only - no addresses.
    """
    seed_everything()
    from penumbra.alerts.builder import alerts_with_positions
    from penumbra.eval import lab
    from penumbra.models.detector import PenumbraDetector
    from penumbra.pcap import assemble

    frame = assemble.assemble(capture)
    meta = frame.attrs["meta"].reset_index(drop=True)
    attack = lab.attack_mask(meta, attacker, target)
    if not attack.any():
        console.print(f"[red]No flows between {attacker} and {target} in this capture.[/red]")
        raise typer.Exit(1)

    det = PenumbraDetector.load(settings().model_dir / model)
    X = frame.copy()
    X.attrs = {}
    X = X[det._feature_names].reset_index(drop=True)
    scored = det.score(X)
    positioned = alerts_with_positions(det, X, scored, dataset=model, include_benign=True)
    verdicts = ["BENIGN"] * len(X)
    for pos, alert in positioned:
        verdicts[pos] = alert.verdict.value

    report = {
        "capture": {
            "flows": int(len(X)),
            "protocols": dict(pd.Series(frame["proto"].astype(str)).value_counts().items()),
        },
        "out_of_the_box": lab.out_of_the_box(scored, verdicts, attack),
        "rebaselined": lab.rebaseline(X, attack, scored["novelty_percentile"].to_numpy()),
    }
    oob, rb = report["out_of_the_box"], report["rebaselined"]
    console.print(
        f"{len(X):,} flows, {oob['attack_flows']['flows']:,} attack. Out of the box: "
        f"{oob['attack_flows']['reached_an_analyst']:,} attack and "
        f"{oob['other_flows']['reached_an_analyst']:,} of {oob['other_flows']['flows']:,} other flows reached an analyst."
    )
    for name in ("stock_unsw_novelty", "local_rebaselined_novelty"):
        r = rb[name]
        console.print(
            f"  {name:<27} AUC {r['roc_auc']:.3f}   recall@5% {r['recall_at_5pct_fpr']:.3f}   "
            f"recall@10% {r['recall_at_10pct_fpr']:.3f}"
        )
    if save:
        out = settings().report_dir / "pcap_lab.json"
        out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")


# =================================================================================================
# Model registry, retraining from promoted verdicts, and the poisoning drill
# =================================================================================================

registry_app = typer.Typer(
    name="registry",
    help="Versioned detectors, the champion pointer, promotion and rollback.",
    no_args_is_help=True,
)
app.add_typer(registry_app)


def _audit(actor: str, action: str, target: str, detail: dict[str, Any]) -> None:
    """Model promotions go in the same hash-chained log as analyst actions."""
    from penumbra.api.security.audit import AuditLog

    AuditLog(settings().state_dir / "audit.jsonl").append(
        actor=actor, role="cli", action=action, target=target, detail=detail
    )


@registry_app.command("init")
def registry_init(
    dataset: DatasetName = "nslkdd",
    by: Annotated[str, typer.Option("--by", help="Who is making this the first champion.")] = "admin",
) -> None:
    """Register the model `penumbra fit` wrote as the first champion.

    The first champion is the only version ever promoted without a gate report, because it has
    nothing to be compared against. The manifest records that it was ungated.
    """
    from penumbra.models.detector import PenumbraDetector

    reg = _registry(dataset)
    if reg.champion():
        console.print(f"[yellow]{dataset} already has a champion ({reg.champion()}).[/yellow]")
        raise typer.Exit(1)
    model_dir = settings().model_dir / dataset.lower()
    if not (model_dir / "detector.joblib").exists():
        console.print(f"[red]No detector at {model_dir}.[/red] Run `penumbra fit -d {dataset}` first.")
        raise typer.Exit(1)
    info = reg.register(
        PenumbraDetector.load(model_dir),
        created_by=by,
        training={"source": str(model_dir), "feedback_rows": 0},
    )
    reg.promote(info.version, approver=by, allow_without_gate=True)
    _audit(
        by, "model.promote", info.version, {"dataset": dataset, "gated": False, "reason": "initial champion"}
    )
    console.print(f"[green]{info.version}[/green] is the {dataset} champion (initial, ungated).")


@registry_app.command("list")
def registry_list(dataset: DatasetName = "nslkdd") -> None:
    """Every version, which one is champion, and whether its gate passed."""
    reg = _registry(dataset)
    champion = reg.champion()
    table = Table(title=f"{dataset} registry")
    for col in ("version", "created by", "parent", "feedback rows", "gate", "intact", ""):
        table.add_column(col)
    for v in reg.versions():
        gate = (
            "-"
            if v.gate is None
            else ("[green]passed[/green]" if v.gate.get("passed") else "[red]failed[/red]")
        )
        intact = "[green]yes[/green]" if not reg.verify(v.version) else "[red]NO[/red]"
        table.add_row(
            v.version,
            v.created_by,
            v.parent or "-",
            str(v.training.get("feedback_rows", "-")),
            gate,
            intact,
            "champion" if v.version == champion else "",
        )
    console.print(table)


@registry_app.command("promote")
def registry_promote(
    version: Annotated[str, typer.Argument()],
    dataset: DatasetName = "nslkdd",
    by: Annotated[str, typer.Option("--by")] = "admin",
) -> None:
    """Make a version the champion. Refused without a passed canary gate report."""
    from penumbra.models.registry import RegistryError

    reg = _registry(dataset)
    try:
        state = reg.promote(version, approver=by)
    except RegistryError as exc:
        console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(1) from exc
    _audit(by, "model.promote", version, {"dataset": dataset, "previous": state["history"][-1]["previous"]})
    console.print(f"[green]{version}[/green] is now the {dataset} champion. Recorded in the audit log.")


@registry_app.command("rollback")
def registry_rollback(
    dataset: DatasetName = "nslkdd", by: Annotated[str, typer.Option("--by")] = "admin"
) -> None:
    """Re-point the champion at the version it replaced. No retrain, no redeploy."""
    from penumbra.models.registry import RegistryError

    reg = _registry(dataset)
    try:
        state = reg.rollback(approver=by)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    _audit(
        by, "model.rollback", state["version"], {"dataset": dataset, "from": state["history"][-1]["previous"]}
    )
    console.print(f"champion is now [green]{state['version']}[/green].")


@registry_app.command("shadow")
def registry_shadow(
    version: Annotated[str, typer.Argument()],
    dataset: DatasetName = "nslkdd",
    rows: Annotated[int, typer.Option("--rows")] = 10_000,
) -> None:
    """Score a challenger beside the champion on the same stream, alerting on nothing.

    Alert volume per head, agreement and Cohen's kappa, and what the challenger drops or adds. The
    report is attached to the challenger's manifest, next to its gate report.
    """
    from penumbra.eval import shadow

    reg = _registry(dataset)
    champion = reg.champion()
    if champion is None or champion == version:
        console.print("[red]Need a champion, and a different version to shadow it with.[/red]")
        raise typer.Exit(1)
    ds = _load(dataset)
    # The live-traffic slice: the canary never sees production scoring, so shadow on feedback rows.
    from penumbra.eval import canary

    _, live, _ = canary.live_split(_canary_labels(dataset, ds))
    live = live[:rows]
    X, y = ds.X_test.iloc[live], ds.y_test.iloc[live].to_numpy()
    from penumbra.models.registry import RegistryError

    try:
        report = shadow.compare(reg.load(champion).score(X), reg.load(version).score(X), y)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    report["champion"] = champion
    console.print(shadow.summary(report))

    reg.attach_shadow(version, report)
    console.print(f"[dim]attached to {version}'s manifest[/dim]")


@registry_app.command("verify")
def registry_verify(dataset: DatasetName = "nslkdd") -> None:
    """Re-hash every artifact against its manifest. Exit 2 if anything was modified."""
    reg = _registry(dataset)
    bad = {v.version: reg.verify(v.version) for v in reg.versions()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        for version, files in bad.items():
            console.print(f"[red]{version}: {files} do not match the manifest[/red]")
        raise typer.Exit(2)
    console.print(f"all {len(reg.versions())} versions intact.")


def _canary_labels(dataset: str, ds):
    """Trusted labels for the canary: fine attack names where the dataset has them."""
    if dataset.lower() in {"nslkdd", "nsl", "kdd"}:
        from penumbra.data.loaders import nsl_kdd

        _, _, fine_test = nsl_kdd.load_with_fine_labels()
        return fine_test.astype(str).reset_index(drop=True)
    return ds.fam_test.astype(str).reset_index(drop=True)


@app.command()
def retrain(
    dataset: DatasetName = "nslkdd",
    db: Annotated[Path | None, typer.Option("--db", help="SQLite store holding promoted verdicts.")] = None,
    by: Annotated[str, typer.Option("--by")] = "admin",
    model: Annotated[str, typer.Option("--model", "-m")] = "rf",
) -> None:
    """Train a challenger on training data plus PROMOTED verdicts, then run the canary gate.

    The challenger is registered but never promoted here. Promotion is a separate, human step
    (`penumbra registry promote`) and is refused unless the gate passed.
    """
    seed_everything()
    from penumbra.eval import canary, retraining
    from penumbra.storage.sqlite import SqliteRepository

    reg = _registry(dataset)
    champion_version = reg.champion()
    if champion_version is None:
        console.print(f"[red]No champion.[/red] Run `penumbra registry init -d {dataset}` first.")
        raise typer.Exit(1)

    repo = SqliteRepository(db or settings().state_dir / "penumbra.db")
    rows = repo.promoted_training_rows()
    ds = _load(dataset)
    augmented, summary = retraining.augment(ds, rows)
    console.print(
        f"[dim]{summary.rows_offered} promoted training rows, {summary.rows_usable} carry the full "
        f"{ds.name} feature set[/dim]"
    )
    if not summary.rows_usable:
        console.print("Nothing to retrain on. Verdicts must be recorded, then promoted by a second senior.")
        raise typer.Exit(1)

    champion = reg.load()
    fine = _canary_labels(dataset, ds)
    can_idx, _, _ = canary.live_split(fine)
    console.print(f"[dim]fitting challenger on {len(augmented.X_train):,} rows...[/dim]")
    challenger, result = retraining.challenge(
        champion,
        augmented,
        ds.X_test.iloc[can_idx],
        ds.y_test.iloc[can_idx],
        fine.iloc[can_idx],
        model_name=model,
    )
    info = reg.register(challenger, created_by=by, parent=champion_version, training=summary.to_dict())
    reg.attach_gate(info.version, result.to_dict())
    _audit(by, "model.register", info.version, {"dataset": dataset, "gate_passed": result.passed})

    console.print(f"\nregistered [bold]{info.version}[/bold] (parent {champion_version})")
    console.print(
        f"  canary recall {result.champion.recall:.4f} -> {result.challenger.recall:.4f}   "
        f"FPR {result.champion.fpr:.4f} -> {result.challenger.fpr:.4f}"
    )
    if result.passed:
        console.print(
            f"  gate [green]PASSED[/green]. Promote with `penumbra registry promote {info.version}`."
        )
    else:
        console.print("  gate [red]FAILED[/red] - this version cannot be promoted:")
        for reason in result.reasons:
            console.print(f"    {reason}")


@app.command("poison-drill")
def poison_drill(
    model: Annotated[str, typer.Option("--model", "-m")] = "rf",
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
    flags_only: Annotated[
        bool, typer.Option("--flags-only", help="Measure the integrity flags only; skip the six refits.")
    ] = False,
    exclude_target: Annotated[
        list[str] | None,
        typer.Option("--exclude-target", help="Skip a type the rule would pick (replication)."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    tag: Annotated[str, typer.Option("--tag", help="Suffix for the report file name.")] = "",
) -> None:
    """E7: poison the feedback loop from one analyst account and measure what the controls catch."""
    seed_everything()
    from penumbra.data.loaders import nsl_kdd
    from penumbra.eval import poisoning

    ds, fine_train, fine_test = nsl_kdd.load_with_fine_labels()
    report = poisoning.run(
        ds,
        fine_train,
        fine_test,
        model_name=model,
        flags_only=flags_only,
        exclude=tuple(exclude_target or ()),
        seed=seed,
        on_progress=lambda m: console.print(f"[dim]  {m}[/dim]"),
    )
    if flags_only:
        for name, arm in report["arms"].items():
            if "integrity" not in arm:
                continue
            i = arm["integrity"]
            console.print(
                f"{name:<18} flipped {arm['n_flipped']:>4}  poisoned flagged {i['poisoned_flagged_any']:.1%}  "
                f"honest-account clearances flagged {i['honest_accounts_clearances_flagged_any']:.1%}  "
                + "  ".join(
                    f"{k}={v['poisoned']:.0%}"
                    for k, v in i["by_flag"].items()
                    if v["poisoned"] == v["poisoned"]
                )
            )
        if save:
            out = settings().report_dir / f"poisoning_flags_nslkdd{tag}.json"
            out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
            console.print(f"[dim]written to {out}[/dim]")
        return

    table = Table(
        title=f"E7 poisoning drill - target {report['target']} ({report['target_train_support']} train rows)"
    )
    for col in (
        "arm",
        "flipped",
        "target recall [95% CI]",
        "overall recall",
        "FPR",
        "gate",
        "poisoned flagged",
    ):
        table.add_column(col)
    for name, arm in report["arms"].items():
        ev = arm["evaluation"]
        lo, hi = ev["target_recall_ci95"]
        gate = (
            "-"
            if "gate" not in arm
            else ("[green]pass[/green]" if arm["gate"]["passed"] else "[red]FAIL[/red]")
        )
        flagged = arm.get("integrity", {}).get("poisoned_flagged_any")
        table.add_row(
            name,
            str(arm.get("n_flipped", "-")),
            f"{ev['target_recall']:.3f} [{lo:.3f}, {hi:.3f}]",
            f"{ev['recall']:.4f}",
            f"{ev['fpr']:.4f}",
            gate,
            "-" if flagged is None or flagged != flagged else f"{flagged:.1%}",
        )
    console.print(table)

    if save:
        out = settings().report_dir / f"poisoning_nslkdd{tag}.json"
        out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        console.print(f"[dim]written to {out}[/dim]")


# =================================================================================================
# The triage copilot
# =================================================================================================

copilot_app = typer.Typer(
    name="copilot", help="The offline, cited ATT&CK triage copilot (BM25, no LLM).", no_args_is_help=True
)
app.add_typer(copilot_app)


@copilot_app.command("build")
def copilot_build() -> None:
    """Reduce the ATT&CK STIX bundle to the local JSONL corpus the copilot retrieves from."""
    from penumbra.rag import corpus

    try:
        path = corpus.build()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    stats = corpus.stats(corpus.load(path))
    console.print(
        f"[green]corpus written[/green] {path}: {stats['n_techniques']:,} techniques "
        f"({stats['n_subtechniques']:,} sub-techniques), {stats['n_with_detection']:,} with detection guidance"
    )


@copilot_app.command("ask")
def copilot_ask(
    query: Annotated[str, typer.Argument(help="What you are looking at, in words.")],
    top: Annotated[int, typer.Option("--top")] = 5,
) -> None:
    """Search the ATT&CK corpus the way the copilot does, and show what it would cite."""
    from penumbra.rag.copilot import Retriever

    try:
        retriever = Retriever()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    hits = retriever.search(query, top=top)
    if not hits:
        console.print("No technique matches those words. The copilot would say so rather than guess.")
        return
    for technique, score in hits:
        console.print(
            f"[bold]{technique.technique_id}[/bold] {technique.name}  [dim]bm25 {score:.2f}  {technique.url}[/dim]"
        )


if __name__ == "__main__":
    app()

"""Command line entry point.

Deliberately the only entry point. There is no Makefile: this machine has no `make`, and a Typer
app declared in `[project.scripts]` works identically on every platform the team uses.

Every number that appears in the report or the slide deck is produced by a command here, so that
"regenerate everything" is one invocation rather than a remembered sequence of notebook cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

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


@app.command("reproduce-all")
def reproduce_all(
    out: Annotated[Path | None, typer.Option("--out", help="Directory for regenerated reports.")] = None,
) -> None:
    """Regenerate every number in the report from scratch."""
    _ = out
    console.print("[yellow]Not implemented yet - lands with Phase 1.[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()

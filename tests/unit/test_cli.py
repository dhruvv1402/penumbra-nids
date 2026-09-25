"""CLI smoke tests for the commands that need no dataset.

Several commands documented in the runbook once did not exist, and one existed with a flag the
docs did not match. These pin that every documented command parses, and that the no-data paths fail
with an instruction rather than a traceback.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from penumbra.cli import app

# Wide and fixed: rich wraps and truncates at the detected terminal width, which is 80 or less on
# CI and wider on a desktop, so width-dependent assertions passed locally and failed there.
WIDE = {"COLUMNS": "300", "TERMINAL_WIDTH": "300"}


class _Runner(CliRunner):
    def invoke(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("env", WIDE)
        return super().invoke(*args, **kwargs)


runner = _Runner()


ANSI = re.compile(r"\[[0-9;]*[A-Za-z]")


def flat(result) -> str:
    """Output with ANSI codes stripped and wrapping collapsed.

    Typer forces a colour terminal when GITHUB_ACTIONS is set, so on CI the help text is full of
    escape codes that split '--from-fixture' into pieces. It is decided at import, so the test
    cannot switch it off; it can only read through it.
    """
    return " ".join(ANSI.sub("", result.output).split())


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    from penumbra import config

    monkeypatch.setenv("PENUMBRA_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PENUMBRA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("PENUMBRA_STATE_ROOT", raising=False)
    config.settings.cache_clear()
    yield tmp_path
    config.settings.cache_clear()


DOCUMENTED = [
    ["data", "fetch", "--help"],
    ["eval", "--help"],
    ["fit", "--help"],
    ["replay", "--help"],
    ["serve", "--help"],
    ["retrain", "--help"],
    ["gate", "--help"],
    ["drift", "--help"],
    ["calibrate", "--help"],
    ["correlate", "--help"],
    ["export-onnx", "--help"],
    ["poison-drill", "--help"],
    ["reproduce-all", "--help"],
    ["registry", "init", "--help"],
    ["registry", "list", "--help"],
    ["registry", "promote", "--help"],
    ["registry", "rollback", "--help"],
    ["registry", "verify", "--help"],
    ["registry", "shadow", "--help"],
    ["copilot", "build", "--help"],
    ["copilot", "ask", "--help"],
]


@pytest.mark.parametrize("argv", DOCUMENTED, ids=lambda a: " ".join(a[:-1]))
def test_documented_commands_exist(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "argv,flag",
    [
        (["replay", "--help"], "--from-fixture"),
        (["replay", "--help"], "--inject-drift"),
        (["poison-drill", "--help"], "--flags-only"),
        (["poison-drill", "--help"], "--exclude-target"),
        (["drift", "--help"], "--inject"),
    ],
)
def test_documented_flags_exist(argv: list[str], flag: str) -> None:
    assert flag in flat(runner.invoke(app, argv))


def test_fixture_replay_without_ingest_reads_the_fixture(isolated, tmp_path) -> None:
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps({"generated_at": "x", "alerts": [], "incidents": []}), encoding="utf-8")
    result = runner.invoke(app, ["replay", "--from-fixture", str(fixture)])
    assert result.exit_code == 0 and "0 alerts" in flat(result)


def test_missing_fixture_is_an_instruction_not_a_traceback(isolated) -> None:
    result = runner.invoke(app, ["replay", "--from-fixture", "nope.json"])
    assert result.exit_code == 1 and "No fixture" in flat(result)


def test_empty_registry_lists_and_verifies_cleanly(isolated) -> None:
    assert runner.invoke(app, ["registry", "list", "-d", "nslkdd"]).exit_code == 0
    assert runner.invoke(app, ["registry", "verify", "-d", "nslkdd"]).exit_code == 0


def test_promote_unknown_version_is_refused(isolated) -> None:
    result = runner.invoke(app, ["registry", "promote", "v999", "-d", "nslkdd"])
    assert result.exit_code == 1 and "refused" in flat(result)


def test_rollback_with_nothing_to_roll_back_to(isolated) -> None:
    result = runner.invoke(app, ["registry", "rollback", "-d", "nslkdd"])
    assert result.exit_code == 1


def test_retrain_without_a_champion_says_what_to_run(isolated) -> None:
    result = runner.invoke(app, ["retrain", "-d", "nslkdd"])
    assert result.exit_code == 1 and "registry init" in flat(result)


def test_copilot_without_a_corpus_says_what_to_run(isolated) -> None:
    result = runner.invoke(app, ["copilot", "ask", "port scan"])
    assert result.exit_code == 1 and "copilot build" in flat(result)


def test_gate_without_a_baseline_says_what_to_run(isolated, tmp_path, monkeypatch) -> None:
    # The gate needs NSL-KDD before it reaches the baseline check, so only the missing-data path is
    # reachable here; it must fail, not pass.
    result = runner.invoke(app, ["gate", "--baseline", str(tmp_path / "none.json")])
    assert result.exit_code != 0

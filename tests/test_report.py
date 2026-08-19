"""Tests for terminal reporting. Renders into an in-memory buffer so
no real terminal is needed and output can be asserted on directly.
"""

from __future__ import annotations

import io

from rich.console import Console

from regeval.diff import diff_runs
from regeval.report import build_diff_table, build_summary_table, print_diff, print_run_summary
from regeval.runner import CaseResult, RunResult


def _make_case(id: str, score: float | None, passed: bool, error: str | None = None) -> CaseResult:
    return CaseResult(
        id=id,
        prompt="p",
        expected="e",
        actual="a" if error is None else None,
        score=score,
        reasoning="",
        passed=passed,
        error=error,
        duration_seconds=0.15,
    )


def _make_run(cases: list[CaseResult]) -> RunResult:
    return RunResult(
        suite_name="my suite",
        scorer="exact_match",
        threshold=0.8,
        timestamp=0.0,
        results=cases,
    )


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=120, no_color=True)
    return console, buffer


def test_summary_table_includes_all_case_ids():
    run = _make_run([_make_case("c1", 1.0, True), _make_case("c2", 0.3, False)])
    table = build_summary_table(run)

    rendered_ids = [str(cell) for row in table.columns[0].cells for cell in [row]]
    assert "c1" in rendered_ids
    assert "c2" in rendered_ids


def test_print_run_summary_shows_pass_count():
    console, buffer = _console()
    run = _make_run([_make_case("c1", 1.0, True), _make_case("c2", 1.0, True)])

    print_run_summary(console, run)
    output = buffer.getvalue()

    assert "2/2 passed" in output
    assert "All cases passed" in output


def test_print_run_summary_shows_failure_count():
    console, buffer = _console()
    run = _make_run([_make_case("c1", 1.0, True), _make_case("c2", 0.2, False)])

    print_run_summary(console, run)
    output = buffer.getvalue()

    assert "1 case(s) failed" in output


def test_print_run_summary_handles_error_case():
    console, buffer = _console()
    run = _make_run([_make_case("c1", None, False, error="simulated failure")])

    print_run_summary(console, run)
    output = buffer.getvalue()

    assert "ERROR" in output


def test_diff_table_includes_regression_marker():
    baseline = _make_run([_make_case("c1", 0.9, True)])
    current = _make_run([_make_case("c1", 0.2, False)])
    diff = diff_runs(baseline, current)

    table = build_diff_table(diff)
    rendered_ids = [str(cell) for cell in table.columns[0].cells]
    assert "c1" in rendered_ids


def test_print_diff_shows_regression_warning():
    console, buffer = _console()
    baseline = _make_run([_make_case("c1", 0.9, True)])
    current = _make_run([_make_case("c1", 0.2, False)])
    diff = diff_runs(baseline, current)

    print_diff(console, diff)
    output = buffer.getvalue()

    assert "1 regression(s) detected" in output


def test_print_diff_shows_no_regressions_message():
    console, buffer = _console()
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True)])
    diff = diff_runs(baseline, current)

    print_diff(console, diff)
    output = buffer.getvalue()

    assert "No regressions detected" in output


def test_print_diff_shows_improvement_count():
    console, buffer = _console()
    baseline = _make_run([_make_case("c1", 0.2, False)])
    current = _make_run([_make_case("c1", 0.95, True)])
    diff = diff_runs(baseline, current)

    print_diff(console, diff)
    output = buffer.getvalue()

    assert "1 case(s) improved" in output


def test_regressed_cases_appear_before_unchanged_in_diff_table():
    """Ordering matters: a maintainer should see what broke before
    scrolling past dozens of unchanged rows.
    """
    baseline = _make_run(
        [_make_case("stable", 1.0, True), _make_case("broken", 0.9, True)]
    )
    current = _make_run(
        [_make_case("stable", 1.0, True), _make_case("broken", 0.1, False)]
    )
    diff = diff_runs(baseline, current)

    table = build_diff_table(diff)
    rendered_ids = [str(cell) for cell in table.columns[0].cells]

    assert rendered_ids.index("broken") < rendered_ids.index("stable")

def test_render_diff_markdown_includes_regression_header():
    baseline = _make_run([_make_case("c1", 0.9, True)])
    current = _make_run([_make_case("c1", 0.2, False)])
    diff = diff_runs(baseline, current)

    from regeval.report import render_diff_markdown

    output = render_diff_markdown(diff)

    assert "regeval regression report" in output
    assert "regression(s) detected" in output
    assert "c1" in output
    assert "🔴 REGRESSED" in output


def test_render_diff_markdown_no_regressions():
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True)])
    diff = diff_runs(baseline, current)

    from regeval.report import render_diff_markdown

    output = render_diff_markdown(diff)

    assert "No regressions detected" in output


def test_render_diff_markdown_is_valid_table_structure():
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True)])
    diff = diff_runs(baseline, current)

    from regeval.report import render_diff_markdown

    output = render_diff_markdown(diff)

    assert "| Case | Status | Baseline | Current | Δ |" in output
    assert "|---|---|---|---|---|" in output
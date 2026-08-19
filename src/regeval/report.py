"""Rich terminal reporting: summary tables, live progress, and colored
regression diffs.

This is the layer that makes regeval's output genuinely pleasant to read,
not just correct. All rendering targets a rich.console.Console, which is
passed in rather than constructed internally -- this makes every function
here testable by rendering into an in-memory buffer instead of the real
terminal.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from regeval.diff import DiffResult, DiffStatus
from regeval.runner import RunResult


def build_progress() -> Progress:
    """A configured Progress instance for live suite-run feedback: spinner,
    description, a bar, and elapsed time. Used as a context manager by the
    CLI's `run` command, with run_suite()'s on_case_complete callback
    advancing it as results stream in.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )

def build_summary_table(run: RunResult) -> Table:
    """A per-case pass/fail table for a single run."""
    table = Table(title=f"{run.suite_name} — {run.pass_count}/{len(run.results)} passed")
    table.add_column("Case", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Time", justify="right", style="dim")

    for result in run.results:
        if result.error is not None:
            status = "[bold red]ERROR[/bold red]"
            score_str = "—"
        elif result.passed:
            status = "[bold green]PASS[/bold green]"
            score_str = f"{result.score:.2f}"
        else:
            status = "[bold red]FAIL[/bold red]"
            score_str = f"{result.score:.2f}" if result.score is not None else "—"

        table.add_row(result.id, status, score_str, f"{result.duration_seconds:.2f}s")

    return table

def print_run_summary(console: Console, run: RunResult) -> None:
    console.print(build_summary_table(run))
    if run.fail_count > 0:
        console.print(f"[bold red]{run.fail_count} case(s) failed.[/bold red]")
    else:
        console.print("[bold green]All cases passed.[/bold green]")

_STATUS_STYLE = {
    DiffStatus.REGRESSED: "bold red",
    DiffStatus.IMPROVED: "bold green",
    DiffStatus.UNCHANGED: "dim",
    DiffStatus.NEW: "cyan",
    DiffStatus.REMOVED: "yellow",
}

_STATUS_LABEL = {
    DiffStatus.REGRESSED: "▼ REGRESSED",
    DiffStatus.IMPROVED: "▲ IMPROVED",
    DiffStatus.UNCHANGED: "  unchanged",
    DiffStatus.NEW: "+ NEW",
    DiffStatus.REMOVED: "- REMOVED",
}


def build_diff_table(diff: DiffResult) -> Table:
    """A regression-focused table: regressed and improved cases first
    (the ones that actually matter), unchanged cases last.
    """
    table = Table(title="Regression diff")
    table.add_column("Case", style="cyan")
    table.add_column("Status", justify="left")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Δ", justify="right")

    ordered = diff.regressed + diff.improved + diff.new + diff.removed + diff.unchanged

    for case_diff in ordered:
        style = _STATUS_STYLE[case_diff.status]
        label = _STATUS_LABEL[case_diff.status]

        baseline_str = (
            f"{case_diff.baseline_score:.2f}" if case_diff.baseline_score is not None else "—"
        )
        current_str = (
            f"{case_diff.current_score:.2f}" if case_diff.current_score is not None else "—"
        )
        delta_str = f"{case_diff.score_delta:+.2f}" if case_diff.score_delta is not None else "—"

        table.add_row(
            case_diff.id,
            f"[{style}]{label}[/{style}]",
            baseline_str,
            current_str,
            delta_str,
        )

    return table


def print_diff(console: Console, diff: DiffResult) -> None:
    console.print(build_diff_table(diff))
    console.print()
    if diff.has_regressions:
        console.print(
            f"[bold red]⚠ {len(diff.regressed)} regression(s) detected.[/bold red]"
        )
    else:
        console.print("[bold green]✓ No regressions detected.[/bold green]")
    if diff.improved:
        console.print(f"[bold green]{len(diff.improved)} case(s) improved.[/bold green]")
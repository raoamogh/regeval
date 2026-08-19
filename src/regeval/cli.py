"""Command-line interface: regeval run, regeval diff, regeval init."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from regeval.exceptions import RegevalError, SuiteError
from regeval.providers.openai_compatible import OpenAICompatibleProvider
from regeval.report import build_progress, print_diff, print_run_summary
from regeval.runner import RunResult, run_suite
from regeval.suite import load_suite

console = Console()

_EXAMPLE_SUITE = """\
name: "Example suite"
provider:
  base_url: http://localhost:11434/v1  # Ollama by default -- no API key needed
  api_key: ollama
  model: llama3
scorer: exact_match
threshold: 0.8
cases:
  - id: math
    prompt: "What is 2 + 2? Answer with just the number, nothing else."
    expected: "4"
  - id: capital
    prompt: "What is the capital of France? Answer with just the city name, nothing else."
    expected: "Paris"
"""


def _build_provider(config: dict) -> OpenAICompatibleProvider:
    missing = [k for k in ("base_url", "api_key", "model") if k not in config]
    if missing:
        raise SuiteError(f"Suite 'provider' section is missing required field(s): {missing}")
    return OpenAICompatibleProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        embedding_model=config.get("embedding_model"),
    )


@click.group()
@click.version_option()
def main() -> None:
    """regeval: catch LLM output regressions before they ship."""


@main.command()
def init() -> None:
    """Scaffold an example suite in the current directory."""
    target = Path("regeval.suite.yaml")
    if target.exists():
        console.print(f"[yellow]{target} already exists — not overwriting.[/yellow]")
        sys.exit(1)

    target.write_text(_EXAMPLE_SUITE)
    console.print(f"[bold green]Created {target}[/bold green]")
    console.print(
        "\nThis example targets a local Ollama server (no API key needed). "
        "Run [bold]ollama pull llama3[/bold] first if you don't have it, then:"
    )
    console.print(f"\n  regeval run {target}\n")


@main.command()
@click.argument("suite_path", type=click.Path(exists=False))
@click.option("--output", "-o", type=click.Path(), default=None, help="Save results as JSON.")
@click.option("--workers", "-w", type=int, default=8, help="Max concurrent requests.")
def run(suite_path: str, output: str | None, workers: int) -> None:
    """Run a suite and print a pass/fail summary."""
    try:
        suite = load_suite(suite_path)
        provider = _build_provider(suite.provider_config)
    except (SuiteError, RegevalError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    progress = build_progress()
    with progress:
        task = progress.add_task(f"Running {suite.name}", total=len(suite.cases))

        def on_complete(_result):
            progress.advance(task)

        try:
            result = run_suite(suite, provider, max_workers=workers, on_case_complete=on_complete)
        except RegevalError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    console.print()
    print_run_summary(console, result)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"\nSaved results to [bold]{output}[/bold]")

    sys.exit(0 if result.fail_count == 0 else 1)


@main.command()
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("current_path", type=click.Path(exists=True))
@click.option(
    "--score-drop-threshold",
    type=float,
    default=0.1,
    help="Flag a still-passing case as regressed if its score drops more than this.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["rich", "markdown"]),
    default="rich",
    help="Output format. 'markdown' is used by the regeval GitHub Action for PR comments.",
)
def diff(
        baseline_path: str, 
        current_path: str, 
        score_drop_threshold: float, 
        output_format: str
    ) -> None:
    """Compare two saved run JSON files and report regressions."""
    try:
        baseline = RunResult.from_dict(json.loads(Path(baseline_path).read_text()))
        current = RunResult.from_dict(json.loads(Path(current_path).read_text()))
    except (json.JSONDecodeError, KeyError) as exc:
        console.print(f"[bold red]Error:[/bold red] could not parse run file: {exc}")
        sys.exit(1)

    from regeval.diff import diff_runs

    result = diff_runs(baseline, current, score_drop_threshold=score_drop_threshold)

    if output_format == "markdown":
        from regeval.report import render_diff_markdown

        print(render_diff_markdown(result))
    else:
        print_diff(console, result)

    sys.exit(1 if result.has_regressions else 0)


if __name__ == "__main__":
    main()
"""Concurrent suite execution engine.

Runs every test case in a Suite against a Provider using a thread pool, so
cases execute in parallel rather than sequentially -- each case involves a
real network round-trip to an LLM API, so running 20 cases one at a time
could take minutes; running them concurrently takes roughly as long as the
single slowest case.

An individual case failure (a provider error, an unparseable judge
response) does not abort the whole run -- it's captured as a failed
CaseResult with an error message, so one flaky call doesn't lose every
other result.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

from regeval.exceptions import RegevalError
from regeval.providers.base import Provider
from regeval.scorers.base import Scorer
from regeval.scorers.exact_match import ExactMatchScorer
from regeval.scorers.llm_judge import LLMJudgeScorer
from regeval.scorers.semantic_similarity import SemanticSimilarityScorer
from regeval.suite import Suite, TestCase


@dataclass
class CaseResult:
    id: str
    prompt: str
    expected: str
    actual: str | None
    score: float | None
    reasoning: str
    passed: bool
    error: str | None
    duration_seconds: float

@dataclass
class RunResult:
    suite_name: str
    scorer: str
    threshold: float
    timestamp: float
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return len(self.results) - self.pass_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "scorer": self.scorer,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            suite_name=data["suite_name"],
            scorer=data["scorer"],
            threshold=data["threshold"],
            timestamp=data["timestamp"],
            results=[CaseResult(**r) for r in data["results"]],
        )

def build_scorer(name: str, provider: Provider) -> Scorer:
    if name == "exact_match":
        return ExactMatchScorer()
    if name == "semantic_similarity":
        return SemanticSimilarityScorer(provider)
    if name == "llm_judge":
        return LLMJudgeScorer(provider)
    raise RegevalError(f"Unknown scorer: {name}")

def _run_case(case: TestCase, provider: Provider, scorer: Scorer, threshold: float) -> CaseResult:
    start = time.monotonic()
    try:
        actual = provider.generate(case.prompt)
        result = scorer.score(actual, case.expected)
        return CaseResult(
            id=case.id,
            prompt=case.prompt,
            expected=case.expected,
            actual=actual,
            score=result.score,
            reasoning=result.reasoning,
            passed=result.score >= threshold,
            error=None,
            duration_seconds=time.monotonic() - start,
        )
    except RegevalError as exc:
        # Only our own known error types are caught here -- a bug that
        # raises something unexpected (e.g. AttributeError from a broken
        # Provider implementation) should still surface loudly, not be
        # silently swallowed into a "failed" result.
        return CaseResult(
            id=case.id,
            prompt=case.prompt,
            expected=case.expected,
            actual=None,
            score=None,
            reasoning="",
            passed=False,
            error=str(exc),
            duration_seconds=time.monotonic() - start,
        )
def run_suite(
    suite: Suite,
    provider: Provider,
    max_workers: int = 8,
    on_case_complete: Callable[[CaseResult], None] | None = None,
) -> RunResult:
    """Execute every case in `suite` concurrently against `provider`.

    Args:
        max_workers: max concurrent threads. Bounded so a large suite
            doesn't open hundreds of simultaneous connections to the
            provider's API.
        on_case_complete: optional callback invoked with each CaseResult
            as it finishes -- used by the CLI to update a live progress
            bar as results stream in, rather than waiting for the whole
            run to finish.
    """
    scorer = build_scorer(suite.scorer, provider)
    run = RunResult(
        suite_name=suite.name,
        scorer=suite.scorer,
        threshold=suite.threshold,
        timestamp=time.time(),
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_case, case, provider, scorer, suite.threshold): case
            for case in suite.cases
        }
        for future in as_completed(futures):
            result = future.result()
            run.results.append(result)
            if on_case_complete is not None:
                on_case_complete(result)

    # Results arrive in completion order (whichever case finishes first),
    # not suite order -- restore suite order so reports are stable and
    # readable, not shuffled by network timing.
    order = {case.id: i for i, case in enumerate(suite.cases)}
    run.results.sort(key=lambda r: order[r.id])

    return run
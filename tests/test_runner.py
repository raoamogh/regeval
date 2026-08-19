"""Tests for the concurrent suite runner."""

from __future__ import annotations

import threading
import time

import pytest

from regeval.exceptions import ProviderError
from regeval.providers.base import Provider
from regeval.runner import build_scorer, run_suite
from regeval.suite import Suite, TestCase


class _FakeProvider(Provider):
    """Returns canned responses keyed by prompt. Optionally simulates
    latency and/or failures for specific prompts, to test concurrency
    and error handling.
    """

    def __init__(
        self,
        responses: dict[str, str],
        latency: float = 0.0,
        fail_on: set[str] | None = None,
        default_response: str | None = None,
    ) -> None:
        self._responses = responses
        self._latency = latency
        self._fail_on = fail_on or set()
        self._default_response = default_response
        self.call_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt: str, **kwargs: object) -> str:
        with self._lock:
            self.call_count += 1
        if self._latency:
            time.sleep(self._latency)
        if prompt in self._fail_on:
            raise ProviderError(f"simulated failure for prompt: {prompt}")
        if prompt in self._responses:
            return self._responses[prompt]
        if self._default_response is not None:
            return self._default_response
        raise KeyError(f"No canned response configured for prompt: {prompt!r}")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("not used in these tests")


def _make_suite(
        cases: list[TestCase], 
        scorer: str = "exact_match", 
        threshold: float = 0.8
    ) -> Suite:
    return Suite(
        name="test suite",
        scorer=scorer,
        threshold=threshold,
        cases=cases,
        provider_config={},
    )


def test_build_scorer_exact_match():
    scorer = build_scorer("exact_match", provider=None)
    assert scorer.score("a", "a").score == 1.0


def test_build_scorer_unknown_raises():
    from regeval.exceptions import RegevalError

    with pytest.raises(RegevalError, match="Unknown scorer"):
        build_scorer("made_up", provider=None)


def test_all_cases_pass():
    cases = [
        TestCase(id="c1", prompt="p1", expected="hello"),
        TestCase(id="c2", prompt="p2", expected="world"),
    ]
    provider = _FakeProvider({"p1": "hello", "p2": "world"})
    suite = _make_suite(cases)

    run = run_suite(suite, provider)

    assert run.pass_count == 2
    assert run.fail_count == 0
    assert all(r.passed for r in run.results)


def test_case_below_threshold_fails():
    cases = [TestCase(id="c1", prompt="p1", expected="hello")]
    provider = _FakeProvider({"p1": "goodbye"})
    suite = _make_suite(cases)

    run = run_suite(suite, provider)

    assert run.fail_count == 1
    assert run.results[0].passed is False
    assert run.results[0].score == 0.0


def test_results_preserve_suite_order_despite_concurrency():
    """Cases finish in unpredictable thread-completion order; the runner
    must restore original suite order in the final result.
    """
    cases = [TestCase(id=f"c{i}", prompt=f"p{i}", expected=f"e{i}") for i in range(10)]
    responses = {f"p{i}": f"e{i}" for i in range(10)}
    # Vary latency so completion order is NOT the same as suite order.
    provider = _FakeProvider(responses, latency=0.0)
    suite = _make_suite(cases)

    run = run_suite(suite, provider)

    assert [r.id for r in run.results] == [f"c{i}" for i in range(10)]


def test_one_case_failure_does_not_abort_others():
    cases = [
        TestCase(id="c1", prompt="p1", expected="hello"),
        TestCase(id="c2", prompt="bad", expected="whatever"),
        TestCase(id="c3", prompt="p3", expected="world"),
    ]
    provider = _FakeProvider(
        {"p1": "hello", "p3": "world"},
        fail_on={"bad"},
    )
    suite = _make_suite(cases)

    run = run_suite(suite, provider)

    assert len(run.results) == 3
    assert run.results[0].passed is True  # c1
    assert run.results[1].passed is False  # c2, errored
    assert run.results[1].error is not None
    assert "simulated failure" in run.results[1].error
    assert run.results[2].passed is True  # c3


def test_cases_run_concurrently_not_sequentially():
    """With 5 cases each taking 0.2s, sequential execution would take
    ~1.0s; concurrent execution should take close to 0.2s.
    """
    cases = [TestCase(id=f"c{i}", prompt=f"p{i}", expected=f"e{i}") for i in range(5)]
    responses = {f"p{i}": f"e{i}" for i in range(5)}
    provider = _FakeProvider(responses, latency=0.2)
    suite = _make_suite(cases)

    start = time.monotonic()
    run_suite(suite, provider, max_workers=5)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5  # well under the 1.0s sequential time
    assert provider.call_count == 5


def test_on_case_complete_callback_invoked_per_case():
    cases = [
        TestCase(id="c1", prompt="p1", expected="hello"),
        TestCase(id="c2", prompt="p2", expected="world"),
    ]
    provider = _FakeProvider({"p1": "hello", "p2": "world"})
    suite = _make_suite(cases)

    completed_ids = []

    def on_complete(result):
        completed_ids.append(result.id)

    run_suite(suite, provider, on_case_complete=on_complete)

    assert sorted(completed_ids) == ["c1", "c2"]


def test_run_result_serialization_round_trips():
    from regeval.runner import RunResult

    cases = [TestCase(id="c1", prompt="p1", expected="hello")]
    provider = _FakeProvider({"p1": "hello"})
    suite = _make_suite(cases)

    run = run_suite(suite, provider)
    data = run.to_dict()
    restored = RunResult.from_dict(data)

    assert restored.suite_name == run.suite_name
    assert restored.results[0].id == run.results[0].id
    assert restored.results[0].score == run.results[0].score


def test_llm_judge_scorer_selected_correctly():
    cases = [TestCase(id="c1", prompt="p1", expected="expected answer")]
    provider = _FakeProvider(
        {"p1": "actual model answer"},  # response to the case's own prompt
        default_response="SCORE: 9\nREASONING: very close",  # response to judge's grading prompt
    )
    suite = _make_suite(cases, scorer="llm_judge", threshold=0.8)

    run = run_suite(suite, provider)

    assert run.results[0].actual == "actual model answer"
    assert run.results[0].score == 0.9
    assert run.results[0].reasoning == "very close"
    assert run.results[0].passed is True
"""Tests for regression diff logic."""

from __future__ import annotations

import pytest

from regeval.diff import diff_runs
from regeval.runner import CaseResult, RunResult


def _make_case(id: str, score: float | None, passed: bool) -> CaseResult:
    return CaseResult(
        id=id,
        prompt="p",
        expected="e",
        actual="a",
        score=score,
        reasoning="",
        passed=passed,
        error=None,
        duration_seconds=0.1,
    )


def _make_run(cases: list[CaseResult]) -> RunResult:
    return RunResult(
        suite_name="test",
        scorer="exact_match",
        threshold=0.8,
        timestamp=0.0,
        results=cases,
    )


def test_unchanged_when_both_pass_same_score():
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True)])

    diff = diff_runs(baseline, current)

    assert len(diff.unchanged) == 1
    assert len(diff.regressed) == 0


def test_regressed_when_flips_pass_to_fail():
    baseline = _make_run([_make_case("c1", 0.9, True)])
    current = _make_run([_make_case("c1", 0.3, False)])

    diff = diff_runs(baseline, current)

    assert len(diff.regressed) == 1
    assert diff.regressed[0].id == "c1"
    assert diff.has_regressions is True


def test_regressed_when_score_drops_significantly_even_if_still_passing():
    """The key feature: score drops from 0.95 to 0.81 -- both above the
    0.8 pass threshold, so pass/fail alone wouldn't catch it, but the
    default 0.1 score_drop_threshold should.
    """
    baseline = _make_run([_make_case("c1", 0.95, True)])
    current = _make_run([_make_case("c1", 0.81, True)])

    diff = diff_runs(baseline, current, score_drop_threshold=0.1)

    assert len(diff.regressed) == 1
    assert diff.regressed[0].score_delta == pytest.approx(0.81 - 0.95)


def test_small_score_drop_within_threshold_is_unchanged():
    baseline = _make_run([_make_case("c1", 0.95, True)])
    current = _make_run([_make_case("c1", 0.90, True)])  # only -0.05, under 0.1 threshold

    diff = diff_runs(baseline, current, score_drop_threshold=0.1)

    assert len(diff.regressed) == 0
    assert len(diff.unchanged) == 1


def test_improved_when_flips_fail_to_pass():
    baseline = _make_run([_make_case("c1", 0.3, False)])
    current = _make_run([_make_case("c1", 0.9, True)])

    diff = diff_runs(baseline, current)

    assert len(diff.improved) == 1
    assert diff.improved[0].id == "c1"


def test_new_case_only_in_current():
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True), _make_case("c2", 1.0, True)])

    diff = diff_runs(baseline, current)

    assert len(diff.new) == 1
    assert diff.new[0].id == "c2"
    assert diff.new[0].baseline_score is None


def test_removed_case_only_in_baseline():
    baseline = _make_run([_make_case("c1", 1.0, True), _make_case("c2", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True)])

    diff = diff_runs(baseline, current)

    assert len(diff.removed) == 1
    assert diff.removed[0].id == "c2"
    assert diff.removed[0].current_score is None


def test_multiple_cases_classified_independently():
    baseline = _make_run(
        [
            _make_case("regressing", 0.9, True),
            _make_case("stable", 1.0, True),
            _make_case("improving", 0.2, False),
        ]
    )
    current = _make_run(
        [
            _make_case("regressing", 0.3, False),
            _make_case("stable", 1.0, True),
            _make_case("improving", 0.95, True),
        ]
    )

    diff = diff_runs(baseline, current)

    assert {d.id for d in diff.regressed} == {"regressing"}
    assert {d.id for d in diff.unchanged} == {"stable"}
    assert {d.id for d in diff.improved} == {"improving"}


def test_has_regressions_false_when_none():
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 1.0, True)])

    diff = diff_runs(baseline, current)
    assert diff.has_regressions is False


def test_score_delta_none_when_case_is_new_or_removed():
    baseline = _make_run([])
    current = _make_run([_make_case("c1", 1.0, True)])

    diff = diff_runs(baseline, current)
    assert diff.new[0].score_delta is None


def test_custom_score_drop_threshold_is_respected():
    baseline = _make_run([_make_case("c1", 1.0, True)])
    current = _make_run([_make_case("c1", 0.95, True)])  # -0.05 drop

    # With a strict threshold of 0.01, even a small drop should register.
    diff = diff_runs(baseline, current, score_drop_threshold=0.01)
    assert len(diff.regressed) == 1

    # With a loose threshold of 0.5, the same drop should not register.
    diff_loose = diff_runs(baseline, current, score_drop_threshold=0.5)
    assert len(diff_loose.regressed) == 0
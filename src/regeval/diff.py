"""Regression detection between two saved runs.

Compares a baseline RunResult against a current RunResult, case by case
(matched by case id), and classifies each into one of: regressed,
improved, unchanged, new (only in current), or removed (only in baseline).

A case counts as "regressed" if it flips from passed to failed, OR its
score drops by more than `score_drop_threshold` even if it's still
technically passing -- catching a case that went from 0.95 to 0.81 is
often more useful than waiting for it to actually cross the pass/fail
line.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from regeval.runner import CaseResult, RunResult


class DiffStatus(Enum):
    REGRESSED = "regressed"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    NEW = "new"
    REMOVED = "removed"

@dataclass
class CaseDiff:
    id: str
    status: DiffStatus
    baseline_score: float | None
    current_score: float | None
    baseline_passed: bool | None
    current_passed: bool | None

    @property
    def score_delta(self) -> float | None:
        if self.baseline_score is None or self.current_score is None:
            return None
        return self.current_score - self.baseline_score

@dataclass
class DiffResult:
    case_diffs: list[CaseDiff]

    @property
    def regressed(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.status == DiffStatus.REGRESSED]

    @property
    def improved(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.status == DiffStatus.IMPROVED]

    @property
    def unchanged(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.status == DiffStatus.UNCHANGED]

    @property
    def new(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.status == DiffStatus.NEW]

    @property
    def removed(self) -> list[CaseDiff]:
        return [d for d in self.case_diffs if d.status == DiffStatus.REMOVED]

    @property
    def has_regressions(self) -> bool:
        return len(self.regressed) > 0


def _classify(
    baseline: CaseResult | None,
    current: CaseResult | None,
    score_drop_threshold: float,
) -> CaseDiff:
    if baseline is None and current is not None:
        return CaseDiff(
            id=current.id,
            status=DiffStatus.NEW,
            baseline_score=None,
            current_score=current.score,
            baseline_passed=None,
            current_passed=current.passed,
        )

    if current is None and baseline is not None:
        return CaseDiff(
            id=baseline.id,
            status=DiffStatus.REMOVED,
            baseline_score=baseline.score,
            current_score=None,
            baseline_passed=baseline.passed,
            current_passed=None,
        )

    assert baseline is not None and current is not None  # both branches above handled None cases

    flipped_to_fail = baseline.passed and not current.passed
    score_dropped_significantly = (
        baseline.score is not None
        and current.score is not None
        and (baseline.score - current.score) > score_drop_threshold
    )

    if flipped_to_fail or score_dropped_significantly:
        status = DiffStatus.REGRESSED
    elif not baseline.passed and current.passed:
        status = DiffStatus.IMPROVED
    else:
        status = DiffStatus.UNCHANGED

    return CaseDiff(
        id=current.id,
        status=status,
        baseline_score=baseline.score,
        current_score=current.score,
        baseline_passed=baseline.passed,
        current_passed=current.passed,
    )


def diff_runs(
    baseline: RunResult,
    current: RunResult,
    score_drop_threshold: float = 0.1,
) -> DiffResult:
    """Compare `baseline` against `current`, matching cases by id.

    Args:
        score_drop_threshold: a case that's still passing but whose score
            dropped by more than this amount is still flagged as
            regressed -- catches quality decay before it crosses the
            pass/fail line.
    """
    baseline_by_id = {r.id: r for r in baseline.results}
    current_by_id = {r.id: r for r in current.results}
    all_ids = list(dict.fromkeys(list(baseline_by_id) + list(current_by_id)))

    case_diffs = [
        _classify(baseline_by_id.get(cid), current_by_id.get(cid), score_drop_threshold)
        for cid in all_ids
    ]

    return DiffResult(case_diffs=case_diffs)
"""Exact-match scorer: strict or normalized string equality."""

from __future__ import annotations

from regeval.scorers.base import Scorer, ScoreResult


class ExactMatchScorer(Scorer):
    """Scores 1.0 if actual == expected (after optional normalization),
    else 0.0. No partial credit -- use SemanticSimilarityScorer or
    LLMJudgeScorer for fuzzy comparisons.
    """

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True) -> None:
        self._case_sensitive = case_sensitive
        self._strip_whitespace = strip_whitespace

    def _normalize(self, text: str) -> str:
        if self._strip_whitespace:
            text = text.strip()
        if not self._case_sensitive:
            text = text.lower()
        return text

    def score(self, actual: str, expected: str) -> ScoreResult:
        match = self._normalize(actual) == self._normalize(expected)
        return ScoreResult(score=1.0 if match else 0.0)
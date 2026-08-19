"""Semantic-similarity scorer: compares meaning via embedding cosine
similarity, rather than exact string match.
"""

from __future__ import annotations

import math

from regeval.providers.base import Provider
from regeval.scorers.base import Scorer, ScoreResult


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

class SemanticSimilarityScorer(Scorer):
    """Scores by embedding both `actual` and `expected` via the given
    Provider, then computing cosine similarity between the two vectors.

    Cosine similarity naturally falls in [-1, 1] for typical text
    embeddings but is clamped to [0, 1] here to match the Scorer contract
    (negative similarity is rare for real text and treated as 0).
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, actual: str, expected: str) -> ScoreResult:
        actual_vec = self._provider.embed(actual)
        expected_vec = self._provider.embed(expected)
        similarity = _cosine_similarity(actual_vec, expected_vec)
        clamped = max(0.0, min(1.0, similarity))
        return ScoreResult(score=clamped)
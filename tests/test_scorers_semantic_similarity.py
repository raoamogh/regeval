"""Tests for SemanticSimilarityScorer."""

from __future__ import annotations

from regeval.providers.base import Provider
from regeval.scorers.semantic_similarity import SemanticSimilarityScorer, _cosine_similarity


class _FakeProvider(Provider):
    """Returns pre-set embeddings for known input strings, so we can test
    the scoring logic without any real embedding model.
    """

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def generate(self, prompt: str, **kwargs: object) -> str:
        raise NotImplementedError("not used in these tests")

    def embed(self, text: str) -> list[float]:
        return self._embeddings[text]


def test_identical_vectors_score_1():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_orthogonal_vectors_score_0():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_vectors_score_negative_one():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_zero_vector_returns_0_not_divide_by_zero():
    # A zero-length vector would cause a ZeroDivisionError without the guard.
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_scorer_uses_provider_embeddings():
    provider = _FakeProvider(
        {
            "close to expected": [1.0, 0.0, 0.0],
            "expected text": [1.0, 0.0, 0.0],
        }
    )
    scorer = SemanticSimilarityScorer(provider)

    result = scorer.score("close to expected", "expected text")
    assert result.score == 1.0


def test_scorer_clamps_negative_similarity_to_zero():
    provider = _FakeProvider(
        {
            "actual": [1.0, 0.0],
            "expected": [-1.0, 0.0],
        }
    )
    scorer = SemanticSimilarityScorer(provider)

    result = scorer.score("actual", "expected")
    assert result.score == 0.0  # clamped, not -1.0


def test_scorer_result_has_no_reasoning():
    provider = _FakeProvider({"a": [1.0], "b": [1.0]})
    scorer = SemanticSimilarityScorer(provider)

    result = scorer.score("a", "b")
    assert result.reasoning == ""
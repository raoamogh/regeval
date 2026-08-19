"""Tests for LLMJudgeScorer."""

from __future__ import annotations

import pytest

from regeval.exceptions import ScorerError
from regeval.providers.base import Provider
from regeval.scorers.llm_judge import LLMJudgeScorer


class _FakeJudgeProvider(Provider):
    """Returns a pre-set judge response, so we can test parsing logic
    without any real model call.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None
        self.last_kwargs: dict | None = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self._response

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("not used in these tests")


def test_parses_score_and_reasoning():
    provider = _FakeJudgeProvider("SCORE: 8\nREASONING: Mostly correct, minor wording difference.")
    scorer = LLMJudgeScorer(provider)

    result = scorer.score(actual="The sky is blue.", expected="The sky appears blue.")

    assert result.score == 0.8
    assert result.reasoning == "Mostly correct, minor wording difference."


def test_normalizes_score_to_0_1_range():
    provider = _FakeJudgeProvider("SCORE: 10\nREASONING: Perfect match.")
    scorer = LLMJudgeScorer(provider)

    result = scorer.score(actual="x", expected="x")
    assert result.score == 1.0


def test_score_of_zero():
    provider = _FakeJudgeProvider("SCORE: 0\nREASONING: Completely wrong.")
    scorer = LLMJudgeScorer(provider)

    result = scorer.score(actual="x", expected="y")
    assert result.score == 0.0


def test_calls_provider_with_temperature_zero():
    provider = _FakeJudgeProvider("SCORE: 5\nREASONING: ok")
    scorer = LLMJudgeScorer(provider)

    scorer.score(actual="x", expected="y")

    assert provider.last_kwargs == {"temperature": 0}


def test_prompt_includes_both_actual_and_expected():
    provider = _FakeJudgeProvider("SCORE: 5\nREASONING: ok")
    scorer = LLMJudgeScorer(provider)

    scorer.score(actual="my actual text", expected="my expected text")

    assert "my actual text" in provider.last_prompt
    assert "my expected text" in provider.last_prompt


def test_unparseable_response_raises_scorer_error():
    provider = _FakeJudgeProvider("I refuse to grade this.")
    scorer = LLMJudgeScorer(provider)

    with pytest.raises(ScorerError, match="did not contain a parseable SCORE"):
        scorer.score(actual="x", expected="y")


def test_score_out_of_range_is_clamped():
    # Defensive: a judge model might not perfectly respect "0 to 10".
    provider = _FakeJudgeProvider("SCORE: 15\nREASONING: overshot")
    scorer = LLMJudgeScorer(provider)

    result = scorer.score(actual="x", expected="y")
    assert result.score == 1.0  # clamped, not 1.5
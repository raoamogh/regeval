"""Tests for ExactMatchScorer."""

from __future__ import annotations

from regeval.scorers.exact_match import ExactMatchScorer


def test_identical_strings_score_1():
    scorer = ExactMatchScorer()
    result = scorer.score("hello world", "hello world")
    assert result.score == 1.0


def test_different_strings_score_0():
    scorer = ExactMatchScorer()
    result = scorer.score("hello world", "goodbye world")
    assert result.score == 0.0


def test_case_insensitive_by_default():
    scorer = ExactMatchScorer()
    result = scorer.score("HELLO", "hello")
    assert result.score == 1.0


def test_case_sensitive_when_configured():
    scorer = ExactMatchScorer(case_sensitive=True)
    result = scorer.score("HELLO", "hello")
    assert result.score == 0.0


def test_whitespace_stripped_by_default():
    scorer = ExactMatchScorer()
    result = scorer.score("  hello  ", "hello")
    assert result.score == 1.0


def test_whitespace_preserved_when_configured():
    scorer = ExactMatchScorer(strip_whitespace=False)
    result = scorer.score("  hello  ", "hello")
    assert result.score == 0.0


def test_result_has_no_reasoning():
    scorer = ExactMatchScorer()
    result = scorer.score("hello", "hello")
    assert result.reasoning == ""
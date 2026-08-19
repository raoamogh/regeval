"""Abstract scorer interface.

A Scorer compares an actual model output against an expected output and
returns a score in [0.0, 1.0], where 1.0 means a perfect match and 0.0
means completely wrong. regeval ships three: exact-match (string equality),
semantic-similarity (embedding cosine similarity), and llm-judge (a second
model call to grade the output).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result of scoring one output against its expected value.

    score: float in [0.0, 1.0].
    reasoning: optional human-readable explanation (populated by scorers
        that can explain themselves, e.g. llm_judge; empty string for
        scorers that can't, e.g. exact_match).
    """

    score: float
    reasoning: str = ""

class Scorer(ABC):
    """Minimal interface a scorer must implement."""

    @abstractmethod
    def score(self, actual: str, expected: str) -> ScoreResult:
        """Compare `actual` (the model's real output) against `expected`
        (what the test case says should happen), and return a ScoreResult.
        """
        raise NotImplementedError
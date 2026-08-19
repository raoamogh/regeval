"""LLM-as-judge scorer: uses a model to grade another model's output
against the expected result, with reasoning.
"""

from __future__ import annotations

import re

from regeval.exceptions import ScorerError
from regeval.providers.base import Provider
from regeval.scorers.base import Scorer, ScoreResult

_JUDGE_PROMPT_TEMPLATE = """You are grading whether an AI response correctly \
matches an expected result. Score strictly.

Expected result:
{expected}

Actual response:
{actual}

Respond in EXACTLY this format, nothing else:
SCORE: <a number from 0 to 10>
REASONING: <one sentence explaining the score>"""

_SCORE_PATTERN = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_REASONING_PATTERN = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


class LLMJudgeScorer(Scorer):
    """Uses `provider` to grade `actual` against `expected` on a 0-10
    scale, normalized to [0.0, 1.0], with a one-sentence reasoning string.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, actual: str, expected: str) -> ScoreResult:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(expected=expected, actual=actual)
        response = self._provider.generate(prompt, temperature=0)

        score_match = _SCORE_PATTERN.search(response)
        if score_match is None:
            raise ScorerError(f"Judge response did not contain a parseable SCORE: {response!r}")

        raw_score = float(score_match.group(1))
        normalized = max(0.0, min(1.0, raw_score / 10.0))

        reasoning_match = _REASONING_PATTERN.search(response)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return ScoreResult(score=normalized, reasoning=reasoning)
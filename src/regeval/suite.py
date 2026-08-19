"""Test suite loading and validation.

A suite is a YAML file defining a name, a scorer to use, a pass/fail
threshold, and a list of test cases (each with a prompt and an expected
result). This module parses and validates that structure, failing loudly
with a specific, actionable message when something's wrong -- a suite file
is hand-written by a user, so unclear errors here directly cost adoption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from regeval.exceptions import SuiteError

VALID_SCORERS = {"exact_match", "semantic_similarity", "llm_judge"}

@dataclass
class TestCase:
    id: str
    prompt: str
    expected: str

@dataclass
class Suite:
    name: str
    scorer: str
    threshold: float
    cases: list[TestCase]
    provider_config: dict

def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite YAML file. Raises SuiteError with a
    specific, actionable message on any problem.
    """
    path = Path(path)
    if not path.exists():
        raise SuiteError(f"Suite file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SuiteError(f"Invalid YAML syntax in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SuiteError(
            f"Suite file {path} must be a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )
    return _validate_and_build(raw, source=str(path))

def _validate_and_build(raw: dict, source: str) -> Suite:
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise SuiteError(f"{source}: 'name' is required and must be a non-empty string")

    scorer = raw.get("scorer")
    if scorer not in VALID_SCORERS:
        raise SuiteError(
            f"{source}: 'scorer' must be one of {sorted(VALID_SCORERS)}, got {scorer!r}"
        )

    threshold = raw.get("threshold", 0.8)
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise SuiteError(
            f"{source}: 'threshold' must be a number between 0 and 1, got {threshold!r}"
        )

    provider_config = raw.get("provider")
    if not isinstance(provider_config, dict):
        raise SuiteError(f"{source}: 'provider' section is required and must be a mapping")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) == 0:
        raise SuiteError(f"{source}: 'cases' is required and must be a non-empty list")

    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    for i, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise SuiteError(
                f"{source}: cases[{i}] must be a mapping, got {type(raw_case).__name__}"
            )

        case_id = raw_case.get("id")
        if not case_id or not isinstance(case_id, str):
            raise SuiteError(f"{source}: cases[{i}] is missing a valid 'id'")
        if case_id in seen_ids:
            raise SuiteError(f"{source}: duplicate case id {case_id!r} — case ids must be unique")
        seen_ids.add(case_id)

        prompt = raw_case.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise SuiteError(f"{source}: case {case_id!r} is missing a valid 'prompt'")

        expected = raw_case.get("expected")
        if expected is None or not isinstance(expected, str):
            raise SuiteError(f"{source}: case {case_id!r} is missing a valid 'expected'")

        cases.append(TestCase(id=case_id, prompt=prompt, expected=expected))

    return Suite(
        name=name,
        scorer=scorer,
        threshold=float(threshold),
        cases=cases,
        provider_config=provider_config,
    )
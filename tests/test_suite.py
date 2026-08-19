"""Tests for suite loading and validation."""

from __future__ import annotations

import pytest

from regeval.exceptions import SuiteError
from regeval.suite import load_suite

VALID_SUITE = """
name: "Test suite"
provider:
  base_url: https://api.example.com/v1
  model: fake-model
scorer: exact_match
threshold: 0.8
cases:
  - id: case1
    prompt: "Say hello"
    expected: "Hello"
  - id: case2
    prompt: "Say goodbye"
    expected: "Goodbye"
"""


def _write_suite(tmp_path, content: str):
    path = tmp_path / "suite.yaml"
    path.write_text(content)
    return path


def test_loads_valid_suite(tmp_path):
    path = _write_suite(tmp_path, VALID_SUITE)
    suite = load_suite(path)

    assert suite.name == "Test suite"
    assert suite.scorer == "exact_match"
    assert suite.threshold == 0.8
    assert len(suite.cases) == 2
    assert suite.cases[0].id == "case1"
    assert suite.cases[0].prompt == "Say hello"
    assert suite.cases[0].expected == "Hello"


def test_missing_file_raises_suite_error(tmp_path):
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "nonexistent.yaml")


def test_invalid_yaml_syntax_raises_suite_error(tmp_path):
    path = _write_suite(tmp_path, "name: [unclosed bracket")
    with pytest.raises(SuiteError, match="Invalid YAML syntax"):
        load_suite(path)


def test_non_mapping_top_level_raises_suite_error(tmp_path):
    path = _write_suite(tmp_path, "- just\n- a\n- list")
    with pytest.raises(SuiteError, match="mapping"):
        load_suite(path)


def test_missing_name_raises_suite_error(tmp_path):
    content = VALID_SUITE.replace('name: "Test suite"\n', "")
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="'name'"):
        load_suite(path)


def test_invalid_scorer_raises_suite_error(tmp_path):
    content = VALID_SUITE.replace("scorer: exact_match", "scorer: made_up_scorer")
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="'scorer'"):
        load_suite(path)


def test_threshold_out_of_range_raises_suite_error(tmp_path):
    content = VALID_SUITE.replace("threshold: 0.8", "threshold: 1.5")
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="'threshold'"):
        load_suite(path)


def test_missing_provider_raises_suite_error(tmp_path):
    content = """
name: "Test"
scorer: exact_match
cases:
  - id: case1
    prompt: "hi"
    expected: "hello"
"""
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="'provider'"):
        load_suite(path)


def test_empty_cases_raises_suite_error(tmp_path):
    content = VALID_SUITE.split("cases:")[0] + "cases: []"
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="'cases'"):
        load_suite(path)


def test_case_missing_id_raises_suite_error(tmp_path):
    content = """
name: "Test"
provider:
  base_url: https://x.com
  model: m
scorer: exact_match
cases:
  - prompt: "hi"
    expected: "hello"
"""
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="missing a valid 'id'"):
        load_suite(path)


def test_case_missing_prompt_raises_suite_error(tmp_path):
    content = """
name: "Test"
provider:
  base_url: https://x.com
  model: m
scorer: exact_match
cases:
  - id: case1
    expected: "hello"
"""
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="missing a valid 'prompt'"):
        load_suite(path)


def test_case_missing_expected_raises_suite_error(tmp_path):
    content = """
name: "Test"
provider:
  base_url: https://x.com
  model: m
scorer: exact_match
cases:
  - id: case1
    prompt: "hi"
"""
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="missing a valid 'expected'"):
        load_suite(path)


def test_duplicate_case_ids_raises_suite_error(tmp_path):
    content = """
name: "Test"
provider:
  base_url: https://x.com
  model: m
scorer: exact_match
cases:
  - id: dupe
    prompt: "hi"
    expected: "hello"
  - id: dupe
    prompt: "bye"
    expected: "goodbye"
"""
    path = _write_suite(tmp_path, content)
    with pytest.raises(SuiteError, match="duplicate case id"):
        load_suite(path)


def test_default_threshold_when_not_specified(tmp_path):
    content = VALID_SUITE.replace("threshold: 0.8\n", "")
    path = _write_suite(tmp_path, content)
    suite = load_suite(path)
    assert suite.threshold == 0.8
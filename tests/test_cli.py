"""Tests for the CLI. Uses click's CliRunner to invoke commands in-process,
with a monkeypatched provider so no real network calls happen.
"""

from __future__ import annotations

import json

import httpx
import pytest
from click.testing import CliRunner

from regeval.cli import main


def _fake_transport(response_text: str = "4"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": response_text}}]})

    return httpx.MockTransport(handler)

def _routing_transport(chat_response: str, embedding_vector: list[float] | None = None):
    """Fake transport that routes based on the actual endpoint path, so it
    can correctly fake both /chat/completions and /embeddings calls in the
    same test -- needed for semantic_similarity (embed) and llm_judge
    (a second chat/completions call for grading).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": embedding_vector}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": chat_response}}]})

    return httpx.MockTransport(handler)


VALID_SUITE_YAML = """\
name: "CLI test suite"
provider:
  base_url: https://fake.example.com/v1
  api_key: test-key
  model: fake-model
scorer: exact_match
threshold: 0.8
cases:
  - id: c1
    prompt: "what is 2+2"
    expected: "4"
"""

SEMANTIC_SUITE_YAML = """\
name: "Semantic test suite"
provider:
  base_url: https://fake.example.com/v1
  api_key: test-key
  model: fake-model
  embedding_model: fake-embed-model
scorer: semantic_similarity
threshold: 0.7
cases:
  - id: c1
    prompt: "Explain what recursion is in programming."
    expected: "Recursion is when a function calls itself to solve a smaller \
    version of the same problem"
"""

JUDGE_SUITE_YAML = """\
name: "Judge test suite"
provider:
  base_url: https://fake.example.com/v1
  api_key: test-key
  model: fake-model
scorer: llm_judge
threshold: 0.7
cases:
  - id: c1
    prompt: "Summarize the plot of Romeo and Juliet in two sentences."
    expected: "Two young lovers from feuding families in Verona fall in love and secretly marry. \
    Their families' conflict ultimately leads to both of their deaths."
"""


@pytest.fixture
def runner():
    return CliRunner()


def test_init_creates_suite_file(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["init"])

    assert result.exit_code == 0
    assert (tmp_path / "regeval.suite.yaml").exists()
    assert "Created" in result.output


def test_init_does_not_overwrite_existing_file(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "regeval.suite.yaml").write_text("existing content")

    result = runner.invoke(main, ["init"])

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert (tmp_path / "regeval.suite.yaml").read_text() == "existing content"


def test_run_missing_suite_file_errors_cleanly(runner):
    result = runner.invoke(main, ["run", "nonexistent.yaml"])

    assert result.exit_code == 1
    assert "Error" in result.output
    assert "not found" in result.output


def test_run_invalid_suite_errors_cleanly(runner, tmp_path):
    bad_suite = tmp_path / "bad.yaml"
    bad_suite.write_text("name: test\nscorer: not_a_real_scorer\n")

    result = runner.invoke(main, ["run", str(bad_suite)])

    assert result.exit_code == 1
    assert "Error" in result.output


def test_run_succeeds_with_fake_provider(runner, tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(VALID_SUITE_YAML)

    # Patch httpx.Client construction so any real provider created inside
    # the CLI uses our fake transport instead of hitting the network.
    original_init = httpx.Client.__init__

    def fake_init(self, *args, **kwargs):
        kwargs["transport"] = _fake_transport("4")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", fake_init)

    result = runner.invoke(main, ["run", str(suite_path)])

    assert result.exit_code == 0
    assert "1/1 passed" in result.output
    assert "All cases passed" in result.output


def test_run_saves_output_json(runner, tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(VALID_SUITE_YAML)
    output_path = tmp_path / "results.json"

    original_init = httpx.Client.__init__

    def fake_init(self, *args, **kwargs):
        kwargs["transport"] = _fake_transport("4")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", fake_init)

    result = runner.invoke(main, ["run", str(suite_path), "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["suite_name"] == "CLI test suite"
    assert data["results"][0]["passed"] is True


def test_run_exits_nonzero_on_failed_case(runner, tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(VALID_SUITE_YAML)

    original_init = httpx.Client.__init__

    def fake_init(self, *args, **kwargs):
        kwargs["transport"] = _fake_transport("wrong answer")  # doesn't match expected "4"
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", fake_init)

    result = runner.invoke(main, ["run", str(suite_path)])

    assert result.exit_code == 1


def test_diff_no_regressions(runner, tmp_path):
    baseline = {
        "suite_name": "s",
        "scorer": "exact_match",
        "threshold": 0.8,
        "timestamp": 0.0,
        "results": [
            {
                "id": "c1",
                "prompt": "p",
                "expected": "e",
                "actual": "e",
                "score": 1.0,
                "reasoning": "",
                "passed": True,
                "error": None,
                "duration_seconds": 0.1,
            }
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(baseline))  # identical

    result = runner.invoke(main, ["diff", str(baseline_path), str(current_path)])

    assert result.exit_code == 0
    assert "No regressions detected" in result.output


def test_diff_detects_regression_and_exits_nonzero(runner, tmp_path):
    baseline = {
        "suite_name": "s",
        "scorer": "exact_match",
        "threshold": 0.8,
        "timestamp": 0.0,
        "results": [
            {
                "id": "c1",
                "prompt": "p",
                "expected": "e",
                "actual": "e",
                "score": 1.0,
                "reasoning": "",
                "passed": True,
                "error": None,
                "duration_seconds": 0.1,
            }
        ],
    }
    current = json.loads(json.dumps(baseline))
    current["results"][0]["score"] = 0.1
    current["results"][0]["passed"] = False

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(current))

    result = runner.invoke(main, ["diff", str(baseline_path), str(current_path)])

    assert result.exit_code == 1
    assert "regression" in result.output.lower()


def test_diff_malformed_json_errors_cleanly(runner, tmp_path):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text("not valid json{{{")
    current_path.write_text("{}")

    result = runner.invoke(main, ["diff", str(baseline_path), str(current_path)])

    assert result.exit_code == 1
    assert "Error" in result.output

def test_run_with_semantic_similarity_scorer(runner, tmp_path, monkeypatch):
    suite_path = tmp_path / "semantic_suite.yaml"
    suite_path.write_text(SEMANTIC_SUITE_YAML)

    original_init = httpx.Client.__init__

    def fake_init(self, *args, **kwargs):
        # Same embedding vector for both actual and expected -> similarity 1.0,
        # proving the scorer genuinely calls embed() twice and compares them,
        # not just returning a hardcoded pass.
        kwargs["transport"] = _routing_transport(
            chat_response="Recursion is a function calling itself to break a problem into smaller pieces.", #noqa
            embedding_vector=[0.2, 0.4, 0.6, 0.1],
        )
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", fake_init)

    result = runner.invoke(main, ["run", str(suite_path)])

    assert result.exit_code == 0
    assert "1/1 passed" in result.output


def test_run_with_llm_judge_scorer(runner, tmp_path, monkeypatch):
    suite_path = tmp_path / "judge_suite.yaml"
    suite_path.write_text(JUDGE_SUITE_YAML)

    original_init = httpx.Client.__init__
    call_log = []

    def fake_init(self, *args, **kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            prompt = body["messages"][0]["content"]
            call_log.append(prompt)
            if "SCORE:" in prompt:  # this is the judge's own grading prompt
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": "SCORE: 9\nREASONING: Captures the key plot points accurately." #noqa
                                }
                            }
                        ]
                    },
                )
            # this is the original case prompt, asking for the actual summary
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Romeo and Juliet are young lovers from rival families "
                                    "who marry secretly, and their deaths ultimately end the feud."
                                )
                            }
                        }
                    ]
                },
            )

        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", fake_init)

    result = runner.invoke(main, ["run", str(suite_path)])

    assert result.exit_code == 0
    assert "1/1 passed" in result.output
    # Proves two distinct generate() calls actually happened: one for the
    # real answer, one for the judge grading it -- not just one call reused.
    assert len(call_log) == 2

def test_diff_markdown_format(runner, tmp_path):
    baseline = {
        "suite_name": "s",
        "scorer": "exact_match",
        "threshold": 0.8,
        "timestamp": 0.0,
        "results": [
            {
                "id": "c1",
                "prompt": "p",
                "expected": "e",
                "actual": "e",
                "score": 1.0,
                "reasoning": "",
                "passed": True,
                "error": None,
                "duration_seconds": 0.1,
            }
        ],
    }
    current = json.loads(json.dumps(baseline))
    current["results"][0]["score"] = 0.1
    current["results"][0]["passed"] = False

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(current))

    result = runner.invoke(
        main, ["diff", str(baseline_path), str(current_path), "--format", "markdown"]
    )

    assert result.exit_code == 1
    assert "## regeval regression report" in result.output
    assert "| Case | Status |" in result.output
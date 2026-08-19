# Contributing to regeval

Thanks for considering contributing! This guide covers everything you need to get set up.

## Development setup

```bash
git clone https://github.com/raoamogh/regeval.git
cd regeval
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Most tests use fake providers (via `httpx.MockTransport`), so **no API key or real LLM is needed to run the test suite.**

If you want to manually try the CLI against a real model without any API key or cost, install [Ollama](https://ollama.com):

```bash
ollama pull llama3
ollama pull nomic-embed-text  # needed for semantic_similarity examples
regeval init
regeval run regeval.suite.yaml
```

## Running tests

```bash
pytest
```

Run lint before submitting anything:

```bash
ruff check src tests
```

Both must pass — matches exactly what CI checks on every pull request.

## Making a change

1. Fork the repo and create a branch: `git checkout -b feature/your-feature-name`
2. Write your change, with tests. New Provider or Scorer implementations especially need tests that prove correctness against a fake transport/provider, not just "it imports without error" — see `tests/test_providers_openai_compatible.py` and `tests/test_scorers_llm_judge.py` for the expected rigor.
3. Run `ruff check src tests` and `pytest` locally — both must pass.
4. Commit with a clear message (e.g. `feat: add Anthropic-native provider`).
5. Push and open a pull request describing what changed and why.

## Adding a new Scorer or Provider

Both are built around abstract interfaces (`regeval.scorers.base.Scorer`, `regeval.providers.base.Provider`) specifically so new implementations can be added without touching the runner, CLI, or reporting code. If you're adding one:

- Implement the interface, with real tests against a fake transport/provider (no real network calls in the test suite).
- Register it in the relevant `__init__.py` and, for scorers, in `build_scorer()` in `runner.py`.
- Update the suite schema validation in `suite.py` if you're adding a new scorer name to `VALID_SCORERS`.

## Reporting bugs

Open a GitHub issue with:
- Python version and OS
- The suite YAML (redact any real API keys/prompts if sensitive)
- What you expected vs. what happened

## AI tool usage

Using AI tools to help write your contribution is fine. See [AI_POLICY.md](AI_POLICY.md) for what's expected — the short version: understand what you're submitting, actually run the tests, and be honest if asked.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.
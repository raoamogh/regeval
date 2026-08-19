"""regeval: catch LLM output regressions before they ship.

pytest-style evaluation for LLM prompts. Write down what "correct" looks
like once as a YAML suite, run it against any OpenAI-compatible provider
(OpenAI, Groq, OpenRouter, Ollama, ...), score outputs with exact-match,
semantic-similarity, or LLM-as-judge scoring, and diff runs against a
saved baseline to catch regressions -- locally, or automatically on every
pull request via the included GitHub Action.

Basic usage (as a library):

    from regeval.suite import load_suite
    from regeval.providers.openai_compatible import OpenAICompatibleProvider
    from regeval.runner import run_suite

    suite = load_suite("suite.yaml")
    provider = OpenAICompatibleProvider(base_url=..., api_key=..., model=...)
    result = run_suite(suite, provider)

Most people will use the CLI instead: `regeval run suite.yaml`.
"""

from regeval.exceptions import ProviderError, RegevalError, ScorerError, SuiteError

__version__ = "0.1.0"

__all__ = [
    "RegevalError",
    "ProviderError",
    "ScorerError",
    "SuiteError",
]
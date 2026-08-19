"""Abstract provider interface.

A Provider is anything that can take a prompt and return a model's text
response (generate), and anything that can turn text into a vector
embedding (embed) -- used by the semantic-similarity scorer.

regeval ships one concrete implementation: OpenAICompatibleProvider, which
works against the OpenAI chat completions + embeddings API shape -- used by
OpenAI, Groq, OpenRouter, Together, Ollama, and most other LLM APIs, since
they've converged on the same request/response format. One implementation,
broad real-world compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Minimal interface a model provider must implement."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Send `prompt` to the model and return its text response.

        kwargs may include provider-specific options (temperature, max_tokens,
        etc.) -- implementations should pass through what they understand and
        ignore the rest.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for `text`.

        Used by the semantic-similarity scorer to compare an actual output
        against an expected output by meaning rather than exact string match.
        """
        raise NotImplementedError
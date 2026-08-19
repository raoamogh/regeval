"""OpenAI-compatible provider: chat completions + embeddings.

Works against any API implementing the OpenAI request/response shape:
OpenAI itself, Groq, OpenRouter, Together AI, Ollama (local), and most
other LLM API providers.
"""

from __future__ import annotations

from typing import Any

import httpx

from regeval.exceptions import ProviderError
from regeval.providers.base import Provider


class OpenAICompatibleProvider(Provider):
    """Calls an OpenAI-compatible /chat/completions and /embeddings endpoint.

    Examples:
        # OpenAI
        OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        )

        # Ollama (local, no real API key needed)
        OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3",
            embedding_model="nomic-embed-text",
        )
    """

    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str,
            embedding_model: str | None = None,
            timeout: float = 30.0,
            client: httpx.Client | None = None
    ) -> None:
        """
        Args:
            base_url: API base URL, e.g. "https://api.groq.com/openai/v1".
            api_key: bearer token for the Authorization header.
            model: model identifier used for generate().
            embedding_model: model identifier used for embed(). Required
                only if you use the semantic-similarity scorer.
            timeout: request timeout in seconds.
            client: an existing httpx.Client to use instead of creating one
                (used in tests to inject a fake client with no real network
                calls).
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._embedding_model = embedding_model
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                json = payload,
                headers = self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"generate() request to {self._base_url} failed: {exc}") from exc

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(
                f"Unexpected generate() response shape from {self._base_url}: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        if self._embedding_model is None:
            raise ProviderError(
                "No embedding_model configured on this provider. "
                "Pass embedding_model=... to use embed() / the semantic-similarity scorer."
            )

        payload = {"model": self._embedding_model, "input": text}
        try:
            response = self._client.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"embed() request to {self._base_url} failed: {exc}") from exc

        try:
            data = response.json()
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(
                f"Unexpected embed() response shape from {self._base_url}: {exc}"
            ) from exc
"""Tests for OpenAICompatibleProvider. Uses a fake httpx client -- no real
network calls, no API key needed, no cost.
"""

from __future__ import annotations

import httpx
import pytest

from regeval.exceptions import ProviderError
from regeval.providers.openai_compatible import OpenAICompatibleProvider


def _fake_client(handler) -> httpx.Client:
    """Build an httpx.Client backed by a fake transport, so requests never
    touch the real network -- `handler` decides what response to return.
    """
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_generate_returns_model_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello there"}}]},
        )

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        client=_fake_client(handler),
    )

    result = provider.generate("hi")
    assert result == "hello there"


def test_generate_sends_prompt_as_user_message():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        client=_fake_client(handler),
    )

    provider.generate("what is 2+2?")

    assert captured["body"]["messages"] == [{"role": "user", "content": "what is 2+2?"}]
    assert captured["body"]["model"] == "fake-model"


def test_generate_passes_through_kwargs():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        client=_fake_client(handler),
    )

    provider.generate("hi", temperature=0.2, max_tokens=50)

    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_tokens"] == 50


def test_generate_sends_auth_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="secret-token",
        model="fake-model",
        client=_fake_client(handler),
    )

    provider.generate("hi")
    assert captured["auth"] == "Bearer secret-token"


def test_generate_raises_provider_error_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        client=_fake_client(handler),
    )

    with pytest.raises(ProviderError, match="failed"):
        provider.generate("hi")


def test_generate_raises_provider_error_on_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        client=_fake_client(handler),
    )

    with pytest.raises(ProviderError, match="Unexpected"):
        provider.generate("hi")


def test_embed_returns_vector():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        embedding_model="fake-embed-model",
        client=_fake_client(handler),
    )

    result = provider.embed("some text")
    assert result == [0.1, 0.2, 0.3]


def test_embed_without_embedding_model_raises():
    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="test-key",
        model="fake-model",
        # no embedding_model passed
        client=_fake_client(lambda r: httpx.Response(200, json={})),
    )

    with pytest.raises(ProviderError, match="No embedding_model configured"):
        provider.embed("some text")


def test_embed_raises_provider_error_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    provider = OpenAICompatibleProvider(
        base_url="https://fake.example.com/v1",
        api_key="bad-key",
        model="fake-model",
        embedding_model="fake-embed-model",
        client=_fake_client(handler),
    )

    with pytest.raises(ProviderError, match="failed"):
        provider.embed("some text")
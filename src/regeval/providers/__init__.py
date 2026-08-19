"""Provider implementations for regeval."""

from regeval.providers.base import Provider
from regeval.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["Provider", "OpenAICompatibleProvider"]
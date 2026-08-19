"""Exception types raised by regeval"""


class RegevalError(Exception):
    """Base exception for all regeval errors."""

class ProviderError(RegevalError):
    """Raised when a provider fails to generate a response or produce
    embeddings (network error, API error, or unexpected response shape).
    """


class SuiteError(RegevalError):
    """Raised when a test suite YAML file is malformed or invalid."""


class ScorerError(RegevalError):
    """Raised when a scorer fails to evaluate an output."""
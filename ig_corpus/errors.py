from __future__ import annotations


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class ApifyError(RuntimeError):
    """Raised when an Apify Actor run or dataset read fails."""


class LLMError(RuntimeError):
    """Raised when an OpenAI model call or structured parse fails."""

from __future__ import annotations


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class ApifyError(RuntimeError):
    """Raised when an Apify Actor run or dataset read fails."""

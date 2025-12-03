from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml
from pydantic import ValidationError

from .config_schema import AppConfig
from .errors import ConfigError


@dataclass(frozen=True)
class RuntimeSecrets:
    apify_token: str
    openai_api_key: str


def load_config(path: str | Path) -> AppConfig:
    """
    Load a YAML config file and validate it into a typed AppConfig.

    Raises ConfigError with a readable validation message on failure.
    """
    p = Path(path)

    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")

    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read config file: {p}") from e

    try:
        data = yaml.safe_load(raw_text)
    except Exception as e:  # PyYAML can raise multiple exception types
        raise ConfigError(f"Failed to parse YAML in {p}: {e}") from e

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML in {p} must be a mapping/object")

    try:
        return AppConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_pydantic_errors(e, p)) from e


def resolve_runtime_secrets(
    config: AppConfig, *, environ: Mapping[str, str] | None = None
) -> RuntimeSecrets:
    """
    Validate that required environment variables are present and non-empty.

    Returns RuntimeSecrets for callers that need the actual values.
    """
    env = os.environ if environ is None else environ

    apify_env = config.apify.token_env
    openai_env = config.openai.api_key_env

    missing: list[str] = []
    if not (env.get(apify_env) or "").strip():
        missing.append(apify_env)
    if not (env.get(openai_env) or "").strip():
        missing.append(openai_env)

    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required environment variables: {joined}")

    return RuntimeSecrets(
        apify_token=env[apify_env].strip(),
        openai_api_key=env[openai_env].strip(),
    )


def config_sha256(config: AppConfig) -> str:
    """
    Compute a stable SHA-256 hash of the config values for reproducibility.
    """
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _format_pydantic_errors(err: ValidationError, path: Path) -> str:
    lines: list[str] = [f"Invalid configuration in {path}:"]
    for item in err.errors():
        loc = ".".join(str(part) for part in item.get("loc", [])) or "<root>"
        msg = item.get("msg", "invalid value")
        lines.append(f"- {loc}: {msg}")
    return "\n".join(lines)

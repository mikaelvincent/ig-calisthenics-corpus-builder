from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_term_list(values: list[str], *, allow_empty: bool) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for item in values:
        term = (item or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)

    if not allow_empty and not out:
        raise ValueError("must contain at least one non-empty term")
    return out


def _validate_env_var_name(value: str) -> str:
    name = (value or "").strip()
    if not _ENV_NAME_RE.fullmatch(name):
        raise ValueError("must be a valid environment variable name")
    return name


PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class TargetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    final_n: PositiveInt = 500
    pool_n: PositiveInt = 650
    sampling_seed: int = 1337

    @model_validator(mode="after")
    def _pool_must_cover_final(self) -> "TargetsConfig":
        if self.pool_n < self.final_n:
            raise ValueError("pool_n must be >= final_n")
        return self


class ApifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_env: str = "APIFY_TOKEN"
    primary_actor: str = "apify/instagram-hashtag-scraper"
    fallback_actor: str = "apify/instagram-scraper"
    results_type: Literal["posts"] = "posts"
    results_limit_per_query: PositiveInt = 150
    keyword_search: bool = True
    run_batch_queries: PositiveInt = 4

    @field_validator("token_env")
    @classmethod
    def _token_env_must_be_valid(cls, v: str) -> str:
        return _validate_env_var_name(v)


class OpenAIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key_env: str = "OPENAI_API_KEY"
    model_primary: str = "gpt-5-nano"
    model_escalation: str = "gpt-5-mini"
    escalation_confidence_threshold: float = Field(0.70, ge=0.0, le=1.0)
    max_output_tokens: PositiveInt = 650

    @field_validator("api_key_env")
    @classmethod
    def _api_key_env_must_be_valid(cls, v: str) -> str:
        return _validate_env_var_name(v)


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_caption_chars: NonNegativeInt = 40
    max_posts_per_user: NonNegativeInt = 10  # 0 disables the cap
    allow_reels: bool = True
    reject_if_sponsored_true: bool = False


class LoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: PositiveInt = 200
    stagnation_window: PositiveInt = 10
    stagnation_min_new_eligible: NonNegativeInt = 15
    max_raw_items: PositiveInt = 20000
    backoff_seconds: NonNegativeInt = 10


class ExpansionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    max_new_terms_per_iter: PositiveInt = 15
    min_hashtag_freq_in_eligible: PositiveInt = 4
    blocklist_terms: list[str] = Field(default_factory=list)

    @field_validator("blocklist_terms")
    @classmethod
    def _normalize_blocklist(cls, v: list[str]) -> list[str]:
        return _normalize_term_list(v, allow_empty=True)


class QueryingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed_terms: list[str] = Field(
        default_factory=lambda: [
            "calisthenics",
            "streetworkout",
            "bodyweighttraining",
            "bodyweightworkout",
        ]
    )
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)

    @field_validator("seed_terms")
    @classmethod
    def _normalize_seed_terms(cls, v: list[str]) -> list[str]:
        return _normalize_term_list(v, allow_empty=False)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    apify: ApifyConfig = Field(default_factory=ApifyConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    querying: QueryingConfig = Field(default_factory=QueryingConfig)

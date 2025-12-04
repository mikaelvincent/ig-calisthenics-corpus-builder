# ig_corpus/dry_run.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .apify_client import InstagramHashtagScraper
from .config import RuntimeSecrets
from .config_schema import AppConfig
from .dedupe import SeenKeys, dedupe_key
from .eligibility import enforce_structured_eligibility
from .llm import OpenAIPostClassifier
from .llm_schema import LLMDecision
from .normalize import normalized_post_from_apify_item, post_for_llm
from .prechecks import run_prechecks


_DRY_RUN_RESULTS_LIMIT = 5
_DRY_RUN_LLM_ITEMS = 3


@dataclass(frozen=True)
class DryRunResult:
    query_term: str
    scraped_count: int
    processed_count: int
    eligible_count: int
    example_decision: dict[str, Any]


def _pick_query_term(config: AppConfig, override: str | None) -> str:
    if override and override.strip():
        return override.strip()

    seed_terms = config.querying.seed_terms
    if not seed_terms:
        raise ValueError("querying.seed_terms must contain at least one term")

    term = (seed_terms[0] or "").strip()
    if not term:
        raise ValueError("querying.seed_terms[0] must be non-empty")

    return term


def _redact_decision_for_print(url: str, decision: LLMDecision) -> dict[str, Any]:
    return {
        "url": url,
        "eligible": decision.eligible,
        "eligibility_reasons": decision.eligibility_reasons,
        "language": decision.language.model_dump(mode="json"),
        "topic": decision.topic.model_dump(mode="json"),
        "commercial": decision.commercial.model_dump(mode="json"),
        "caption_quality": decision.caption_quality.model_dump(mode="json"),
        "tags": decision.tags.model_dump(mode="json"),
        "overall_confidence": decision.overall_confidence,
    }


def run_dry_run(
    config: AppConfig,
    secrets: RuntimeSecrets,
    *,
    query_term: str | None = None,
    scraper: InstagramHashtagScraper | None = None,
    classifier: OpenAIPostClassifier | None = None,
) -> DryRunResult:
    term = _pick_query_term(config, query_term)

    apify_cfg = config.apify.model_copy(
        update={
            "results_limit_per_query": _DRY_RUN_RESULTS_LIMIT,
            "run_batch_queries": 1,
        }
    )

    apify_scraper = scraper or InstagramHashtagScraper(secrets.apify_token)
    _, items = apify_scraper.run_and_fetch(
        [term],
        apify=apify_cfg,
        dataset_limit=_DRY_RUN_RESULTS_LIMIT,
        clean=True,
    )

    post_classifier = classifier or OpenAIPostClassifier(
        secrets.openai_api_key,
        openai_cfg=config.openai,
    )

    seen = SeenKeys()
    processed = 0
    eligible = 0
    example: dict[str, Any] | None = None

    for item in items:
        normalized = normalized_post_from_apify_item(item)
        if normalized is None:
            continue

        key = dedupe_key(normalized)
        if seen.has(key):
            continue
        seen.add(key)

        checks = run_prechecks(normalized, filters=config.filters)
        if not checks.passed:
            continue

        decision = post_classifier.classify(post_for_llm(normalized))
        decision = enforce_structured_eligibility(decision)
        processed += 1
        if decision.eligible:
            eligible += 1

        if example is None:
            example = _redact_decision_for_print(normalized.url, decision)

        if processed >= _DRY_RUN_LLM_ITEMS:
            break

    if processed <= 0 or example is None:
        raise RuntimeError("Dry-run did not produce any LLM decisions")

    return DryRunResult(
        query_term=term,
        scraped_count=len(items),
        processed_count=processed,
        eligible_count=eligible,
        example_decision=example,
    )

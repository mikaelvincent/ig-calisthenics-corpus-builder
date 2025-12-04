from __future__ import annotations

from dataclasses import dataclass

from .codebook import CodebookData
from .config_schema import AppConfig


@dataclass(frozen=True)
class MethodsOverview:
    paragraphs: tuple[str, ...]
    steps: tuple[str, ...]


def _count_actor_runs(data: CodebookData, *, actor_id: str) -> int:
    key = (actor_id or "").strip()
    if not key:
        return 0
    return sum(1 for r in data.actor_runs if str(r.actor_id) == key)


def build_methods_overview(config: AppConfig, data: CodebookData) -> MethodsOverview:
    """
    Create a short, config-driven methods narrative for the PDF codebook.

    This is intentionally descriptive rather than technical: it mirrors what a reader
    needs to understand how posts were collected and filtered.
    """
    primary_actor = (config.apify.primary_actor or "").strip() or "apify/instagram-hashtag-scraper"
    fallback_actor = (config.apify.fallback_actor or "").strip() or "apify/instagram-scraper"

    keyword_search = "enabled" if bool(config.apify.keyword_search) else "disabled"
    results_type = (config.apify.results_type or "").strip() or "posts"
    results_limit = int(config.apify.results_limit_per_query)
    batch_terms = int(config.apify.run_batch_queries)

    seed_terms_n = int(len(config.querying.seed_terms))
    exp_enabled = bool(config.querying.expansion.enabled)
    exp_max_terms = int(config.querying.expansion.max_new_terms_per_iter)
    exp_min_freq = int(config.querying.expansion.min_hashtag_freq_in_eligible)
    exp_blocklist_n = int(len(config.querying.expansion.blocklist_terms))

    min_caption_chars = int(config.filters.min_caption_chars)
    allow_reels = bool(config.filters.allow_reels)
    max_posts_per_user = int(config.filters.max_posts_per_user)

    model_primary = (config.openai.model_primary or "").strip() or "gpt-5-nano"
    model_escalation = (config.openai.model_escalation or "").strip()
    esc_threshold = float(config.openai.escalation_confidence_threshold)
    openai_concurrency = int(config.openai.max_concurrent_requests)

    stagnation_window = int(config.loop.stagnation_window)
    stagnation_min_new = int(config.loop.stagnation_min_new_eligible)

    total_runs = int(len(data.actor_runs))
    primary_runs = _count_actor_runs(data, actor_id=primary_actor)
    fallback_runs = _count_actor_runs(data, actor_id=fallback_actor)

    runs_sentence = ""
    if total_runs > 0:
        runs_sentence = (
            f"Actor execution metadata recorded {total_runs} runs "
            f"({primary_runs} primary, {fallback_runs} fallback)."
        )

    escalation_clause = ""
    if model_escalation and model_escalation != model_primary:
        escalation_clause = (
            f" Low-confidence cases can be re-labeled with `{model_escalation}` when "
            f"`overall_confidence < {esc_threshold:.2f}`."
        )

    expansion_clause = (
        f"Query expansion is {'enabled' if exp_enabled else 'disabled'}."
        + (
            f" When enabled, up to {exp_max_terms} new hashtag terms are enqueued per iteration "
            f"when they appear at least {exp_min_freq} times in the eligible pool "
            f"(blocklist size: {exp_blocklist_n})."
            if exp_enabled
            else ""
        )
    )

    paragraphs: list[str] = [
        (
            "Public Instagram posts are collected via Apify Actor runs. "
            f"The primary collector is `{primary_actor}` using `keywordSearch` {keyword_search}, "
            f"requesting `{results_type}` with up to {results_limit} results per query term "
            f"(processed in batches of {batch_terms} terms per Actor run)."
        ),
        (
            "The build proceeds as an iterative feedback loop: scrape → normalize/dedupe → "
            "deterministic prechecks → one-post-per-call LLM labeling → pool update → "
            "query-queue expansion. "
            f"Stagnation is detected over a rolling window of {stagnation_window} iterations; "
            f"if fewer than {stagnation_min_new} new eligible posts accumulate over that window, "
            f"the fallback discovery path can invoke `{fallback_actor}` to search for additional hashtags "
            "and scrape the corresponding hashtag URLs."
        ),
        (
            "Each candidate post is labeled using Structured Outputs so results conform to a strict JSON schema. "
            f"The primary labeling model is `{model_primary}`."
            f"{escalation_clause}"
            f" Labeling runs with up to {openai_concurrency} concurrent OpenAI requests."
        ),
        (
            "Deterministic gatekeeping complements model outputs: captions must meet basic analyzability thresholds "
            f"(minimum caption length {min_caption_chars} characters), reels are "
            f"{'allowed' if allow_reels else 'rejected'}, and a dominance guard can cap per-user contributions "
            f"(max_posts_per_user={max_posts_per_user}, where 0 disables the cap). "
            f"{expansion_clause} "
            f"{runs_sentence}".strip()
        ),
    ]

    steps: list[str] = [
        f"Initialize a FIFO query queue from {seed_terms_n} seed terms.",
        (
            f"Run `{primary_actor}` on a batch of up to {batch_terms} terms; fetch dataset items and normalize fields."
        ),
        "Deduplicate posts by stable keys (id/shortcode/url) and apply fast prechecks to avoid unnecessary LLM calls.",
        (
            f"Label candidates with `{model_primary}` using a strict JSON schema (up to {openai_concurrency} concurrent calls); "
            "compute eligibility from structured fields and apply the per-user dominance guard."
        ),
        (
            "Add eligible posts to the pool and harvest hashtags from eligible items to expand the query queue "
            "(subject to frequency thresholds and a blocklist)."
            if exp_enabled
            else "Add eligible posts to the pool (query expansion disabled)."
        ),
        (
            f"On stagnation, use `{fallback_actor}` to discover and scrape additional hashtags, then continue iterating."
        ),
        (
            "Stop when the eligible pool target is met, then sample the final corpus deterministically using "
            "the configured sampling seed."
        ),
    ]

    return MethodsOverview(
        paragraphs=tuple(p for p in paragraphs if (p or "").strip()),
        steps=tuple(s for s in steps if (s or "").strip()),
    )

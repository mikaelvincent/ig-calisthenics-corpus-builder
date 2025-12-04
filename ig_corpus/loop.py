from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable
from urllib.parse import urlsplit

from .apify_client import ActorRunRef, InstagramHashtagScraper, InstagramScraper
from .config import RuntimeSecrets, config_sha256
from .config_schema import AppConfig
from .dedupe import dedupe_key
from .llm import OpenAIPostClassifier
from .llm_schema import LLMDecision
from .normalize import normalized_post_from_apify_item, post_for_llm
from .prechecks import run_prechecks
from .query_queue import TermQueue, normalize_term
from .stagnation import StagnationTracker
from .storage import SQLiteStateStore


@dataclass(frozen=True)
class FeedbackLoopResult:
    run_id: str
    status: str
    iterations: int
    raw_posts: int
    decisions: int
    eligible: int


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def _db_count(store: SQLiteStateStore, sql: str) -> int:
    row = store.conn.execute(sql).fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except Exception:
        return 0


def _eligible_count(store: SQLiteStateStore) -> int:
    return _db_count(store, "SELECT COUNT(1) FROM eligible_posts")


def _decision_count(store: SQLiteStateStore) -> int:
    return _db_count(store, "SELECT COUNT(1) FROM llm_decisions")


def _raw_count(store: SQLiteStateStore) -> int:
    return _db_count(store, "SELECT COUNT(1) FROM raw_posts")


def _extract_hashtag_from_url(url: str) -> str | None:
    u = (url or "").strip()
    if not u:
        return None

    try:
        parts = urlsplit(u)
    except Exception:
        return None

    path = (parts.path or "").strip("/")
    if not path:
        return None

    segs = [s for s in path.split("/") if s]
    for i in range(len(segs) - 2):
        if segs[i] == "explore" and segs[i + 1] == "tags":
            tag = (segs[i + 2] or "").strip()
            return tag or None

    return None


def _extract_hashtag_search_urls(items: Iterable[dict[str, Any]]) -> list[str]:
    keys = (
        "url",
        "pageUrl",
        "page_url",
        "hashtagUrl",
        "hashtag_url",
        "tagUrl",
        "tag_url",
    )

    out: list[str] = []
    seen: set[str] = set()

    for item in items:
        for k in keys:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                url = v.strip()
                key = url.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(url)

        v_urls = item.get("urls")
        if isinstance(v_urls, list):
            for v in v_urls:
                if isinstance(v, str) and v.strip():
                    url = v.strip()
                    key = url.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(url)

    return out


def _load_existing_counters(store: SQLiteStateStore) -> tuple[Counter[str], Counter[str]]:
    hashtag_counts: Counter[str] = Counter()
    user_counts: Counter[str] = Counter()

    rows = store.conn.execute(
        """
        SELECT p.raw_json
        FROM raw_posts p
        JOIN latest_llm_decisions d ON d.post_key = p.post_key
        WHERE d.eligible = 1
        """.strip()
    ).fetchall()

    for r in rows:
        raw = (r[0] or "").strip()
        if not raw:
            continue

        try:
            item = json.loads(raw)
        except Exception:
            continue

        if not isinstance(item, dict):
            continue

        post = normalized_post_from_apify_item(item)
        if post is None:
            continue

        for tag in post.hashtags:
            hashtag_counts[tag] += 1

        user_key = None
        if post.owner_username:
            user_key = f"user:{post.owner_username.casefold()}"
        elif post.owner_id:
            user_key = f"user_id:{post.owner_id}"
        if user_key:
            user_counts[user_key] += 1

    return hashtag_counts, user_counts


def _selected_expansion_terms(
    hashtag_counts: Counter[str],
    *,
    min_freq: int,
    max_terms: int,
    blocklist_keys: set[str],
    attempted_keys: set[str],
    present_keys: set[str],
) -> list[str]:
    if max_terms <= 0:
        return []

    candidates: list[tuple[int, str]] = []
    for term, freq in hashtag_counts.items():
        key = term.casefold()
        if key in blocklist_keys:
            continue
        if key in attempted_keys:
            continue
        if key in present_keys:
            continue
        candidates.append((int(freq), term))

    candidates.sort(key=lambda x: (-x[0], x[1].casefold(), x[1]))

    out: list[str] = []
    for freq, term in candidates:
        if freq < min_freq:
            break
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def _apply_dominance_guard(
    decision: LLMDecision,
    *,
    owner_username: str | None,
    owner_id: str | None,
    max_posts_per_user: int,
    user_counts: Counter[str],
) -> LLMDecision:
    if not decision.eligible:
        return decision
    if max_posts_per_user <= 0:
        return decision

    user_key = None
    if owner_username:
        user_key = f"user:{owner_username.casefold()}"
    elif owner_id:
        user_key = f"user_id:{owner_id}"

    if not user_key:
        return decision

    if user_counts[user_key] >= max_posts_per_user:
        return decision.model_copy(
            update={
                "eligible": False,
                "eligibility_reasons": list(decision.eligibility_reasons) + ["dominance_guard"],
            }
        )

    user_counts[user_key] += 1
    return decision


def run_feedback_loop(
    config: AppConfig,
    secrets: RuntimeSecrets,
    *,
    store: SQLiteStateStore,
    scraper: InstagramHashtagScraper | None = None,
    fallback_scraper: InstagramScraper | None = None,
    classifier: OpenAIPostClassifier | None = None,
) -> FeedbackLoopResult:
    cfg_hash = config_sha256(config)
    versions = {
        "python": sys.version.split()[0],
        "apify-client": _pkg_version("apify-client"),
        "openai": _pkg_version("openai"),
        "pydantic": _pkg_version("pydantic"),
    }

    run_record = store.create_run(
        config_hash=cfg_hash,
        sampling_seed=config.targets.sampling_seed,
        versions=versions,
    )

    primary = scraper or InstagramHashtagScraper(secrets.apify_token)
    fallback = fallback_scraper or InstagramScraper(secrets.apify_token)
    post_classifier = classifier or OpenAIPostClassifier(
        secrets.openai_api_key,
        openai_cfg=config.openai,
    )

    eligible_total = _eligible_count(store)
    raw_total = _raw_count(store)
    decision_total = _decision_count(store)

    seen_keys = store.seen_post_keys()
    hashtag_counts, user_counts = _load_existing_counters(store)

    blocklist_keys = {normalize_term(t).casefold() for t in config.querying.expansion.blocklist_terms}
    attempted_keys: set[str] = set()

    queue = TermQueue(config.querying.seed_terms)

    stagnation = StagnationTracker(
        window_size=config.loop.stagnation_window,
        min_new_total=config.loop.stagnation_min_new_eligible,
    )

    current_results_limit = int(config.apify.results_limit_per_query)
    max_results_limit = 500

    def _record_actor_run(run_ref: ActorRunRef) -> None:
        store.record_apify_actor_run(
            run_id=run_record.run_id,
            actor_id=run_ref.actor_id,
            actor_run_id=run_ref.run_id,
            dataset_id=run_ref.default_dataset_id,
        )

    def _ingest_post_items(items: list[dict[str, Any]], *, actor_source: str) -> tuple[int, int]:
        nonlocal eligible_total, raw_total, decision_total

        new_eligible = 0
        processed = 0

        for item in items:
            post = normalized_post_from_apify_item(item)
            if post is None:
                continue

            post_key = dedupe_key(post)
            if post_key in seen_keys:
                continue

            if raw_total >= config.loop.max_raw_items:
                break

            store.upsert_raw_post(
                post_key=post_key,
                url=post.url,
                actor_source=actor_source,
                raw_item=item,
            )
            seen_keys.add(post_key)
            raw_total += 1

            checks = run_prechecks(post, filters=config.filters)
            if not checks.passed:
                continue

            decision, model_used = post_classifier.classify_with_metadata(post_for_llm(post))
            decision = _apply_dominance_guard(
                decision,
                owner_username=post.owner_username,
                owner_id=post.owner_id,
                max_posts_per_user=int(config.filters.max_posts_per_user),
                user_counts=user_counts,
            )

            store.record_llm_decision(
                post_key=post_key,
                url=post.url,
                model=model_used,
                decision=decision,
            )
            decision_total += 1
            processed += 1

            if decision.eligible:
                eligible_total += 1
                new_eligible += 1
                for tag in post.hashtags:
                    hashtag_counts[tag] += 1

            if eligible_total >= config.targets.pool_n:
                break

        return processed, new_eligible

    for iteration in range(int(config.loop.max_iterations)):
        if eligible_total >= config.targets.pool_n:
            store.finish_run(run_record.run_id)
            return FeedbackLoopResult(
                run_id=run_record.run_id,
                status="completed_pool",
                iterations=iteration,
                raw_posts=raw_total,
                decisions=decision_total,
                eligible=eligible_total,
            )

        if raw_total >= config.loop.max_raw_items:
            store.finish_run(run_record.run_id)
            return FeedbackLoopResult(
                run_id=run_record.run_id,
                status="max_raw_items",
                iterations=iteration,
                raw_posts=raw_total,
                decisions=decision_total,
                eligible=eligible_total,
            )

        batch = queue.pop_batch(int(config.apify.run_batch_queries))
        if not batch:
            queue.add_many(config.querying.seed_terms)
            batch = queue.pop_batch(int(config.apify.run_batch_queries))

        if not batch:
            store.finish_run(run_record.run_id)
            return FeedbackLoopResult(
                run_id=run_record.run_id,
                status="empty_query_queue",
                iterations=iteration,
                raw_posts=raw_total,
                decisions=decision_total,
                eligible=eligible_total,
            )

        for b in batch:
            attempted_keys.add(b.casefold())

        apify_cfg = config.apify.model_copy(update={"results_limit_per_query": int(current_results_limit)})

        run_ref, items = primary.run_and_fetch(
            batch,
            apify=apify_cfg,
            dataset_limit=None,
            clean=True,
        )
        _record_actor_run(run_ref)

        _, new_eligible_primary = _ingest_post_items(items, actor_source=run_ref.actor_id)

        if config.querying.expansion.enabled:
            new_terms = _selected_expansion_terms(
                hashtag_counts,
                min_freq=int(config.querying.expansion.min_hashtag_freq_in_eligible),
                max_terms=int(config.querying.expansion.max_new_terms_per_iter),
                blocklist_keys=blocklist_keys,
                attempted_keys=attempted_keys,
                present_keys=queue.present_keys(),
            )
            queue.add_many(new_terms)

        stagnated = stagnation.push(new_eligible_primary)

        if stagnated and eligible_total < config.targets.pool_n:
            current_results_limit = min(max_results_limit, max(10, current_results_limit * 2))

            search_seeds = _selected_expansion_terms(
                hashtag_counts,
                min_freq=1,
                max_terms=3,
                blocklist_keys=blocklist_keys,
                attempted_keys=set(),
                present_keys=set(),
            )
            if not search_seeds:
                search_seeds = [t for t in config.querying.seed_terms if normalize_term(t)]

            discovered_urls: list[str] = []
            for seed in search_seeds[:3]:
                search_run, search_items = fallback.search_hashtags_and_fetch(
                    seed,
                    apify=apify_cfg,
                    search_limit=20,
                    dataset_limit=50,
                    clean=True,
                )
                _record_actor_run(search_run)

                urls = _extract_hashtag_search_urls(search_items)
                discovered_urls.extend(urls)

            deduped_urls: list[str] = []
            seen_url_keys: set[str] = set()
            for u in discovered_urls:
                k = (u or "").strip().casefold()
                if not k or k in seen_url_keys:
                    continue
                seen_url_keys.add(k)
                deduped_urls.append(u)

            discovered_terms: list[str] = []
            for u in deduped_urls:
                tag = _extract_hashtag_from_url(u)
                if tag:
                    key = tag.casefold()
                    if key not in blocklist_keys and key not in attempted_keys:
                        discovered_terms.append(tag)

            queue.add_many(discovered_terms)

            scrape_urls = [u for u in deduped_urls if _extract_hashtag_from_url(u)]
            scrape_urls = scrape_urls[:10]
            if scrape_urls:
                scrape_run, scrape_items = fallback.scrape_urls_and_fetch(
                    scrape_urls,
                    apify=apify_cfg,
                    results_limit=min(50, int(current_results_limit)),
                    dataset_limit=None,
                    clean=True,
                )
                _record_actor_run(scrape_run)
                _, _ = _ingest_post_items(scrape_items, actor_source=scrape_run.actor_id)

        if config.loop.backoff_seconds > 0:
            time.sleep(float(config.loop.backoff_seconds))

    store.finish_run(run_record.run_id)
    return FeedbackLoopResult(
        run_id=run_record.run_id,
        status="max_iterations",
        iterations=int(config.loop.max_iterations),
        raw_posts=raw_total,
        decisions=decision_total,
        eligible=eligible_total,
    )

from __future__ import annotations

import unittest
from typing import Any

from ig_corpus.apify_client import ActorRunRef
from ig_corpus.config import RuntimeSecrets
from ig_corpus.config_schema import (
    AppConfig,
    ApifyConfig,
    ExpansionConfig,
    FiltersConfig,
    LoopConfig,
    OpenAIConfig,
    QueryingConfig,
    TargetsConfig,
)
from ig_corpus.llm import PostForLLM
from ig_corpus.llm_schema import LLMDecision
from ig_corpus.loop import run_feedback_loop
from ig_corpus.storage import SQLiteStateStore


def _decision(*, eligible: bool) -> LLMDecision:
    payload = {
        "eligible": eligible,
        "eligibility_reasons": ["ok" if eligible else "reject"],
        "language": {"is_english": True, "confidence": 0.9},
        "topic": {
            "is_bodyweight_calisthenics": True,
            "confidence": 0.9,
            "topic_notes": "test",
        },
        "commercial": {"is_exclusively_commercial": False, "signals": []},
        "caption_quality": {"is_analyzable": True, "issues": []},
        "tags": {
            "genre": "training_log",
            "narrative_labels": [],
            "discourse_moves": [],
            "neoliberal_signals": [],
        },
        "overall_confidence": 0.9,
    }
    return LLMDecision.model_validate(payload)


_LONG_CAPTION = "x" * 80


class _FakePrimaryScraper:
    def __init__(self, items_by_term: dict[str, list[dict[str, Any]]]) -> None:
        self._items_by_term = items_by_term
        self.calls: list[list[str]] = []
        self._run_no = 0

    def run_and_fetch(self, terms: Any, *, apify: Any, timeout_secs: Any = None, dataset_limit: Any = None, clean: Any = True) -> Any:
        self._run_no += 1
        batch = list(terms)
        self.calls.append(batch)

        items: list[dict[str, Any]] = []
        for t in batch:
            items.extend(list(self._items_by_term.get(t, [])))

        return (
            ActorRunRef(actor_id=apify.primary_actor, run_id=f"run_{self._run_no}", default_dataset_id=f"ds_{self._run_no}"),
            items,
        )


class _FakeFallbackScraper:
    def __init__(self, *, search_items: list[dict[str, Any]], scrape_items: list[dict[str, Any]]) -> None:
        self._search_items = search_items
        self._scrape_items = scrape_items
        self.search_calls: list[str] = []
        self.scrape_calls: list[list[str]] = []
        self._run_no = 0

    def search_hashtags_and_fetch(self, query: str, *, apify: Any, search_limit: int = 20, timeout_secs: Any = None, dataset_limit: Any = None, clean: bool = True) -> Any:
        self._run_no += 1
        self.search_calls.append(query)
        return (
            ActorRunRef(actor_id=apify.fallback_actor, run_id=f"search_{self._run_no}", default_dataset_id=f"ds_search_{self._run_no}"),
            list(self._search_items),
        )

    def scrape_urls_and_fetch(self, urls: Any, *, apify: Any, results_limit: int, timeout_secs: Any = None, dataset_limit: Any = None, clean: bool = True) -> Any:
        self._run_no += 1
        batch = list(urls)
        self.scrape_calls.append(batch)
        return (
            ActorRunRef(actor_id=apify.fallback_actor, run_id=f"scrape_{self._run_no}", default_dataset_id=f"ds_scrape_{self._run_no}"),
            list(self._scrape_items),
        )


class _FakeClassifier:
    def __init__(self, eligibles: list[bool]) -> None:
        self._eligibles = list(eligibles)
        self.calls: list[str] = []

    def classify_with_metadata(self, post: PostForLLM) -> tuple[LLMDecision, str, int | None]:
        self.calls.append(post.url)
        eligible = self._eligibles.pop(0) if self._eligibles else False
        return _decision(eligible=eligible), "gpt-5-nano", None


class TestFeedbackLoop(unittest.TestCase):
    def test_expands_queue_from_eligible_hashtags(self) -> None:
        cfg = AppConfig(
            targets=TargetsConfig(final_n=1, pool_n=2, sampling_seed=1),
            apify=ApifyConfig(
                token_env="APIFY_TOKEN",
                primary_actor="apify/instagram-hashtag-scraper",
                fallback_actor="apify/instagram-scraper",
                results_type="posts",
                results_limit_per_query=5,
                keyword_search=True,
                run_batch_queries=1,
            ),
            openai=OpenAIConfig(max_output_tokens=50),
            filters=FiltersConfig(min_caption_chars=40, max_posts_per_user=0, allow_reels=True, reject_if_sponsored_true=False),
            loop=LoopConfig(max_iterations=10, stagnation_window=3, stagnation_min_new_eligible=1, max_raw_items=1000, backoff_seconds=0),
            querying=QueryingConfig(
                seed_terms=["t1"],
                expansion=ExpansionConfig(enabled=True, max_new_terms_per_iter=10, min_hashtag_freq_in_eligible=1, blocklist_terms=[]),
            ),
        )

        secrets = RuntimeSecrets(apify_token="apify", openai_api_key="openai")

        primary_items = {
            "t1": [{"url": "https://example.com/p/1", "caption": _LONG_CAPTION, "hashtags": ["newtag"]}],
            "newtag": [{"url": "https://example.com/p/2", "caption": _LONG_CAPTION, "hashtags": ["newtag"]}],
        }

        scraper = _FakePrimaryScraper(primary_items)
        classifier = _FakeClassifier([True, True])

        with SQLiteStateStore.open(":memory:") as store:
            result = run_feedback_loop(cfg, secrets, store=store, scraper=scraper, classifier=classifier)

        self.assertEqual(result.status, "completed_pool")
        self.assertEqual(scraper.calls, [["t1"], ["newtag"]])

    def test_stagnation_invokes_fallback(self) -> None:
        cfg = AppConfig(
            targets=TargetsConfig(final_n=1, pool_n=1, sampling_seed=1),
            apify=ApifyConfig(
                token_env="APIFY_TOKEN",
                primary_actor="apify/instagram-hashtag-scraper",
                fallback_actor="apify/instagram-scraper",
                results_type="posts",
                results_limit_per_query=5,
                keyword_search=True,
                run_batch_queries=1,
            ),
            openai=OpenAIConfig(max_output_tokens=50),
            filters=FiltersConfig(min_caption_chars=40, max_posts_per_user=0, allow_reels=True, reject_if_sponsored_true=False),
            loop=LoopConfig(max_iterations=5, stagnation_window=2, stagnation_min_new_eligible=1, max_raw_items=1000, backoff_seconds=0),
            querying=QueryingConfig(
                seed_terms=["t1", "t2"],
                expansion=ExpansionConfig(enabled=True, max_new_terms_per_iter=10, min_hashtag_freq_in_eligible=1, blocklist_terms=[]),
            ),
        )

        secrets = RuntimeSecrets(apify_token="apify", openai_api_key="openai")

        # Primary yields only short captions, failing prechecks => eligible growth stays 0.
        primary_items = {
            "t1": [{"url": "https://example.com/p/a", "caption": "short"}],
            "t2": [{"url": "https://example.com/p/b", "caption": "short"}],
        }

        scraper = _FakePrimaryScraper(primary_items)

        fallback = _FakeFallbackScraper(
            search_items=[{"url": "https://www.instagram.com/explore/tags/fallbacktag/"}],
            scrape_items=[{"url": "https://example.com/p/fb1", "caption": _LONG_CAPTION, "hashtags": ["fallbacktag"]}],
        )

        classifier = _FakeClassifier([True])

        with SQLiteStateStore.open(":memory:") as store:
            result = run_feedback_loop(
                cfg,
                secrets,
                store=store,
                scraper=scraper,
                fallback_scraper=fallback,
                classifier=classifier,
            )

        self.assertEqual(result.status, "completed_pool")
        self.assertGreaterEqual(len(fallback.search_calls), 1)
        self.assertGreaterEqual(len(fallback.scrape_calls), 1)
        self.assertEqual(classifier.calls, ["https://example.com/p/fb1"])


if __name__ == "__main__":
    unittest.main()

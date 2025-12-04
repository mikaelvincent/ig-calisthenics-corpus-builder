# tests/test_dry_run.py
from __future__ import annotations

import unittest
from typing import Any

from ig_corpus.apify_client import ActorRunRef
from ig_corpus.config import RuntimeSecrets
from ig_corpus.config_schema import AppConfig
from ig_corpus.dry_run import run_dry_run
from ig_corpus.llm import PostForLLM
from ig_corpus.llm_schema import LLMDecision


def _decision(*, eligible: bool) -> LLMDecision:
    payload = {
        "eligible": eligible,
        "eligibility_reasons": ["ok" if eligible else "reject"],
        "language": {"is_english": True, "confidence": 0.9},
        "topic": {
            "is_bodyweight_calisthenics": bool(eligible),
            "confidence": 0.9 if eligible else 0.1,
            "topic_notes": "test",
        },
        "commercial": {"is_exclusively_commercial": False, "signals": []},
        "caption_quality": {"is_analyzable": True, "issues": []},
        "tags": {
            "genre": "training_log" if eligible else "other",
            "narrative_labels": [],
            "discourse_moves": [],
            "neoliberal_signals": [],
        },
        "overall_confidence": 0.9,
    }
    return LLMDecision.model_validate(payload)


_LONG_CAPTION = "x" * 60


class _FakeScraper:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self.calls: list[dict[str, Any]] = []

    def run_and_fetch(
        self,
        terms: Any,
        *,
        apify: Any,
        timeout_secs: Any = None,
        dataset_limit: Any = None,
        clean: Any = True,
    ) -> Any:
        self.calls.append(
            {
                "terms": list(terms),
                "results_limit_per_query": apify.results_limit_per_query,
                "dataset_limit": dataset_limit,
                "clean": clean,
            }
        )
        return (
            ActorRunRef(
                actor_id=apify.primary_actor,
                run_id="run_1",
                default_dataset_id="ds_1",
            ),
            list(self._items),
        )


class _FakeClassifier:
    def __init__(self, eligibles: list[bool]) -> None:
        self._eligibles = list(eligibles)
        self.calls: list[str] = []

    def classify(self, post: PostForLLM) -> LLMDecision:
        self.calls.append(post.url)
        eligible = self._eligibles.pop(0) if self._eligibles else False
        return _decision(eligible=eligible)


class TestDryRun(unittest.TestCase):
    def test_dry_run_overrides_limits_and_processes_three(self) -> None:
        cfg = AppConfig()
        secrets = RuntimeSecrets(apify_token="apify", openai_api_key="openai")

        items = [
            {"url": "https://example.com/p/1", "caption": _LONG_CAPTION},
            {"url": "https://example.com/p/2", "caption": _LONG_CAPTION},
            {"url": "https://example.com/p/3", "caption": _LONG_CAPTION},
            {"url": "https://example.com/p/4", "caption": _LONG_CAPTION},
        ]

        scraper = _FakeScraper(items)
        classifier = _FakeClassifier([True, False, True])

        result = run_dry_run(cfg, secrets, scraper=scraper, classifier=classifier)

        self.assertEqual(result.scraped_count, 4)
        self.assertEqual(result.processed_count, 3)
        self.assertEqual(result.eligible_count, 2)
        self.assertEqual(
            classifier.calls,
            [
                "https://example.com/p/1",
                "https://example.com/p/2",
                "https://example.com/p/3",
            ],
        )

        self.assertEqual(len(scraper.calls), 1)
        self.assertEqual(scraper.calls[0]["results_limit_per_query"], 5)
        self.assertEqual(scraper.calls[0]["dataset_limit"], 5)

    def test_dry_run_skips_items_without_url(self) -> None:
        cfg = AppConfig()
        secrets = RuntimeSecrets(apify_token="apify", openai_api_key="openai")

        items = [
            {"caption": _LONG_CAPTION},
            {"url": "https://example.com/p/1", "caption": _LONG_CAPTION},
            {"url": "https://example.com/p/2", "caption": _LONG_CAPTION},
            {"url": "https://example.com/p/3", "caption": _LONG_CAPTION},
        ]

        scraper = _FakeScraper(items)
        classifier = _FakeClassifier([True, True, False])

        result = run_dry_run(cfg, secrets, scraper=scraper, classifier=classifier)

        self.assertEqual(result.scraped_count, 4)
        self.assertEqual(result.processed_count, 3)
        self.assertEqual(
            classifier.calls,
            [
                "https://example.com/p/1",
                "https://example.com/p/2",
                "https://example.com/p/3",
            ],
        )


if __name__ == "__main__":
    unittest.main()

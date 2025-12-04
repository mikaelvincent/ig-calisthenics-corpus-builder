# tests/test_resume.py
from __future__ import annotations

import unittest
from typing import Any

from ig_corpus.apify_client import ActorRunRef
from ig_corpus.config import RuntimeSecrets, config_sha256
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


_LONG_CAPTION = "x" * 80


class _FakePrimaryScraper:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self.calls: list[list[str]] = []
        self._run_no = 0

    def run_and_fetch(
        self,
        terms: Any,
        *,
        apify: Any,
        timeout_secs: Any = None,
        dataset_limit: Any = None,
        clean: Any = True,
    ) -> Any:
        _ = (timeout_secs, dataset_limit, clean)
        self._run_no += 1
        batch = list(terms)
        self.calls.append(batch)

        return (
            ActorRunRef(
                actor_id=apify.primary_actor,
                run_id=f"run_{self._run_no}",
                default_dataset_id=f"ds_{self._run_no}",
            ),
            list(self._items),
        )


class _FakeClassifier:
    def __init__(self, eligibles: list[bool]) -> None:
        self._eligibles = list(eligibles)
        self.calls: list[str] = []

    def classify_with_metadata(self, post: PostForLLM) -> tuple[LLMDecision, str, int | None]:
        self.calls.append(post.url)
        eligible = self._eligibles.pop(0) if self._eligibles else False
        return _decision(eligible=eligible), "gpt-5-nano", None


class TestResume(unittest.TestCase):
    def test_resume_selects_latest_unfinished_run(self) -> None:
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
            filters=FiltersConfig(
                min_caption_chars=40,
                max_posts_per_user=0,
                allow_reels=True,
                reject_if_sponsored_true=False,
            ),
            loop=LoopConfig(
                max_iterations=5,
                stagnation_window=2,
                stagnation_min_new_eligible=0,
                max_raw_items=1000,
                backoff_seconds=0,
            ),
            querying=QueryingConfig(
                seed_terms=["t1"],
                expansion=ExpansionConfig(enabled=False, max_new_terms_per_iter=0, min_hashtag_freq_in_eligible=1, blocklist_terms=[]),
            ),
        )

        secrets = RuntimeSecrets(apify_token="apify", openai_api_key="openai")

        scraper = _FakePrimaryScraper(
            items=[{"url": "https://example.com/p/1", "caption": _LONG_CAPTION, "hashtags": ["calisthenics"]}]
        )
        classifier = _FakeClassifier([True])

        with SQLiteStateStore.open(":memory:") as store:
            cfg_hash = config_sha256(cfg)

            store.create_run(
                config_hash=cfg_hash,
                sampling_seed=cfg.targets.sampling_seed,
                versions={"python": "3.x"},
                run_id="run_1",
                started_at="2025-12-01T00:00:00+00:00",
            )
            store.create_run(
                config_hash=cfg_hash,
                sampling_seed=cfg.targets.sampling_seed,
                versions={"python": "3.x"},
                run_id="run_2",
                started_at="2025-12-02T00:00:00+00:00",
            )

            result = run_feedback_loop(
                cfg,
                secrets,
                store=store,
                scraper=scraper,
                classifier=classifier,
                resume=True,
            )

            self.assertTrue(result.resumed)
            self.assertEqual(result.run_id, "run_2")

            run1 = store.get_run("run_1")
            run2 = store.get_run("run_2")
            self.assertIsNotNone(run1)
            self.assertIsNotNone(run2)
            assert run1 is not None
            assert run2 is not None

            self.assertIsNone(run1.ended_at)
            self.assertIsNotNone(run2.ended_at)


if __name__ == "__main__":
    unittest.main()

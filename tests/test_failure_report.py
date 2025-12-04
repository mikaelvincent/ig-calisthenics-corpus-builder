from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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
from ig_corpus.failure_report import build_failure_report
from ig_corpus.loop import run_feedback_loop
from ig_corpus.run_log import RunLogger
from ig_corpus.storage import SQLiteStateStore


class TestFailureReport(unittest.TestCase):
    def test_builds_max_iterations_report(self) -> None:
        cfg = AppConfig(
            targets=TargetsConfig(final_n=1, pool_n=10, sampling_seed=1),
            loop=LoopConfig(
                max_iterations=3,
                stagnation_window=2,
                stagnation_min_new_eligible=1,
                max_raw_items=100,
                backoff_seconds=0,
            ),
        )

        report = build_failure_report(
            status="max_iterations",
            config=cfg,
            iterations=3,
            raw_posts=12,
            decisions=5,
            eligible=2,
            recent_new_eligible_total=0,
        )

        self.assertEqual(report["status"], "max_iterations")
        self.assertIn("summary", report)
        self.assertIn("details", report)
        self.assertIn("recommendations", report)
        self.assertIn("max_iterations", report["details"])
        self.assertGreater(len(report["recommendations"]), 0)

    def test_loop_writes_failure_report_event(self) -> None:
        class _FakePrimaryScraper:
            def run_and_fetch(
                self,
                terms: Any,
                *,
                apify: Any,
                timeout_secs: Any = None,
                dataset_limit: Any = None,
                clean: Any = True,
            ) -> Any:
                _ = (terms, timeout_secs, dataset_limit, clean)
                return (
                    ActorRunRef(
                        actor_id=apify.primary_actor,
                        run_id="run_1",
                        default_dataset_id="ds_1",
                    ),
                    [{"url": "https://example.com/p/1", "caption": "short"}],
                )

        class _NeverClassifier:
            def classify_with_metadata(self, post: Any) -> Any:
                raise AssertionError("Classifier should not be called")

        cfg = AppConfig(
            targets=TargetsConfig(final_n=1, pool_n=5, sampling_seed=1),
            apify=ApifyConfig(
                token_env="APIFY_TOKEN",
                primary_actor="apify/instagram-hashtag-scraper",
                fallback_actor="apify/instagram-scraper",
                results_type="posts",
                results_limit_per_query=1,
                keyword_search=True,
                run_batch_queries=1,
            ),
            openai=OpenAIConfig(max_output_tokens=20),
            filters=FiltersConfig(
                min_caption_chars=40,
                max_posts_per_user=0,
                allow_reels=True,
                reject_if_sponsored_true=False,
            ),
            loop=LoopConfig(
                max_iterations=1,
                stagnation_window=5,
                stagnation_min_new_eligible=1,
                max_raw_items=100,
                backoff_seconds=0,
            ),
            querying=QueryingConfig(
                seed_terms=["t1"],
                expansion=ExpansionConfig(
                    enabled=False,
                    max_new_terms_per_iter=1,
                    min_hashtag_freq_in_eligible=1,
                    blocklist_terms=[],
                ),
            ),
        )

        secrets = RuntimeSecrets(apify_token="apify", openai_api_key="openai")

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "run.log"

            with RunLogger.open(log_path, overwrite=True) as logger:
                with SQLiteStateStore.open(":memory:") as store:
                    result = run_feedback_loop(
                        cfg,
                        secrets,
                        store=store,
                        scraper=_FakePrimaryScraper(),
                        classifier=_NeverClassifier(),  # type: ignore[arg-type]
                        logger=logger,
                    )

            self.assertEqual(result.status, "max_iterations")
            self.assertIsNotNone(result.failure_report)

            events: list[str] = []
            for ln in log_path.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                ev = obj.get("event")
                if isinstance(ev, str):
                    events.append(ev)

            self.assertIn("run_failure_report", events)


if __name__ == "__main__":
    unittest.main()

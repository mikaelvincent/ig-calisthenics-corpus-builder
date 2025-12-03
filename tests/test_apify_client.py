from __future__ import annotations

import unittest
from typing import Any


class _FakeDatasetClient:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self.calls: list[dict[str, Any]] = []

    def iterate_items(self, *, limit: int | None = None, clean: bool | None = None) -> Any:
        self.calls.append({"limit": limit, "clean": clean})
        n = len(self._items) if limit is None else min(limit, len(self._items))
        for i in range(n):
            yield self._items[i]


class _FakeActorClient:
    def __init__(self, run_result: dict[str, Any] | None) -> None:
        self._run_result = run_result
        self.calls: list[dict[str, Any]] = []

    def call(self, *, run_input: Any = None, timeout_secs: int | None = None) -> Any:
        self.calls.append({"run_input": run_input, "timeout_secs": timeout_secs})
        return self._run_result


class _FakeApifyClient:
    def __init__(self, *, run_result: dict[str, Any] | None, items: list[dict[str, Any]]) -> None:
        self.actor_ids: list[str] = []
        self.dataset_ids: list[str] = []
        self._actor_client = _FakeActorClient(run_result)
        self._dataset_client = _FakeDatasetClient(items)

    def actor(self, actor_id: str) -> _FakeActorClient:
        self.actor_ids.append(actor_id)
        return self._actor_client

    def dataset(self, dataset_id: str) -> _FakeDatasetClient:
        self.dataset_ids.append(dataset_id)
        return self._dataset_client


class TestInstagramHashtagScraper(unittest.TestCase):
    def test_run_and_fetch_builds_expected_input(self) -> None:
        from ig_corpus.apify_client import InstagramHashtagScraper
        from ig_corpus.config_schema import ApifyConfig

        fake = _FakeApifyClient(
            run_result={"id": "run_1", "defaultDatasetId": "ds_1"},
            items=[{"url": "u1"}, {"url": "u2"}],
        )
        apify_cfg = ApifyConfig(
            token_env="APIFY_TOKEN",
            primary_actor="apify/instagram-hashtag-scraper",
            fallback_actor="apify/instagram-scraper",
            results_type="posts",
            results_limit_per_query=5,
            keyword_search=True,
            run_batch_queries=4,
        )

        scraper = InstagramHashtagScraper("x", client=fake)  # type: ignore[arg-type]
        run, items = scraper.run_and_fetch(
            [" #rainbow ", "rainbow", "  "],
            apify=apify_cfg,
            timeout_secs=60,
        )

        self.assertEqual(run.default_dataset_id, "ds_1")
        self.assertEqual(items, [{"url": "u1"}, {"url": "u2"}])
        self.assertEqual(fake.actor_ids, ["apify/instagram-hashtag-scraper"])
        self.assertEqual(fake.dataset_ids, ["ds_1"])

        call = fake._actor_client.calls[0]
        self.assertEqual(call["timeout_secs"], 60)
        self.assertEqual(
            call["run_input"],
            {
                "hashtags": ["rainbow"],
                "resultsType": "posts",
                "resultsLimit": 5,
                "keywordSearch": True,
            },
        )

    def test_run_and_fetch_many_chunks_terms(self) -> None:
        from ig_corpus.apify_client import InstagramHashtagScraper
        from ig_corpus.config_schema import ApifyConfig

        fake = _FakeApifyClient(
            run_result={"id": "run_1", "defaultDatasetId": "ds_1"},
            items=[],
        )
        apify_cfg = ApifyConfig(
            token_env="APIFY_TOKEN",
            primary_actor="apify/instagram-hashtag-scraper",
            fallback_actor="apify/instagram-scraper",
            results_type="posts",
            results_limit_per_query=1,
            keyword_search=True,
            run_batch_queries=2,
        )
        scraper = InstagramHashtagScraper("x", client=fake)  # type: ignore[arg-type]
        runs, items = scraper.run_and_fetch_many(["a", "b", "c"], apify=apify_cfg)

        self.assertEqual(items, [])
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(fake._actor_client.calls), 2)

        first_input = fake._actor_client.calls[0]["run_input"]
        second_input = fake._actor_client.calls[1]["run_input"]
        self.assertEqual(first_input["hashtags"], ["a", "b"])
        self.assertEqual(second_input["hashtags"], ["c"])

    def test_run_once_raises_on_failed_run(self) -> None:
        from ig_corpus.apify_client import InstagramHashtagScraper
        from ig_corpus.config_schema import ApifyConfig
        from ig_corpus.errors import ApifyError

        fake = _FakeApifyClient(run_result=None, items=[])
        apify_cfg = ApifyConfig(
            token_env="APIFY_TOKEN",
            primary_actor="apify/instagram-hashtag-scraper",
            fallback_actor="apify/instagram-scraper",
            results_type="posts",
            results_limit_per_query=1,
            keyword_search=True,
            run_batch_queries=1,
        )
        scraper = InstagramHashtagScraper("x", client=fake)  # type: ignore[arg-type]
        with self.assertRaises(ApifyError):
            scraper.run_once(["rainbow"], apify=apify_cfg)


if __name__ == "__main__":
    unittest.main()

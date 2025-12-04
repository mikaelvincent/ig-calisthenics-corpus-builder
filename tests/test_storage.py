from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from ig_corpus.llm_schema import LLMDecision
from ig_corpus.storage import SQLiteStateStore


def _decision(*, eligible: bool, confidence: float = 0.9) -> LLMDecision:
    payload: dict[str, Any] = {
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
        "overall_confidence": float(confidence),
    }
    return LLMDecision.model_validate(payload)


class TestSQLiteStateStore(unittest.TestCase):
    def test_creates_and_reads_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "state.sqlite"

            with SQLiteStateStore.open(db_path) as store:
                run = store.create_run(
                    config_hash="abc123",
                    sampling_seed=1337,
                    versions={"python": "3.x"},
                    run_id="run_test",
                    started_at="2025-12-01T00:00:00+00:00",
                )

                self.assertEqual(run.run_id, "run_test")
                self.assertEqual(run.config_hash, "abc123")
                self.assertEqual(run.sampling_seed, 1337)

                fetched = store.get_run("run_test")
                self.assertIsNotNone(fetched)
                assert fetched is not None
                self.assertEqual(fetched.run_id, "run_test")

    def test_upserts_raw_posts_and_tracks_seen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "state.sqlite"
            with SQLiteStateStore.open(db_path) as store:
                store.upsert_raw_post(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    actor_source="apify/actor",
                    raw_item={"url": "https://example.com/p/1", "caption": "hi"},
                    fetched_at="2025-12-01T00:00:00+00:00",
                )
                store.upsert_raw_post(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    actor_source=None,
                    raw_item={"url": "https://example.com/p/1", "caption": "updated"},
                    fetched_at="2025-12-01T00:00:01+00:00",
                )

                self.assertEqual(store.raw_post_count(), 1)
                self.assertEqual(store.seen_post_keys(), {"id:1"})

    def test_latest_decision_and_eligible_view_use_latest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "state.sqlite"
            with SQLiteStateStore.open(db_path) as store:
                store.upsert_raw_post(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    actor_source="apify/actor",
                    raw_item={"url": "https://example.com/p/1", "caption": "x" * 80},
                    fetched_at="2025-12-01T00:00:00+00:00",
                )

                store.record_llm_decision(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    model="gpt-5-nano",
                    decision=_decision(eligible=False, confidence=0.6),
                    created_at="2025-12-01T00:00:10+00:00",
                )
                store.record_llm_decision(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    model="gpt-5-mini",
                    decision=_decision(eligible=True, confidence=0.95),
                    created_at="2025-12-01T00:00:11+00:00",
                )

                latest = store.latest_decision("id:1")
                self.assertIsNotNone(latest)
                assert latest is not None
                self.assertTrue(latest.eligible)

                eligible = store.eligible_posts()
                self.assertEqual(len(eligible), 1)
                self.assertEqual(eligible[0].post_key, "id:1")
                self.assertEqual(eligible[0].model, "gpt-5-mini")
                self.assertGreaterEqual(eligible[0].overall_confidence, 0.70)

    def test_record_decision_requires_raw_post_first(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "state.sqlite"
            with SQLiteStateStore.open(db_path) as store:
                with self.assertRaises(Exception):
                    store.record_llm_decision(
                        post_key="id:missing",
                        url="https://example.com/p/missing",
                        model="gpt-5-nano",
                        decision=_decision(eligible=True),
                    )


if __name__ == "__main__":
    unittest.main()

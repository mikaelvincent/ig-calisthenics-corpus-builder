from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from ig_corpus.config_schema import AppConfig, TargetsConfig
from ig_corpus.export_pdf import export_codebook_pdf
from ig_corpus.llm_schema import LLMDecision
from ig_corpus.storage import SQLiteStateStore


def _decision(*, eligible: bool) -> LLMDecision:
    payload: dict[str, Any] = {
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
            "narrative_labels": ["consistency"],
            "discourse_moves": ["advice"],
            "neoliberal_signals": [],
        },
        "overall_confidence": 0.9,
    }
    return LLMDecision.model_validate(payload)


class TestExportPDF(unittest.TestCase):
    def test_exports_non_empty_pdf(self) -> None:
        try:
            import reportlab  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise AssertionError("reportlab is required for this test") from e

        cfg = AppConfig(targets=TargetsConfig(final_n=1, pool_n=1, sampling_seed=123))

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "codebook.pdf"

            with SQLiteStateStore.open(":memory:") as store:
                store.create_run(
                    config_hash="hash1",
                    sampling_seed=cfg.targets.sampling_seed,
                    versions={"python": "3.x"},
                    run_id="run_test",
                    started_at="2025-12-01T00:00:00+00:00",
                )

                store.record_apify_actor_run(
                    run_id="run_test",
                    actor_id="apify/instagram-hashtag-scraper",
                    actor_run_id="run_1",
                    dataset_id="ds_1",
                    created_at="2025-12-01T00:00:00+00:00",
                )

                store.upsert_raw_post(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    actor_source="apify/instagram-hashtag-scraper",
                    raw_item={
                        "url": "https://example.com/p/1",
                        "caption": "x" * 80,
                        "hashtags": ["tag1", "tag2"],
                    },
                    fetched_at="2025-12-01T00:00:00+00:00",
                )
                store.record_llm_decision(
                    post_key="id:1",
                    url="https://example.com/p/1",
                    model="gpt-5-nano",
                    decision=_decision(eligible=True),
                    created_at="2025-12-01T00:00:01+00:00",
                )

                export_codebook_pdf(cfg, store, out_path, run_id="run_test")

            self.assertTrue(out_path.exists())
            data = out_path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            self.assertGreater(len(data), 500)


if __name__ == "__main__":
    unittest.main()

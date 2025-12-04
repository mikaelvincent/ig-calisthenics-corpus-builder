from __future__ import annotations

import unittest
from typing import Any

from ig_corpus.eligibility import enforce_structured_eligibility
from ig_corpus.llm_schema import LLMDecision


def _mk_decision(payload_overrides: dict[str, Any]) -> LLMDecision:
    base: dict[str, Any] = {
        "eligible": True,
        "eligibility_reasons": ["model_claim"],
        "language": {"is_english": True, "confidence": 0.9},
        "topic": {"is_bodyweight_calisthenics": True, "confidence": 0.9, "topic_notes": "ok"},
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

    merged = dict(base)
    merged.update(payload_overrides)
    return LLMDecision.model_validate(merged)


class TestEligibilityEnforcement(unittest.TestCase):
    def test_overrides_when_commercial_conflicts(self) -> None:
        decision = _mk_decision(
            {
                "eligible": True,
                "commercial": {"is_exclusively_commercial": True, "signals": ["code"]},
            }
        )
        enforced = enforce_structured_eligibility(decision)

        self.assertFalse(enforced.eligible)
        self.assertIn("eligibility_overridden_reject", enforced.eligibility_reasons)
        self.assertIn("eligibility_rule:exclusively_commercial", enforced.eligibility_reasons)

    def test_overrides_when_model_marks_reject_but_fields_pass(self) -> None:
        decision = _mk_decision(
            {
                "eligible": False,
                "eligibility_reasons": ["model_reject"],
            }
        )
        enforced = enforce_structured_eligibility(decision)

        self.assertTrue(enforced.eligible)
        self.assertIn("eligibility_overridden_accept", enforced.eligibility_reasons)
        self.assertIn("model_reject", enforced.eligibility_reasons)


if __name__ == "__main__":
    unittest.main()

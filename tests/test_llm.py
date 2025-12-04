from __future__ import annotations

import unittest
from typing import Any

from ig_corpus.config_schema import OpenAIConfig
from ig_corpus.llm import OpenAIPostClassifier, PostForLLM
from ig_corpus.llm_schema import DECISION_JSON_SCHEMA, DECISION_SCHEMA_NAME


class _FakeOutputTextPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeOutputMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeOutputTextPart(text)]


class _FakeResponse:
    def __init__(self, *, output_text: str, output: list[Any] | None = None) -> None:
        self.output_text = output_text
        self.output = output or []


class _FakeResponses:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Fake client received more calls than expected")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, response: _FakeResponse | list[_FakeResponse]) -> None:
        responses = response if isinstance(response, list) else [response]
        self.responses = _FakeResponses(responses)


_DECISION_JSON = """\
{
  "eligible": true,
  "eligibility_reasons": ["English", "Bodyweight training"],
  "language": { "is_english": true, "confidence": 0.95 },
  "topic": {
    "is_bodyweight_calisthenics": true,
    "confidence": 0.9,
    "topic_notes": "Mentions pull-ups and push-ups."
  },
  "commercial": { "is_exclusively_commercial": false, "signals": [] },
  "caption_quality": { "is_analyzable": true, "issues": [] },
  "tags": {
    "genre": "training_log",
    "narrative_labels": ["consistency"],
    "discourse_moves": ["advice"],
    "neoliberal_signals": []
  },
  "overall_confidence": 0.9
}
"""

_DECISION_JSON_LOW_CONF = """\
{
  "eligible": true,
  "eligibility_reasons": ["Unclear caption intent"],
  "language": { "is_english": true, "confidence": 0.7 },
  "topic": {
    "is_bodyweight_calisthenics": true,
    "confidence": 0.6,
    "topic_notes": "Possible bodyweight training, but ambiguous."
  },
  "commercial": { "is_exclusively_commercial": false, "signals": [] },
  "caption_quality": { "is_analyzable": true, "issues": ["fragmentary_caption"] },
  "tags": {
    "genre": "other",
    "narrative_labels": [],
    "discourse_moves": [],
    "neoliberal_signals": []
  },
  "overall_confidence": 0.4
}
"""


class TestOpenAIPostClassifier(unittest.TestCase):
    def test_classify_sends_json_schema_format(self) -> None:
        cfg = OpenAIConfig(max_output_tokens=123)
        fake = _FakeClient(_FakeResponse(output_text=_DECISION_JSON))
        classifier = OpenAIPostClassifier("sk-test", openai_cfg=cfg, client=fake)  # type: ignore[arg-type]

        decision = classifier.classify(PostForLLM(url="https://example.com/p/abc", caption="Hello"))

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.tags.genre, "training_log")

        call = fake.responses.calls[0]
        self.assertEqual(call["model"], cfg.model_primary)
        self.assertEqual(call["max_output_tokens"], 123)

        text_cfg = call["text"]["format"]
        self.assertEqual(text_cfg["type"], "json_schema")
        self.assertEqual(text_cfg["name"], DECISION_SCHEMA_NAME)
        self.assertTrue(text_cfg["strict"])
        self.assertEqual(text_cfg["schema"], DECISION_JSON_SCHEMA)

    def test_classify_falls_back_to_output_items(self) -> None:
        cfg = OpenAIConfig(max_output_tokens=50)
        fake_response = _FakeResponse(
            output_text="",
            output=[_FakeOutputMessage(_DECISION_JSON)],
        )
        fake = _FakeClient(fake_response)
        classifier = OpenAIPostClassifier("sk-test", openai_cfg=cfg, client=fake)  # type: ignore[arg-type]

        decision = classifier.classify(PostForLLM(url="https://example.com/p/xyz"))

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.language.is_english, True)

    def test_classify_escalates_on_low_confidence(self) -> None:
        cfg = OpenAIConfig(
            model_primary="gpt-5-nano",
            model_escalation="gpt-5-mini",
            escalation_confidence_threshold=0.70,
            max_output_tokens=80,
        )
        fake = _FakeClient(
            [
                _FakeResponse(output_text=_DECISION_JSON_LOW_CONF),
                _FakeResponse(output_text=_DECISION_JSON),
            ]
        )
        classifier = OpenAIPostClassifier("sk-test", openai_cfg=cfg, client=fake)  # type: ignore[arg-type]

        decision = classifier.classify(PostForLLM(url="https://example.com/p/lowconf", caption="Hi"))

        self.assertTrue(decision.eligible)
        self.assertGreaterEqual(decision.overall_confidence, 0.70)
        self.assertEqual(len(fake.responses.calls), 2)
        self.assertEqual(fake.responses.calls[0]["model"], "gpt-5-nano")
        self.assertEqual(fake.responses.calls[1]["model"], "gpt-5-mini")

    def test_classify_escalates_on_parse_failure(self) -> None:
        cfg = OpenAIConfig(
            model_primary="gpt-5-nano",
            model_escalation="gpt-5-mini",
            escalation_confidence_threshold=0.70,
            max_output_tokens=80,
        )
        fake = _FakeClient(
            [
                _FakeResponse(output_text="{not-json"),
                _FakeResponse(output_text=_DECISION_JSON),
            ]
        )
        classifier = OpenAIPostClassifier("sk-test", openai_cfg=cfg, client=fake)  # type: ignore[arg-type]

        decision = classifier.classify(PostForLLM(url="https://example.com/p/badjson", caption="Hi"))

        self.assertTrue(decision.eligible)
        self.assertEqual(len(fake.responses.calls), 2)
        self.assertEqual(fake.responses.calls[0]["model"], "gpt-5-nano")
        self.assertEqual(fake.responses.calls[1]["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DECISION_SCHEMA_NAME = "ig_corpus_post_decision"

GENRE = Literal[
    "training_log",
    "tutorial_coaching",
    "motivation_mindset",
    "personal_story_reflection",
    "identity_community",
    "transformation_progress",
    "injury_rehab",
    "humor_meme",
    "educational_sciency",
    "other",
]

# NOTE: This schema is intentionally hand-authored to stay within the subset of JSON
# Schema required by Structured Outputs and to keep it stable across Pydantic changes.
DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "eligible": {"type": "boolean"},
        "eligibility_reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
        "language": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_english": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
            "required": ["is_english", "confidence"],
        },
        "topic": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_bodyweight_calisthenics": {"type": "boolean"},
                "confidence": {"type": "number"},
                "topic_notes": {"type": "string"},
            },
            "required": ["is_bodyweight_calisthenics", "confidence", "topic_notes"],
        },
        "commercial": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_exclusively_commercial": {"type": "boolean"},
                "signals": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["is_exclusively_commercial", "signals"],
        },
        "caption_quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_analyzable": {"type": "boolean"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["is_analyzable", "issues"],
        },
        "tags": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "genre": {
                    "type": "string",
                    "enum": [
                        "training_log",
                        "tutorial_coaching",
                        "motivation_mindset",
                        "personal_story_reflection",
                        "identity_community",
                        "transformation_progress",
                        "injury_rehab",
                        "humor_meme",
                        "educational_sciency",
                        "other",
                    ],
                },
                "narrative_labels": {"type": "array", "items": {"type": "string"}},
                "discourse_moves": {"type": "array", "items": {"type": "string"}},
                "neoliberal_signals": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["genre", "narrative_labels", "discourse_moves", "neoliberal_signals"],
        },
        "overall_confidence": {"type": "number"},
    },
    "required": [
        "eligible",
        "eligibility_reasons",
        "language",
        "topic",
        "commercial",
        "caption_quality",
        "tags",
        "overall_confidence",
    ],
}


class LanguageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_english: bool
    confidence: float = Field(ge=0.0, le=1.0)


class TopicResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_bodyweight_calisthenics: bool
    confidence: float = Field(ge=0.0, le=1.0)
    topic_notes: str


class CommercialResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_exclusively_commercial: bool
    signals: list[str]


class CaptionQualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_analyzable: bool
    issues: list[str]


class TagResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    genre: GENRE
    narrative_labels: list[str]
    discourse_moves: list[str]
    neoliberal_signals: list[str]


class LLMDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool
    eligibility_reasons: list[str]
    language: LanguageResult
    topic: TopicResult
    commercial: CommercialResult
    caption_quality: CaptionQualityResult
    tags: TagResult
    overall_confidence: float = Field(ge=0.0, le=1.0)

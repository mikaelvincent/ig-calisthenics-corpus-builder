from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .apify_client import ActorRunRef
from .llm import PostForLLM
from .llm_schema import LLMDecision


_OFFLINE_CAPTION_1 = (
    "Pull-up progression day: strict reps, scapular control, and hollow holds. "
    "Bodyweight training is simple, not easy. Logging what worked and what didn't."
)
_OFFLINE_CAPTION_2 = (
    "Quick session: dips + push-ups ladder, then core. Calisthenics basics done with intent."
    " Notes in caption for future me."
)
_OFFLINE_CAPTION_3 = (
    "Skill work: handstand line drills and freestanding attempts. "
    "Focusing on stacking and breathing under tension."
)

_DEFAULT_OFFLINE_ITEMS: list[dict[str, Any]] = [
    {
        "url": "https://example.com/p/1",
        "caption": _OFFLINE_CAPTION_1,
        "hashtags": ["calisthenics", "streetworkout", "bodyweighttraining"],
        "mentions": ["coach"],
        "type": "Image",
        "productType": "feed",
        "isSponsored": False,
        "timestamp": "2025-01-01T00:00:00Z",
    },
    {
        "url": "https://example.com/p/2",
        "caption": _OFFLINE_CAPTION_2,
        "hashtags": ["calisthenics", "dips", "pushups"],
        "mentions": [],
        "type": "Image",
        "productType": "feed",
        "isSponsored": False,
        "timestamp": "2025-01-02T00:00:00Z",
    },
    {
        "url": "https://example.com/p/3",
        "caption": _OFFLINE_CAPTION_3,
        "hashtags": ["calisthenics", "handstand", "bodyweightworkout"],
        "mentions": [],
        "type": "Video",
        "productType": "feed",
        "isSponsored": False,
        "timestamp": "2025-01-03T00:00:00Z",
    },
    {
        "url": "https://example.com/p/4",
        "caption": _OFFLINE_CAPTION_1,
        "hashtags": ["calisthenics", "pullups"],
        "mentions": [],
        "type": "Image",
        "productType": "feed",
        "isSponsored": False,
        "timestamp": "2025-01-04T00:00:00Z",
    },
    {
        "url": "https://example.com/p/5",
        "caption": _OFFLINE_CAPTION_2,
        "hashtags": ["streetworkout", "bodyweighttraining"],
        "mentions": [],
        "type": "Image",
        "productType": "feed",
        "isSponsored": False,
        "timestamp": "2025-01-05T00:00:00Z",
    },
]


def _numeric_suffix(value: str) -> int | None:
    s = (value or "").strip().rstrip("/")
    if not s:
        return None

    digits: list[str] = []
    for ch in reversed(s):
        if ch.isdigit():
            digits.append(ch)
            continue
        break

    if not digits:
        return None

    try:
        return int("".join(reversed(digits)))
    except Exception:
        return None


def _offline_decision(*, eligible: bool) -> LLMDecision:
    payload: dict[str, Any] = {
        "eligible": bool(eligible),
        "eligibility_reasons": ["offline_accept"] if eligible else ["offline_reject"],
        "language": {"is_english": True, "confidence": 0.99},
        "topic": {
            "is_bodyweight_calisthenics": bool(eligible),
            "confidence": 0.95 if eligible else 0.6,
            "topic_notes": "offline_stub",
        },
        "commercial": {"is_exclusively_commercial": False, "signals": []},
        "caption_quality": {"is_analyzable": True, "issues": []},
        "tags": {
            "genre": "training_log" if eligible else "other",
            "narrative_labels": ["offline"],
            "discourse_moves": ["log"],
            "neoliberal_signals": [],
        },
        "overall_confidence": 0.95 if eligible else 0.6,
    }
    return LLMDecision.model_validate(payload)


@dataclass
class OfflineInstagramHashtagScraper:
    """
    Network-free stub for dry-run smoke checks.

    Provides a small, deterministic set of dataset-like items with fields that
    normalize cleanly for the pipeline.
    """

    items: Sequence[dict[str, Any]] = tuple(_DEFAULT_OFFLINE_ITEMS)

    def run_and_fetch(
        self,
        terms: Sequence[str],
        *,
        apify: Any,
        timeout_secs: int | None = None,
        dataset_limit: int | None = None,
        clean: bool = True,
    ) -> tuple[ActorRunRef, list[dict[str, Any]]]:
        _ = (terms, timeout_secs, clean)
        actor_id = str(getattr(apify, "primary_actor", "offline"))
        limit = len(self.items) if dataset_limit is None else max(0, int(dataset_limit))
        run_ref = ActorRunRef(
            actor_id=actor_id,
            run_id="offline_run",
            default_dataset_id="offline_dataset",
        )
        return run_ref, list(self.items)[:limit]


class OfflinePostClassifier:
    """
    Deterministic, schema-valid classifier stub for offline `dry-run`.

    Marks urls ending with an odd numeric suffix as eligible, and even as rejected.
    """

    def classify(self, post: PostForLLM) -> LLMDecision:
        suffix = _numeric_suffix(post.url)
        if suffix is None:
            return _offline_decision(eligible=True)
        return _offline_decision(eligible=(suffix % 2 == 1))

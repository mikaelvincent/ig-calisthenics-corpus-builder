from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config_schema import FiltersConfig
from .post import NormalizedPost


@dataclass(frozen=True)
class PrecheckResult:
    passed: bool
    reasons: Sequence[str]


def run_prechecks(post: NormalizedPost, *, filters: FiltersConfig) -> PrecheckResult:
    """
    Fast, deterministic checks to reduce LLM calls.

    These checks are meant to be conservative: reject only when the post clearly cannot
    be used, regardless of semantics.
    """
    reasons: list[str] = []

    caption = (post.caption or "").strip()
    if not caption:
        reasons.append("missing_caption")
    elif filters.min_caption_chars > 0 and len(caption) < filters.min_caption_chars:
        reasons.append("caption_too_short")

    if not filters.allow_reels:
        product = (post.product_type or "").strip().casefold()
        if product:
            if product != "feed":
                reasons.append("reels_not_allowed")
        else:
            post_type = (post.type or "").strip().casefold()
            if post_type == "video":
                reasons.append("reels_not_allowed")

    if filters.reject_if_sponsored_true and post.is_sponsored is True:
        reasons.append("sponsored_rejected")

    return PrecheckResult(passed=not reasons, reasons=reasons)

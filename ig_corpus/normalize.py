from __future__ import annotations

from typing import Any, Mapping

from .llm import PostForLLM


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_str_list(value: Any, *, strip_prefix: str | None = None) -> list[str] | None:
    if value is None:
        return None

    out: list[str] = []

    def _norm(item: str) -> str | None:
        s = (item or "").strip()
        if not s:
            return None
        if strip_prefix and s.startswith(strip_prefix):
            s = s[len(strip_prefix) :].strip()
        return s or None

    if isinstance(value, str):
        normed = _norm(value)
        return [normed] if normed else []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                normed = _norm(item)
                if normed:
                    out.append(normed)
        return out

    return None


def post_for_llm_from_apify_item(item: Mapping[str, Any]) -> PostForLLM | None:
    """
    Best-effort extraction of LLM-relevant fields from Apify Instagram dataset items.

    This is intentionally permissive: fields are optional except for a usable URL.
    """
    url = (
        _coerce_str(item.get("url"))
        or _coerce_str(item.get("postUrl"))
        or _coerce_str(item.get("post_url"))
        or _coerce_str(item.get("postURL"))
    )
    if not url:
        return None

    caption = (
        _coerce_str(item.get("caption"))
        or _coerce_str(item.get("captionText"))
        or _coerce_str(item.get("text"))
        or _coerce_str(item.get("caption_text"))
    )

    hashtags = _coerce_str_list(item.get("hashtags"), strip_prefix="#")
    if hashtags is None:
        hashtags = _coerce_str_list(item.get("hashTags"), strip_prefix="#")

    mentions = _coerce_str_list(item.get("mentions"), strip_prefix="@")
    if mentions is None:
        mentions = _coerce_str_list(item.get("userMentions"), strip_prefix="@")

    alt = (
        _coerce_str(item.get("alt"))
        or _coerce_str(item.get("accessibility_caption"))
        or _coerce_str(item.get("accessibilityCaption"))
        or _coerce_str(item.get("accessibilityCaptionText"))
    )

    post_type = _coerce_str(item.get("type")) or _coerce_str(item.get("postType"))
    product_type = _coerce_str(item.get("productType")) or _coerce_str(item.get("product_type"))

    is_sponsored = _coerce_bool(item.get("isSponsored"))
    if is_sponsored is None:
        is_sponsored = _coerce_bool(item.get("sponsored"))

    timestamp = (
        _coerce_str(item.get("timestamp"))
        or _coerce_str(item.get("takenAt"))
        or _coerce_str(item.get("taken_at"))
        or _coerce_str(item.get("date"))
    )

    return PostForLLM(
        url=url,
        caption=caption,
        hashtags=hashtags,
        mentions=mentions,
        alt=alt,
        type=post_type,
        product_type=product_type,
        is_sponsored=is_sponsored,
        timestamp=timestamp,
    )

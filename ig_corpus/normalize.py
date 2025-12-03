from __future__ import annotations

from typing import Any, Mapping

from .llm import PostForLLM
from .post import NormalizedPost


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return None


def _coerce_id(value: Any) -> str | None:
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    if isinstance(value, int):
        return str(value)
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


def _dedupe_terms(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        term = (item or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return tuple(out)


def normalized_post_from_apify_item(item: Mapping[str, Any]) -> NormalizedPost | None:
    """
    Best-effort extraction of a normalized post record from Apify dataset items.

    This supports the primary Hashtag Scraper output and tolerates minor field variations.
    """
    url = (
        _coerce_str(item.get("url"))
        or _coerce_str(item.get("postUrl"))
        or _coerce_str(item.get("post_url"))
        or _coerce_str(item.get("postURL"))
    )
    if not url:
        return None

    post_id = (
        _coerce_id(item.get("id"))
        or _coerce_id(item.get("postId"))
        or _coerce_id(item.get("post_id"))
    )
    short_code = (
        _coerce_str(item.get("shortCode"))
        or _coerce_str(item.get("shortcode"))
        or _coerce_str(item.get("short_code"))
    )

    caption = (
        _coerce_str(item.get("caption"))
        or _coerce_str(item.get("captionText"))
        or _coerce_str(item.get("text"))
        or _coerce_str(item.get("caption_text"))
    )

    hashtags = _coerce_str_list(item.get("hashtags"), strip_prefix="#")
    if hashtags is None:
        hashtags = _coerce_str_list(item.get("hashTags"), strip_prefix="#")
    hashs = _dedupe_terms(hashtags or [])

    mentions = _coerce_str_list(item.get("mentions"), strip_prefix="@")
    if mentions is None:
        mentions = _coerce_str_list(item.get("userMentions"), strip_prefix="@")
    ments = _dedupe_terms(mentions or [])

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

    owner_username = (
        _coerce_str(item.get("ownerUsername"))
        or _coerce_str(item.get("owner_username"))
        or _coerce_str(item.get("username"))
    )
    owner_id = _coerce_id(item.get("ownerId")) or _coerce_id(item.get("owner_id"))

    owner_obj = item.get("owner")
    if (owner_username is None or owner_id is None) and isinstance(owner_obj, Mapping):
        owner_username = owner_username or _coerce_str(owner_obj.get("username"))
        owner_id = owner_id or _coerce_id(owner_obj.get("id"))

    return NormalizedPost(
        url=url,
        post_id=post_id,
        short_code=short_code,
        owner_username=owner_username,
        owner_id=owner_id,
        caption=caption,
        hashtags=hashs,
        mentions=ments,
        alt=alt,
        type=post_type,
        product_type=product_type,
        is_sponsored=is_sponsored,
        timestamp=timestamp,
    )


def post_for_llm(post: NormalizedPost) -> PostForLLM:
    return PostForLLM(
        url=post.url,
        caption=post.caption,
        hashtags=post.hashtags,
        mentions=post.mentions,
        alt=post.alt,
        type=post.type,
        product_type=post.product_type,
        is_sponsored=post.is_sponsored,
        timestamp=post.timestamp,
    )


def post_for_llm_from_apify_item(item: Mapping[str, Any]) -> PostForLLM | None:
    post = normalized_post_from_apify_item(item)
    if post is None:
        return None
    return post_for_llm(post)

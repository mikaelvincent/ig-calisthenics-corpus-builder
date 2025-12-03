from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from .post import NormalizedPost


def canonicalize_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""

    try:
        parts = urlsplit(value)
    except Exception:
        return value.rstrip("/")

    if not parts.scheme or not parts.netloc:
        return value.rstrip("/")

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = (parts.path or "").rstrip("/")
    if not path:
        path = "/"

    return urlunsplit((scheme, netloc, path, "", ""))


def dedupe_key(post: NormalizedPost) -> str:
    if post.post_id:
        return f"id:{post.post_id}"
    if post.short_code:
        return f"shortcode:{post.short_code}"
    return f"url:{canonicalize_url(post.url)}"


@dataclass
class SeenKeys:
    keys: set[str] = field(default_factory=set)

    def has(self, key: str) -> bool:
        return key in self.keys

    def add(self, key: str) -> None:
        self.keys.add(key)

    def add_post(self, post: NormalizedPost) -> str:
        key = dedupe_key(post)
        self.add(key)
        return key

    def has_post(self, post: NormalizedPost) -> bool:
        return self.has(dedupe_key(post))

    def update(self, posts: Iterable[NormalizedPost]) -> None:
        for post in posts:
            self.add_post(post)

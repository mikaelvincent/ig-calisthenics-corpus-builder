# tests/test_dedupe.py
from __future__ import annotations

import unittest

from ig_corpus.dedupe import canonicalize_url, dedupe_key
from ig_corpus.post import NormalizedPost


class TestDedupe(unittest.TestCase):
    def test_canonicalize_strips_query_and_frag(self) -> None:
        url = "https://www.instagram.com/p/AbC/?utm_source=x#frag"
        self.assertEqual(canonicalize_url(url), "https://instagram.com/p/AbC")

    def test_dedupe_key_prefers_id(self) -> None:
        post = NormalizedPost(
            url="https://instagram.com/p/x/",
            post_id="123",
            short_code="abc",
        )
        self.assertEqual(dedupe_key(post), "id:123")

    def test_dedupe_key_uses_shortcode(self) -> None:
        post = NormalizedPost(
            url="https://instagram.com/p/x/",
            post_id=None,
            short_code="abc",
        )
        self.assertEqual(dedupe_key(post), "shortcode:abc")


if __name__ == "__main__":
    unittest.main()

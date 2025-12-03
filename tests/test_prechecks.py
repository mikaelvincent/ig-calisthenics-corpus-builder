# tests/test_prechecks.py
from __future__ import annotations

import unittest

from ig_corpus.config_schema import FiltersConfig
from ig_corpus.post import NormalizedPost
from ig_corpus.prechecks import run_prechecks


class TestPrechecks(unittest.TestCase):
    def test_rejects_missing_caption(self) -> None:
        post = NormalizedPost(url="https://example.com/p/1", caption=None)
        res = run_prechecks(post, filters=FiltersConfig())
        self.assertFalse(res.passed)
        self.assertIn("missing_caption", res.reasons)

    def test_rejects_short_caption(self) -> None:
        post = NormalizedPost(url="https://example.com/p/1", caption="x" * 10)
        res = run_prechecks(post, filters=FiltersConfig(min_caption_chars=40))
        self.assertFalse(res.passed)
        self.assertIn("caption_too_short", res.reasons)

    def test_rejects_reels_when_disabled(self) -> None:
        post = NormalizedPost(
            url="https://example.com/p/1",
            caption="x" * 60,
            product_type="clips",
        )
        res = run_prechecks(post, filters=FiltersConfig(allow_reels=False))
        self.assertFalse(res.passed)
        self.assertIn("reels_not_allowed", res.reasons)

    def test_rejects_sponsored_when_configured(self) -> None:
        post = NormalizedPost(
            url="https://example.com/p/1",
            caption="x" * 60,
            is_sponsored=True,
        )
        res = run_prechecks(post, filters=FiltersConfig(reject_if_sponsored_true=True))
        self.assertFalse(res.passed)
        self.assertIn("sponsored_rejected", res.reasons)


if __name__ == "__main__":
    unittest.main()

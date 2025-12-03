# tests/test_normalize.py
from __future__ import annotations

import unittest

from ig_corpus.normalize import normalized_post_from_apify_item


class TestNormalize(unittest.TestCase):
    def test_extracts_common_actor_fields(self) -> None:
        item = {
            "id": "1",
            "shortCode": "AbC",
            "url": "https://www.instagram.com/p/AbC/",
            "ownerUsername": "user1",
            "ownerId": "42",
            "caption": "Hello world",
            "hashtags": ["#One", "two", "TWO"],
            "mentions": ["@a", "b", "B"],
            "type": "Image",
            "productType": "feed",
            "isSponsored": False,
            "timestamp": "2025-11-07T20:56:47.000Z",
        }

        post = normalized_post_from_apify_item(item)
        self.assertIsNotNone(post)
        assert post is not None

        self.assertEqual(post.post_id, "1")
        self.assertEqual(post.short_code, "AbC")
        self.assertEqual(post.url, "https://www.instagram.com/p/AbC/")
        self.assertEqual(post.owner_username, "user1")
        self.assertEqual(post.owner_id, "42")
        self.assertEqual(post.hashtags, ("One", "two"))
        self.assertEqual(post.mentions, ("a", "b"))
        self.assertEqual(post.product_type, "feed")
        self.assertEqual(post.is_sponsored, False)

    def test_returns_none_without_url(self) -> None:
        post = normalized_post_from_apify_item({"caption": "hi"})
        self.assertIsNone(post)


if __name__ == "__main__":
    unittest.main()

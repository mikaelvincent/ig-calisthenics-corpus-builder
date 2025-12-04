from __future__ import annotations

import unittest

from ig_corpus.query_queue import TermQueue, normalize_term


class TestQueryQueue(unittest.TestCase):
    def test_normalize_term_strips_hash_and_space(self) -> None:
        self.assertEqual(normalize_term(" #Calisthenics "), "Calisthenics")
        self.assertEqual(normalize_term("  "), "")
        self.assertEqual(normalize_term("#"), "")

    def test_queue_dedupes_only_when_present(self) -> None:
        q = TermQueue()
        self.assertTrue(q.add("#A"))
        self.assertFalse(q.add("a"))
        batch = q.pop_batch(1)
        self.assertEqual(batch, ["A"])
        self.assertTrue(q.add("a"))

    def test_pop_batch_removes_presence(self) -> None:
        q = TermQueue(["a", "b", "c"])
        self.assertEqual(len(q), 3)
        popped = q.pop_batch(2)
        self.assertEqual(popped, ["a", "b"])
        self.assertEqual(len(q), 1)
        self.assertTrue(q.add("a"))


if __name__ == "__main__":
    unittest.main()

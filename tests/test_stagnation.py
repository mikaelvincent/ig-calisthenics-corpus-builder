from __future__ import annotations

import unittest

from ig_corpus.stagnation import StagnationTracker


class TestStagnationTracker(unittest.TestCase):
    def test_does_not_trigger_until_window_full(self) -> None:
        s = StagnationTracker(window_size=3, min_new_total=5)
        self.assertFalse(s.push(0))
        self.assertFalse(s.push(0))
        self.assertFalse(s.push(0))
        self.assertTrue(s.total() < 5)

    def test_triggers_when_sum_below_threshold(self) -> None:
        s = StagnationTracker(window_size=2, min_new_total=1)
        self.assertFalse(s.push(0))
        self.assertTrue(s.push(0))

    def test_does_not_trigger_when_threshold_met(self) -> None:
        s = StagnationTracker(window_size=2, min_new_total=2)
        self.assertFalse(s.push(1))
        self.assertFalse(s.push(1))


if __name__ == "__main__":
    unittest.main()

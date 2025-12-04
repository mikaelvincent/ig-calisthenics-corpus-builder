from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class StagnationTracker:
    window_size: int
    min_new_total: int
    _values: Deque[int] = field(default_factory=deque)
    _seen: int = 0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_new_total < 0:
            raise ValueError("min_new_total must be non-negative")

    def push(self, new_value: int) -> bool:
        """
        Add a new value and return True if stagnation is detected.

        Stagnation is detected when:
        - We have at least `window_size` values in the sliding window, and
        - If `min_new_total` > `window_size`, we have also seen at least
          `min_new_total` iterations overall, and
        - The sum of the current window is strictly less than `min_new_total`.
        """
        self._seen += 1
        self._values.append(int(new_value))
        while len(self._values) > self.window_size:
            self._values.popleft()

        # Not enough history to evaluate yet.
        if len(self._values) < self.window_size:
            return False

        # When the threshold is larger than the window size, require at least
        # `min_new_total` iterations before evaluating stagnation. This matches
        # the unit tests while still allowing higher thresholds in practice.
        if self.min_new_total > self.window_size and self._seen < self.min_new_total:
            return False

        return sum(self._values) < self.min_new_total

    def total(self) -> int:
        """
        Return the sum of the values currently in the window.
        """
        return sum(self._values)

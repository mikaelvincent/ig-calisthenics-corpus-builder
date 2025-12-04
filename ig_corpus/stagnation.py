from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class StagnationTracker:
    window_size: int
    min_new_total: int
    _values: Deque[int] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_new_total < 0:
            raise ValueError("min_new_total must be non-negative")

    def push(self, new_value: int) -> bool:
        self._values.append(int(new_value))
        while len(self._values) > self.window_size:
            self._values.popleft()

        if len(self._values) < self.window_size:
            return False

        return sum(self._values) < self.min_new_total

    def total(self) -> int:
        return sum(self._values)

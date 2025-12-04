from __future__ import annotations

from collections import deque
from typing import Iterable


def normalize_term(value: str) -> str:
    term = (value or "").strip()
    if term.startswith("#"):
        term = term[1:].strip()
    return term


class TermQueue:
    """
    Simple FIFO queue for query terms, with per-queue deduping.

    Deduping is scoped to what is currently queued; a term can be re-added after it has been popped.
    """

    def __init__(self, initial: Iterable[str] | None = None) -> None:
        self._queue: deque[str] = deque()
        self._present: set[str] = set()

        if initial is not None:
            self.add_many(initial)

    def __len__(self) -> int:
        return len(self._queue)

    def present_keys(self) -> set[str]:
        return set(self._present)

    def add(self, term: str) -> bool:
        norm = normalize_term(term)
        if not norm:
            return False

        key = norm.casefold()
        if key in self._present:
            return False

        self._present.add(key)
        self._queue.append(norm)
        return True

    def add_many(self, terms: Iterable[str]) -> int:
        added = 0
        for t in terms:
            if self.add(t):
                added += 1
        return added

    def pop_batch(self, size: int) -> list[str]:
        if size <= 0:
            raise ValueError("size must be positive")

        out: list[str] = []
        for _ in range(min(size, len(self._queue))):
            term = self._queue.popleft()
            self._present.discard(term.casefold())
            out.append(term)

        return out

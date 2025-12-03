from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class NormalizedPost:
    """A stable, minimal post record used for filtering and exports."""

    url: str
    post_id: str | None = None
    short_code: str | None = None

    owner_username: str | None = None
    owner_id: str | None = None

    caption: str | None = None
    hashtags: Sequence[str] = ()
    mentions: Sequence[str] = ()

    alt: str | None = None
    type: str | None = None
    product_type: str | None = None
    is_sponsored: bool | None = None
    timestamp: str | None = None

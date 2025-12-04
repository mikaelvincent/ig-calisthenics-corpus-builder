from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """
    Exponential backoff retry policy.

    - max_attempts counts the initial attempt (max_attempts=3 => 1 try + 2 retries).
    - base_delay_seconds is the first delay after the first failure.
    - jitter_ratio adds multiplicative jitter in [1-jitter, 1+jitter].
    - retry_after_cap_seconds caps any Retry-After override (0 disables the cap).
    """

    max_attempts: int = 6
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 20.0
    jitter_ratio: float = 0.25
    retry_after_cap_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be >= 0")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be >= 0")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not (0.0 <= self.jitter_ratio <= 1.0):
            raise ValueError("jitter_ratio must be between 0 and 1")
        if self.retry_after_cap_seconds < 0:
            raise ValueError("retry_after_cap_seconds must be >= 0")


@dataclass(frozen=True)
class RetryEvent:
    operation: str
    failure_attempt: int
    next_attempt: int
    max_attempts: int

    delay_seconds: float
    retry_after_seconds: float | None
    reason: str | None

    error_type: str
    error_message: str

    context_url: str | None


IsRetryableFn = Callable[[BaseException], tuple[bool, float | None, str | None]]
OnRetryFn = Callable[[RetryEvent], None]
SleepFn = Callable[[float], None]


def _compute_backoff_seconds(failure_attempt: int, cfg: RetryConfig) -> float:
    # failure_attempt=1 => base delay.
    exponent = max(0, int(failure_attempt) - 1)
    delay = cfg.base_delay_seconds * (2**exponent)
    return min(cfg.max_delay_seconds, max(0.0, float(delay)))


def _apply_jitter(delay: float, cfg: RetryConfig) -> float:
    d = max(0.0, float(delay))
    if d == 0.0 or cfg.jitter_ratio <= 0:
        return d
    factor = random.uniform(1.0 - cfg.jitter_ratio, 1.0 + cfg.jitter_ratio)
    return max(0.0, d * factor)


def _normalize_retry_after(value: float | None, cfg: RetryConfig) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except Exception:
        return None
    if seconds < 0:
        return None
    if cfg.retry_after_cap_seconds > 0:
        seconds = min(seconds, float(cfg.retry_after_cap_seconds))
    return seconds


def call_with_retries(
    fn: Callable[[], T],
    *,
    cfg: RetryConfig,
    is_retryable: IsRetryableFn,
    operation: str,
    on_retry: OnRetryFn | None = None,
    sleep_fn: SleepFn | None = None,
    context_url: str | None = None,
) -> T:
    """
    Call fn() with retries on retryable failures.

    Retry logic:
    - If is_retryable(exc) => (True, retry_after_seconds, reason), retry until max_attempts.
    - Delay uses exponential backoff with optional Retry-After override and jitter.
    """
    op = (operation or "").strip() or "operation"
    sleeper = sleep_fn or time.sleep

    for attempt in range(1, int(cfg.max_attempts) + 1):
        try:
            return fn()
        except Exception as exc:
            retryable, retry_after, reason = is_retryable(exc)

            if not retryable or attempt >= int(cfg.max_attempts):
                raise

            ra = _normalize_retry_after(retry_after, cfg)
            delay = _compute_backoff_seconds(attempt, cfg)
            if ra is not None:
                delay = max(delay, ra)
            delay = _apply_jitter(delay, cfg)

            if on_retry is not None:
                msg = (str(exc) or "").strip()
                on_retry(
                    RetryEvent(
                        operation=op,
                        failure_attempt=int(attempt),
                        next_attempt=int(attempt) + 1,
                        max_attempts=int(cfg.max_attempts),
                        delay_seconds=float(delay),
                        retry_after_seconds=ra,
                        reason=reason,
                        error_type=type(exc).__name__,
                        error_message=msg,
                        context_url=context_url,
                    )
                )

            if delay > 0:
                sleeper(float(delay))

    # Unreachable, but keeps typing happy.
    raise RuntimeError(f"Retry loop exited unexpectedly for operation={op}")

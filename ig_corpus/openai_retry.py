from __future__ import annotations

from typing import Any, Mapping


def _extract_headers(obj: Any) -> Mapping[str, Any]:
    headers = getattr(obj, "headers", None)
    if headers is None:
        return {}

    if isinstance(headers, Mapping):
        return headers

    # httpx.Headers is iterable over items; try coercion.
    try:
        return dict(headers)
    except Exception:
        return {}


def _parse_retry_after(headers: Mapping[str, Any]) -> float | None:
    if not headers:
        return None

    val: Any = None
    for key in ("retry-after", "Retry-After", "RETRY-AFTER"):
        try:
            val = headers.get(key)  # type: ignore[call-arg]
        except Exception:
            val = None
        if val is not None:
            break

    if val is None:
        return None

    try:
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        return float(str(val).strip())
    except Exception:
        return None


def _extract_retry_after_seconds(exc: BaseException) -> float | None:
    direct = getattr(exc, "retry_after", None)
    if direct is None:
        direct = getattr(exc, "retry_after_seconds", None)

    if direct is not None:
        try:
            return float(direct)
        except Exception:
            pass

    response = getattr(exc, "response", None)
    headers = _extract_headers(response)
    return _parse_retry_after(headers)


def _extract_status_code(exc: BaseException) -> int | None:
    val = getattr(exc, "status_code", None)
    if val is None:
        val = getattr(exc, "statusCode", None)
    if val is None:
        val = getattr(exc, "http_status", None)

    if val is None:
        return None

    try:
        return int(val)
    except Exception:
        return None


def is_retryable_openai_exception(exc: BaseException) -> tuple[bool, float | None, str | None]:
    """
    OpenAI retry policy aligned with documented transient failures:
    - connection/timeout errors
    - HTTP 408, 409, 429
    - HTTP 5xx
    """
    retry_after = _extract_retry_after_seconds(exc)

    try:
        from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError  # type: ignore
    except Exception:
        APIConnectionError = None  # type: ignore[assignment]
        APITimeoutError = None  # type: ignore[assignment]
        APIStatusError = None  # type: ignore[assignment]
        RateLimitError = None  # type: ignore[assignment]

    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return True, retry_after, "connection_error"

    if APITimeoutError is not None and isinstance(exc, APITimeoutError):
        return True, retry_after, "timeout"

    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True, retry_after, "rate_limited"

    if APIStatusError is not None and isinstance(exc, APIStatusError):
        code = _extract_status_code(exc)
        if code in (408, 409, 429) or (isinstance(code, int) and code >= 500):
            return True, retry_after, f"http_{code}" if code is not None else "http_status"
        return False, None, f"http_{code}" if code is not None else "http_status"

    # Fallback if SDK exceptions change.
    code = _extract_status_code(exc)
    if code in (408, 409, 429) or (isinstance(code, int) and code >= 500):
        return True, retry_after, f"http_{code}"

    return False, None, None

from __future__ import annotations


def _extract_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "statusCode", "status", "http_status", "httpStatusCode"):
        val = getattr(exc, attr, None)
        if val is None:
            continue
        try:
            return int(val)
        except Exception:
            continue
    return None


def _looks_like_timeout_or_connection(exc: BaseException) -> bool:
    # Avoid importing optional HTTP stacks; use conservative heuristics.
    name = type(exc).__name__.casefold()
    mod = type(exc).__module__.casefold()

    if "timeout" in name or "timeout" in mod:
        return True
    if "connection" in name or "connect" in name:
        return True
    if "connection" in mod or "connect" in mod:
        return True
    return False


def is_retryable_apify_exception(exc: BaseException) -> tuple[bool, float | None, str | None]:
    """
    Apify retry policy aligned with client behavior:
    - network/connection errors
    - HTTP 500+
    - HTTP 429
    """
    try:
        from apify_client.errors import ApifyApiError  # type: ignore
    except Exception:
        ApifyApiError = None  # type: ignore[assignment]

    if ApifyApiError is not None and isinstance(exc, ApifyApiError):
        code = _extract_status_code(exc)
        if code == 429 or (isinstance(code, int) and code >= 500):
            return True, None, f"http_{code}" if code is not None else "http_status"
        return False, None, f"http_{code}" if code is not None else "http_status"

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True, None, "network_error"

    if _looks_like_timeout_or_connection(exc):
        return True, None, "network_error"

    return False, None, None

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from apify_client import ApifyClient
from apify_client.errors import ApifyApiError

from .config_schema import ApifyConfig
from .errors import ApifyError


@dataclass(frozen=True)
class ActorRunRef:
    actor_id: str
    run_id: str
    default_dataset_id: str


def _normalize_terms(terms: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for raw in terms:
        term = (raw or "").strip()
        if term.startswith("#"):
            term = term[1:].strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)

    return out


def _normalize_urls(urls: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for raw in urls:
        url = (raw or "").strip()
        if not url:
            continue
        key = url.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(url)

    return out


def _chunked(values: Sequence[str], size: int) -> Iterator[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")

    batch: list[str] = []
    for item in values:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class InstagramHashtagScraper:
    """
    Thin wrapper around Apify's maintained Instagram Hashtag Scraper Actor.

    This module is deliberately low-level: it only runs the Actor and retrieves dataset items.
    """

    def __init__(self, token: str, *, client: ApifyClient | None = None) -> None:
        self._client = client or ApifyClient(token=token)

    def run_once(
        self,
        terms: Sequence[str],
        *,
        apify: ApifyConfig,
        timeout_secs: int | None = None,
    ) -> ActorRunRef:
        """
        Run the primary hashtag/keyword Actor once for a single batch of terms.

        Returns the run id and default dataset id so callers can persist run metadata.
        """
        normalized = _normalize_terms(terms)
        if not normalized:
            raise ApifyError("At least one non-empty query term is required")

        run_input: dict[str, Any] = {
            "hashtags": normalized,
            "resultsType": apify.results_type,
            "resultsLimit": apify.results_limit_per_query,
            "keywordSearch": apify.keyword_search,
        }

        try:
            result = self._client.actor(apify.primary_actor).call(
                run_input=run_input,
                timeout_secs=timeout_secs,
            )
        except ApifyApiError as e:
            raise ApifyError(
                f"Apify Actor call failed ({apify.primary_actor}): {e}"
            ) from e
        except Exception as e:
            raise ApifyError(
                f"Unexpected error while calling Apify Actor ({apify.primary_actor}): {e}"
            ) from e

        if result is None:
            raise ApifyError(f"Apify Actor run failed ({apify.primary_actor})")

        run_id = (result.get("id") or "").strip()
        dataset_id = (result.get("defaultDatasetId") or "").strip()
        if not run_id or not dataset_id:
            raise ApifyError(
                f"Apify Actor run response missing run id or default dataset id: {result}"
            )

        return ActorRunRef(
            actor_id=apify.primary_actor,
            run_id=run_id,
            default_dataset_id=dataset_id,
        )

    def iter_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int | None = None,
        clean: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream items from a dataset produced by an Actor run.

        Use clean=True to skip hidden/empty fields, which keeps downstream normalization simpler.
        """
        ds = (dataset_id or "").strip()
        if not ds:
            raise ApifyError("dataset_id must be a non-empty string")

        try:
            yield from self._client.dataset(ds).iterate_items(limit=limit, clean=clean)
        except ApifyApiError as e:
            raise ApifyError(f"Failed to read dataset items ({ds}): {e}") from e
        except Exception as e:
            raise ApifyError(f"Unexpected error while reading dataset ({ds}): {e}") from e

    def fetch_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int | None = None,
        clean: bool = True,
    ) -> list[dict[str, Any]]:
        return list(self.iter_dataset_items(dataset_id, limit=limit, clean=clean))

    def run_and_fetch(
        self,
        terms: Sequence[str],
        *,
        apify: ApifyConfig,
        timeout_secs: int | None = None,
        dataset_limit: int | None = None,
        clean: bool = True,
    ) -> tuple[ActorRunRef, list[dict[str, Any]]]:
        run = self.run_once(terms, apify=apify, timeout_secs=timeout_secs)
        items = self.fetch_dataset_items(
            run.default_dataset_id,
            limit=dataset_limit,
            clean=clean,
        )
        return run, items

    def run_and_fetch_many(
        self,
        terms: Sequence[str],
        *,
        apify: ApifyConfig,
        timeout_secs: int | None = None,
        dataset_limit_per_run: int | None = None,
        clean: bool = True,
    ) -> tuple[list[ActorRunRef], list[dict[str, Any]]]:
        """
        Run the primary actor in chunks and return merged items.

        Chunk size is controlled by apify.run_batch_queries.
        """
        normalized = _normalize_terms(terms)
        runs: list[ActorRunRef] = []
        items: list[dict[str, Any]] = []

        for batch in _chunked(normalized, apify.run_batch_queries):
            run, batch_items = self.run_and_fetch(
                batch,
                apify=apify,
                timeout_secs=timeout_secs,
                dataset_limit=dataset_limit_per_run,
                clean=clean,
            )
            runs.append(run)
            items.extend(batch_items)

        return runs, items


class InstagramScraper:
    """
    Thin wrapper around Apify's maintained Instagram Scraper Actor.

    This actor supports:
    - hashtag discovery via search (`searchType="hashtag"`)
    - scraping posts from known Instagram URLs (`directUrls`)
    """

    def __init__(self, token: str, *, client: ApifyClient | None = None) -> None:
        self._client = client or ApifyClient(token=token)

    def run_search_hashtags(
        self,
        query: str,
        *,
        apify: ApifyConfig,
        search_limit: int = 20,
        timeout_secs: int | None = None,
    ) -> ActorRunRef:
        q = (query or "").strip()
        if q.startswith("#"):
            q = q[1:].strip()
        if not q:
            raise ApifyError("search query must be non-empty")

        run_input: dict[str, Any] = {
            "search": q,
            "searchType": "hashtag",
            "searchLimit": int(search_limit),
        }

        try:
            result = self._client.actor(apify.fallback_actor).call(
                run_input=run_input,
                timeout_secs=timeout_secs,
            )
        except ApifyApiError as e:
            raise ApifyError(
                f"Apify Actor call failed ({apify.fallback_actor}): {e}"
            ) from e
        except Exception as e:
            raise ApifyError(
                f"Unexpected error while calling Apify Actor ({apify.fallback_actor}): {e}"
            ) from e

        if result is None:
            raise ApifyError(f"Apify Actor run failed ({apify.fallback_actor})")

        run_id = (result.get("id") or "").strip()
        dataset_id = (result.get("defaultDatasetId") or "").strip()
        if not run_id or not dataset_id:
            raise ApifyError(
                f"Apify Actor run response missing run id or default dataset id: {result}"
            )

        return ActorRunRef(
            actor_id=apify.fallback_actor,
            run_id=run_id,
            default_dataset_id=dataset_id,
        )

    def run_scrape_urls(
        self,
        urls: Sequence[str],
        *,
        apify: ApifyConfig,
        results_limit: int,
        timeout_secs: int | None = None,
    ) -> ActorRunRef:
        normalized_urls = _normalize_urls(urls)
        if not normalized_urls:
            raise ApifyError("At least one non-empty URL is required for directUrls")

        run_input: dict[str, Any] = {
            "directUrls": normalized_urls,
            "resultsType": apify.results_type,
            "resultsLimit": int(results_limit),
        }

        try:
            result = self._client.actor(apify.fallback_actor).call(
                run_input=run_input,
                timeout_secs=timeout_secs,
            )
        except ApifyApiError as e:
            raise ApifyError(
                f"Apify Actor call failed ({apify.fallback_actor}): {e}"
            ) from e
        except Exception as e:
            raise ApifyError(
                f"Unexpected error while calling Apify Actor ({apify.fallback_actor}): {e}"
            ) from e

        if result is None:
            raise ApifyError(f"Apify Actor run failed ({apify.fallback_actor})")

        run_id = (result.get("id") or "").strip()
        dataset_id = (result.get("defaultDatasetId") or "").strip()
        if not run_id or not dataset_id:
            raise ApifyError(
                f"Apify Actor run response missing run id or default dataset id: {result}"
            )

        return ActorRunRef(
            actor_id=apify.fallback_actor,
            run_id=run_id,
            default_dataset_id=dataset_id,
        )

    def iter_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int | None = None,
        clean: bool = True,
    ) -> Iterator[dict[str, Any]]:
        ds = (dataset_id or "").strip()
        if not ds:
            raise ApifyError("dataset_id must be a non-empty string")

        try:
            yield from self._client.dataset(ds).iterate_items(limit=limit, clean=clean)
        except ApifyApiError as e:
            raise ApifyError(f"Failed to read dataset items ({ds}): {e}") from e
        except Exception as e:
            raise ApifyError(f"Unexpected error while reading dataset ({ds}): {e}") from e

    def fetch_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int | None = None,
        clean: bool = True,
    ) -> list[dict[str, Any]]:
        return list(self.iter_dataset_items(dataset_id, limit=limit, clean=clean))

    def search_hashtags_and_fetch(
        self,
        query: str,
        *,
        apify: ApifyConfig,
        search_limit: int = 20,
        timeout_secs: int | None = None,
        dataset_limit: int | None = None,
        clean: bool = True,
    ) -> tuple[ActorRunRef, list[dict[str, Any]]]:
        run = self.run_search_hashtags(
            query,
            apify=apify,
            search_limit=search_limit,
            timeout_secs=timeout_secs,
        )
        items = self.fetch_dataset_items(run.default_dataset_id, limit=dataset_limit, clean=clean)
        return run, items

    def scrape_urls_and_fetch(
        self,
        urls: Sequence[str],
        *,
        apify: ApifyConfig,
        results_limit: int,
        timeout_secs: int | None = None,
        dataset_limit: int | None = None,
        clean: bool = True,
    ) -> tuple[ActorRunRef, list[dict[str, Any]]]:
        run = self.run_scrape_urls(
            urls,
            apify=apify,
            results_limit=results_limit,
            timeout_secs=timeout_secs,
        )
        items = self.fetch_dataset_items(run.default_dataset_id, limit=dataset_limit, clean=clean)
        return run, items

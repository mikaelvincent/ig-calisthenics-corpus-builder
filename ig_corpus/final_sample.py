from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from .errors import StorageError
from .storage import SQLiteStateStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pool_keys_sha256(pool_keys: Sequence[str]) -> str:
    """
    Hash the ordered pool key list used for sampling.

    The order matters because sampling is performed over the ordered pool.
    """
    payload = json.dumps(
        list(pool_keys),
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def pick_final_keys(pool_keys: Sequence[str], *, final_n: int, seed: int) -> set[str]:
    if final_n <= 0 or not pool_keys:
        return set()

    n = min(int(final_n), len(pool_keys))
    rng = random.Random(int(seed))
    chosen = rng.sample(list(pool_keys), k=n)
    return set(chosen)


def fetch_eligible_pool_keys(store: SQLiteStateStore, *, limit: int) -> list[str]:
    if limit <= 0:
        return []

    rows = store.conn.execute(
        """
        SELECT p.post_key
        FROM raw_posts p
        JOIN latest_llm_decisions d
          ON d.post_key = p.post_key
        WHERE d.eligible = 1
        ORDER BY d.created_at ASC, p.post_key ASC
        LIMIT ?
        """.strip(),
        (int(limit),),
    ).fetchall()

    out: list[str] = []
    for r in rows:
        out.append(str(r["post_key"]))
    return out


@dataclass(frozen=True)
class FinalSampleMeta:
    run_id: str
    sampling_seed: int
    pool_n: int
    final_n: int
    pool_keys_sha256: str
    created_at: str


def load_final_sample_meta(store: SQLiteStateStore, *, run_id: str) -> FinalSampleMeta | None:
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id must be non-empty")

    row = store.conn.execute(
        """
        SELECT run_id, sampling_seed, pool_n, final_n, pool_keys_sha256, created_at
        FROM final_sample_runs
        WHERE run_id = ?
        """.strip(),
        (rid,),
    ).fetchone()

    if row is None:
        return None

    return FinalSampleMeta(
        run_id=str(row["run_id"]),
        sampling_seed=int(row["sampling_seed"]),
        pool_n=int(row["pool_n"]),
        final_n=int(row["final_n"]),
        pool_keys_sha256=str(row["pool_keys_sha256"]),
        created_at=str(row["created_at"]),
    )


def load_final_sample_keys(store: SQLiteStateStore, *, run_id: str) -> set[str]:
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id must be non-empty")

    rows = store.conn.execute(
        """
        SELECT post_key
        FROM final_samples
        WHERE run_id = ?
        """.strip(),
        (rid,),
    ).fetchall()

    return {str(r["post_key"]) for r in rows}


def ensure_final_sample(
    store: SQLiteStateStore,
    *,
    run_id: str,
    pool_keys: Sequence[str],
    sampling_seed: int,
    pool_n: int,
    final_n: int,
    persist: bool,
) -> tuple[set[str], FinalSampleMeta | None]:
    """
    Return final sample keys for a run, optionally persisting if missing.

    If a persisted sample exists, it is always used.
    If persistence is requested and the stored meta conflicts, StorageError is raised.
    """
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id must be non-empty")

    existing = load_final_sample_meta(store, run_id=rid)
    if existing is not None:
        keys = load_final_sample_keys(store, run_id=rid)
        return keys, existing

    keys = pick_final_keys(pool_keys, final_n=int(final_n), seed=int(sampling_seed))
    if not persist:
        return keys, None

    actual_pool_sha = pool_keys_sha256(pool_keys)
    meta = FinalSampleMeta(
        run_id=rid,
        sampling_seed=int(sampling_seed),
        pool_n=int(pool_n),
        final_n=int(final_n),
        pool_keys_sha256=actual_pool_sha,
        created_at=_utc_now_iso(),
    )

    try:
        with store.conn:
            store.conn.execute(
                """
                INSERT OR IGNORE INTO final_sample_runs(
                  run_id, sampling_seed, pool_n, final_n, pool_keys_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    meta.run_id,
                    meta.sampling_seed,
                    meta.pool_n,
                    meta.final_n,
                    meta.pool_keys_sha256,
                    meta.created_at,
                ),
            )

            store.conn.executemany(
                """
                INSERT OR IGNORE INTO final_samples(run_id, post_key)
                VALUES (?, ?)
                """.strip(),
                [(meta.run_id, k) for k in sorted(keys)],
            )
    except Exception as e:
        raise StorageError(f"Failed to persist final sample: {e}") from e

    stored = load_final_sample_meta(store, run_id=rid)
    if stored is None:
        raise StorageError("Failed to read final sample metadata after insert")

    if (
        stored.sampling_seed != meta.sampling_seed
        or stored.pool_n != meta.pool_n
        or stored.final_n != meta.final_n
        or stored.pool_keys_sha256 != meta.pool_keys_sha256
    ):
        raise StorageError("Persisted final sample metadata did not match expected values")

    return keys, stored

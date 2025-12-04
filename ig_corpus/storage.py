from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import StorageError
from .llm_schema import LLMDecision
from .storage_schema import initialize_sqlite


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _as_path(value: str | Path) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    started_at: str
    ended_at: str | None
    config_hash: str
    sampling_seed: int | None
    versions: dict[str, str]


@dataclass(frozen=True)
class EligiblePostRecord:
    post_key: str
    url: str
    actor_source: str | None
    fetched_at: str
    model: str
    overall_confidence: float
    tokens_total: int | None
    decided_at: str
    decision_json: str


class SQLiteStateStore:
    """
    Small persistence layer for a resume-capable run state.

    This store is intentionally minimal: it keeps raw items, model decisions, and run metadata.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, path: str | Path) -> "SQLiteStateStore":
        db_path = _as_path(path)
        if db_path != ":memory:":
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(db_path)
        except sqlite3.DatabaseError as e:
            raise StorageError(f"Failed to open sqlite database: {db_path}: {e}") from e

        try:
            initialize_sqlite(conn)
        except Exception as e:
            conn.close()
            raise StorageError(f"Failed to initialize sqlite schema: {e}") from e

        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteStateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def create_run(
        self,
        *,
        config_hash: str,
        sampling_seed: int | None = None,
        versions: Mapping[str, str] | None = None,
        run_id: str | None = None,
        started_at: str | None = None,
    ) -> RunRecord:
        rid = (run_id or uuid.uuid4().hex).strip()
        if not rid:
            raise ValueError("run_id must be non-empty")

        cfg_hash = (config_hash or "").strip()
        if not cfg_hash:
            raise ValueError("config_hash must be non-empty")

        start = (started_at or _utc_now_iso()).strip()
        versions_json = _json_dumps(dict(versions or {}))

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO runs(
                      run_id, started_at, ended_at, config_hash, sampling_seed, versions_json
                    ) VALUES (?, ?, NULL, ?, ?, ?)
                    """.strip(),
                    (rid, start, cfg_hash, sampling_seed, versions_json),
                )
        except sqlite3.DatabaseError as e:
            raise StorageError(f"Failed to create run record: {e}") from e

        record = self.get_run(rid)
        if record is None:
            raise StorageError("Failed to read run record after insert")
        return record

    def finish_run(self, run_id: str, *, ended_at: str | None = None) -> None:
        rid = (run_id or "").strip()
        if not rid:
            raise ValueError("run_id must be non-empty")

        end = (ended_at or _utc_now_iso()).strip()

        try:
            with self._conn:
                self._conn.execute(
                    "UPDATE runs SET ended_at = ? WHERE run_id = ?",
                    (end, rid),
                )
        except sqlite3.DatabaseError as e:
            raise StorageError(f"Failed to finish run: {e}") from e

    def get_run(self, run_id: str) -> RunRecord | None:
        rid = (run_id or "").strip()
        if not rid:
            raise ValueError("run_id must be non-empty")

        row = self._conn.execute(
            "SELECT run_id, started_at, ended_at, config_hash, sampling_seed, versions_json FROM runs WHERE run_id = ?",
            (rid,),
        ).fetchone()
        if row is None:
            return None

        versions_raw = (row["versions_json"] or "{}").strip()
        try:
            versions = json.loads(versions_raw)
        except Exception:
            versions = {}

        if not isinstance(versions, dict):
            versions = {}

        return RunRecord(
            run_id=str(row["run_id"]),
            started_at=str(row["started_at"]),
            ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
            config_hash=str(row["config_hash"]),
            sampling_seed=int(row["sampling_seed"]) if row["sampling_seed"] is not None else None,
            versions={str(k): str(v) for k, v in versions.items()},
        )

    def record_apify_actor_run(
        self,
        *,
        run_id: str,
        actor_id: str,
        actor_run_id: str,
        dataset_id: str,
        created_at: str | None = None,
    ) -> None:
        rid = (run_id or "").strip()
        if not rid:
            raise ValueError("run_id must be non-empty")

        a = (actor_id or "").strip()
        r = (actor_run_id or "").strip()
        d = (dataset_id or "").strip()
        if not a or not r or not d:
            raise ValueError("actor_id, actor_run_id, and dataset_id must be non-empty")

        ts = (created_at or _utc_now_iso()).strip()

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO apify_actor_runs(
                      run_id, actor_id, actor_run_id, dataset_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """.strip(),
                    (rid, a, r, d, ts),
                )
        except sqlite3.DatabaseError as e:
            raise StorageError(f"Failed to record actor run: {e}") from e

    def upsert_raw_post(
        self,
        *,
        post_key: str,
        url: str,
        raw_item: Mapping[str, Any],
        actor_source: str | None = None,
        fetched_at: str | None = None,
    ) -> None:
        key = (post_key or "").strip()
        u = (url or "").strip()
        if not key or not u:
            raise ValueError("post_key and url must be non-empty")

        raw_json = _json_dumps(dict(raw_item))
        src = (actor_source or "").strip() or None
        ts = (fetched_at or _utc_now_iso()).strip()

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO raw_posts(post_key, url, actor_source, raw_json, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(post_key) DO UPDATE SET
                      url = excluded.url,
                      actor_source = COALESCE(excluded.actor_source, raw_posts.actor_source),
                      raw_json = excluded.raw_json,
                      fetched_at = excluded.fetched_at
                    """.strip(),
                    (key, u, src, raw_json, ts),
                )
        except sqlite3.DatabaseError as e:
            raise StorageError(f"Failed to upsert raw post: {e}") from e

    def seen_post_keys(self, *, limit: int | None = None) -> set[str]:
        if limit is not None and limit <= 0:
            return set()

        sql = "SELECT post_key FROM raw_posts"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)

        rows = self._conn.execute(sql, params).fetchall()
        return {str(r["post_key"]) for r in rows}

    def record_llm_decision(
        self,
        *,
        post_key: str,
        url: str,
        model: str,
        decision: LLMDecision,
        tokens_total: int | None = None,
        created_at: str | None = None,
    ) -> None:
        key = (post_key or "").strip()
        u = (url or "").strip()
        m = (model or "").strip()
        if not key or not u or not m:
            raise ValueError("post_key, url, and model must be non-empty")

        eligible_int = 1 if decision.eligible else 0
        confidence = float(decision.overall_confidence)
        decision_json = decision.model_dump_json(
            indent=None,
            by_alias=False,
            exclude_none=False,
        )
        ts = (created_at or _utc_now_iso()).strip()

        tok = int(tokens_total) if tokens_total is not None else None

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO llm_decisions(
                      post_key, url, model, eligible, overall_confidence,
                      decision_json, tokens_total, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """.strip(),
                    (key, u, m, eligible_int, confidence, decision_json, tok, ts),
                )
        except sqlite3.IntegrityError as e:
            raise StorageError(
                "Failed to record decision; ensure raw_posts contains post_key first"
            ) from e
        except sqlite3.DatabaseError as e:
            raise StorageError(f"Failed to record decision: {e}") from e

    def latest_decision(self, post_key: str) -> LLMDecision | None:
        key = (post_key or "").strip()
        if not key:
            raise ValueError("post_key must be non-empty")

        row = self._conn.execute(
            """
            SELECT decision_json
            FROM llm_decisions
            WHERE post_key = ?
            ORDER BY id DESC
            LIMIT 1
            """.strip(),
            (key,),
        ).fetchone()
        if row is None:
            return None

        raw = (row["decision_json"] or "").strip()
        if not raw:
            return None

        try:
            return LLMDecision.model_validate_json(raw)
        except Exception as e:
            raise StorageError(f"Stored decision_json could not be parsed: {e}") from e

    def eligible_posts(self, *, limit: int | None = None) -> list[EligiblePostRecord]:
        if limit is not None and limit <= 0:
            return []

        sql = """
        SELECT
          post_key, url, actor_source, fetched_at, model, overall_confidence,
          tokens_total, decided_at, decision_json
        FROM eligible_posts
        ORDER BY decided_at DESC
        """.strip()
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)

        rows = self._conn.execute(sql, params).fetchall()
        out: list[EligiblePostRecord] = []
        for r in rows:
            out.append(
                EligiblePostRecord(
                    post_key=str(r["post_key"]),
                    url=str(r["url"]),
                    actor_source=str(r["actor_source"]) if r["actor_source"] is not None else None,
                    fetched_at=str(r["fetched_at"]),
                    model=str(r["model"]),
                    overall_confidence=float(r["overall_confidence"]),
                    tokens_total=int(r["tokens_total"]) if r["tokens_total"] is not None else None,
                    decided_at=str(r["decided_at"]),
                    decision_json=str(r["decision_json"]),
                )
            )
        return out

    def raw_post_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(1) AS n FROM raw_posts").fetchone()
        return int(row["n"]) if row is not None else 0

    def decision_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(1) AS n FROM llm_decisions").fetchone()
        return int(row["n"]) if row is not None else 0

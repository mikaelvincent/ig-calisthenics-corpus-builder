from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_sqlite(conn: sqlite3.Connection) -> None:
    """
    Initialize the SQLite database with a small migration system.

    This function is idempotent: it can be called on every startup.
    """
    _configure_connection(conn)
    _apply_migrations(conn)


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")

    # WAL is best-effort (e.g., in-memory DBs won't use it).
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.DatabaseError:
        pass


_MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  config_hash TEXT NOT NULL,
  sampling_seed INTEGER,
  versions_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS apify_actor_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  actor_run_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  UNIQUE (run_id, actor_run_id)
);

CREATE INDEX IF NOT EXISTS idx_apify_actor_runs_run_id
  ON apify_actor_runs(run_id);

CREATE TABLE IF NOT EXISTS raw_posts (
  post_key TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  actor_source TEXT,
  raw_json TEXT NOT NULL,
  fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_posts_url
  ON raw_posts(url);

CREATE TABLE IF NOT EXISTS llm_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  post_key TEXT NOT NULL,
  url TEXT NOT NULL,
  model TEXT NOT NULL,
  eligible INTEGER NOT NULL,
  overall_confidence REAL NOT NULL,
  decision_json TEXT NOT NULL,
  tokens_total INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY (post_key) REFERENCES raw_posts(post_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_llm_decisions_post_key
  ON llm_decisions(post_key);

CREATE INDEX IF NOT EXISTS idx_llm_decisions_eligible
  ON llm_decisions(eligible);

CREATE INDEX IF NOT EXISTS idx_llm_decisions_created_at
  ON llm_decisions(created_at);

-- One row per post_key: the most recently inserted decision.
CREATE VIEW IF NOT EXISTS latest_llm_decisions AS
SELECT d.*
FROM llm_decisions d
JOIN (
  SELECT post_key, MAX(id) AS max_id
  FROM llm_decisions
  GROUP BY post_key
) latest
ON latest.post_key = d.post_key AND latest.max_id = d.id;

CREATE VIEW IF NOT EXISTS eligible_posts AS
SELECT
  p.post_key,
  p.url,
  p.actor_source,
  p.fetched_at,
  d.model,
  d.overall_confidence,
  d.tokens_total,
  d.created_at AS decided_at,
  d.decision_json
FROM raw_posts p
JOIN latest_llm_decisions d
  ON d.post_key = p.post_key
WHERE d.eligible = 1;
""".strip()
}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )

    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    applied: set[int] = {int(r[0]) for r in rows}

    for version in range(1, SCHEMA_VERSION + 1):
        if version in applied:
            continue

        script = _MIGRATIONS.get(version)
        if not script:
            raise RuntimeError(f"Missing migration script for version={version}")

        with conn:
            conn.executescript(script)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, _utc_now_iso()),
            )

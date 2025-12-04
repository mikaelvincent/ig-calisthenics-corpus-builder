from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .config_schema import AppConfig
from .llm_schema import LLMDecision
from .normalize import normalized_post_from_apify_item
from .storage import RunRecord, SQLiteStateStore


INCLUSION_RULES: tuple[str, ...] = (
    "Caption language is English (reject if mostly non-English or too mixed).",
    "Content is clearly about calisthenics / street workout / bodyweight training.",
    "Caption is linguistically analyzable (reject empty/emoji-only/hashtag-only).",
    "Not exclusively commercial (reject pure ads/discount codes with no training substance).",
)

EXCLUSION_RULES: tuple[str, ...] = (
    "Gym-only weightlifting/bodybuilding/powerlifting content.",
    "CrossFit-only, yoga, parkour, bouldering, and other unrelated sports.",
)

TAG_FIELD_DEFINITIONS: dict[str, str] = {
    "genre": "Single best-fit narrative/functional category for the caption.",
    "narrative_labels": "1–3 short emergent labels, free-form.",
    "discourse_moves": "Communicative moves present (e.g., advice, confession, challenge, call-to-action).",
    "neoliberal_signals": "Only if present; self-optimization/hustle/productivity framing, etc.",
}

GENRE_VALUES: tuple[str, ...] = (
    "training_log",
    "tutorial_coaching",
    "motivation_mindset",
    "personal_story_reflection",
    "identity_community",
    "transformation_progress",
    "injury_rehab",
    "humor_meme",
    "educational_sciency",
    "other",
)


@dataclass(frozen=True)
class ActorRunInfo:
    actor_id: str
    actor_run_id: str
    dataset_id: str
    created_at: str


@dataclass(frozen=True)
class CodebookCounts:
    raw_posts: int
    decision_records: int
    labeled_posts: int
    eligible_total: int
    rejected_total: int
    eligible_in_pool: int
    final_sample_n: int


@dataclass(frozen=True)
class CodebookStats:
    top_hashtags: list[tuple[str, int]]
    top_genres: list[tuple[str, int]]
    top_narrative_labels: list[tuple[str, int]]
    top_neoliberal_signals: list[tuple[str, int]]


@dataclass(frozen=True)
class CodebookData:
    run: RunRecord | None
    counts: CodebookCounts
    actor_runs: list[ActorRunInfo]
    stats: CodebookStats

    inclusion_rules: tuple[str, ...]
    exclusion_rules: tuple[str, ...]
    tag_field_definitions: dict[str, str]
    genre_values: tuple[str, ...]


def _db_scalar_int(store: SQLiteStateStore, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = store.conn.execute(sql, params).fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except Exception:
        return 0


def _fetch_actor_runs(store: SQLiteStateStore, *, run_id: str) -> list[ActorRunInfo]:
    rid = (run_id or "").strip()
    if not rid:
        return []

    rows = store.conn.execute(
        """
        SELECT actor_id, actor_run_id, dataset_id, created_at
        FROM apify_actor_runs
        WHERE run_id = ?
        ORDER BY created_at ASC, actor_run_id ASC
        """.strip(),
        (rid,),
    ).fetchall()

    out: list[ActorRunInfo] = []
    for r in rows:
        out.append(
            ActorRunInfo(
                actor_id=str(r["actor_id"]),
                actor_run_id=str(r["actor_run_id"]),
                dataset_id=str(r["dataset_id"]),
                created_at=str(r["created_at"]),
            )
        )
    return out


def _fetch_eligible_pool_rows(
    store: SQLiteStateStore, *, limit: int
) -> list[tuple[str, str, str]]:
    if limit <= 0:
        return []

    rows = store.conn.execute(
        """
        SELECT p.post_key, p.raw_json, d.decision_json
        FROM raw_posts p
        JOIN latest_llm_decisions d
          ON d.post_key = p.post_key
        WHERE d.eligible = 1
        ORDER BY d.created_at ASC, p.post_key ASC
        LIMIT ?
        """.strip(),
        (int(limit),),
    ).fetchall()

    out: list[tuple[str, str, str]] = []
    for r in rows:
        out.append((str(r["post_key"]), str(r["raw_json"]), str(r["decision_json"])))
    return out


def _top(counter: Counter[str], *, limit: int) -> list[tuple[str, int]]:
    items = [(k, int(v)) for k, v in counter.items() if (k or "").strip() and int(v) > 0]
    items.sort(key=lambda x: (-x[1], x[0].casefold(), x[0]))
    return items[: max(0, int(limit))]


def collect_codebook_data(config: AppConfig, store: SQLiteStateStore, *, run_id: str) -> CodebookData:
    run = store.get_run(run_id)

    raw_posts = int(store.raw_post_count())
    decision_records = int(store.decision_count())
    labeled_posts = _db_scalar_int(store, "SELECT COUNT(1) FROM latest_llm_decisions")
    eligible_total = _db_scalar_int(store, "SELECT COUNT(1) FROM eligible_posts")
    rejected_total = max(0, int(labeled_posts) - int(eligible_total))

    pool_limit = int(config.targets.pool_n)
    eligible_rows = _fetch_eligible_pool_rows(store, limit=pool_limit)
    eligible_in_pool = len(eligible_rows)

    final_sample_n = min(int(config.targets.final_n), eligible_in_pool)

    hashtag_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    narrative_counts: Counter[str] = Counter()
    neoliberal_counts: Counter[str] = Counter()

    for _, raw_json, decision_json in eligible_rows:
        item: Any
        try:
            item = json.loads(raw_json)
        except Exception:
            item = None

        if isinstance(item, dict):
            post = normalized_post_from_apify_item(item)
            if post is not None:
                for tag in post.hashtags:
                    t = (tag or "").strip()
                    if t:
                        hashtag_counts[t] += 1

        try:
            decision = LLMDecision.model_validate_json(decision_json)
        except Exception:
            continue

        genre_counts[str(decision.tags.genre)] += 1

        for lab in decision.tags.narrative_labels:
            t = (lab or "").strip()
            if t:
                narrative_counts[t] += 1

        for sig in decision.tags.neoliberal_signals:
            t = (sig or "").strip()
            if t:
                neoliberal_counts[t] += 1

    stats = CodebookStats(
        top_hashtags=_top(hashtag_counts, limit=25),
        top_genres=_top(genre_counts, limit=20),
        top_narrative_labels=_top(narrative_counts, limit=25),
        top_neoliberal_signals=_top(neoliberal_counts, limit=25),
    )

    actor_runs = _fetch_actor_runs(store, run_id=run_id)

    counts = CodebookCounts(
        raw_posts=raw_posts,
        decision_records=decision_records,
        labeled_posts=labeled_posts,
        eligible_total=eligible_total,
        rejected_total=rejected_total,
        eligible_in_pool=eligible_in_pool,
        final_sample_n=final_sample_n,
    )

    return CodebookData(
        run=run,
        counts=counts,
        actor_runs=actor_runs,
        stats=stats,
        inclusion_rules=INCLUSION_RULES,
        exclusion_rules=EXCLUSION_RULES,
        tag_field_definitions=dict(TAG_FIELD_DEFINITIONS),
        genre_values=GENRE_VALUES,
    )

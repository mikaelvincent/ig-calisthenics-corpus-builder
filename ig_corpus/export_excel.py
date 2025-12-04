from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config_schema import AppConfig
from .errors import ExportError
from .final_sample import (
    ensure_final_sample,
    load_final_sample_keys,
    load_final_sample_meta,
    pool_keys_sha256,
)
from .llm_schema import LLMDecision
from .normalize import normalized_post_from_apify_item
from .storage import SQLiteStateStore


_EXCEL_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_excel_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    s = value
    if not s:
        return s
    if s.startswith(_EXCEL_FORMULA_PREFIXES):
        return "'" + s
    return s


def _fmt_space_join(values: Iterable[str], *, prefix: str = "") -> str:
    out: list[str] = []
    for v in values:
        t = (v or "").strip()
        if t:
            out.append(f"{prefix}{t}" if prefix else t)
    return " ".join(out)


def _fmt_pipe_join(values: Iterable[str]) -> str:
    out: list[str] = []
    for v in values:
        t = (v or "").strip()
        if t:
            out.append(t)
    return " | ".join(out)


def _loads_json_object(raw: str) -> dict[str, Any]:
    try:
        val = json.loads(raw)
    except Exception as e:
        raise ExportError(f"Failed to parse stored JSON: {e}") from e
    if not isinstance(val, dict):
        raise ExportError("Stored JSON was not an object")
    return val


def _db_scalar_int(store: SQLiteStateStore, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = store.conn.execute(sql, params).fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except Exception:
        return 0


def _fetch_latest_posts_with_decisions(
    store: SQLiteStateStore,
    *,
    eligible: bool,
    limit: int | None,
    order_asc: bool,
) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        return []

    where = "d.eligible = 1" if eligible else "d.eligible = 0"
    order = "ASC" if order_asc else "DESC"

    sql = f"""
    SELECT
      p.post_key,
      p.url,
      p.actor_source,
      p.fetched_at,
      p.raw_json,
      d.model,
      d.overall_confidence,
      d.tokens_total,
      d.created_at AS decided_at,
      d.decision_json
    FROM raw_posts p
    JOIN latest_llm_decisions d
      ON d.post_key = p.post_key
    WHERE {where}
    ORDER BY d.created_at {order}, p.post_key {order}
    """.strip()

    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)

    rows = store.conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    return out


def _flatten_row(
    *,
    post_key: str,
    url: str,
    actor_source: str | None,
    fetched_at: str,
    raw_json: str,
    model: str,
    tokens_total: int | None,
    decided_at: str,
    overall_confidence: float,
    decision_json: str,
    selected_final: bool,
) -> tuple[dict[str, Any], LLMDecision]:
    raw_item = _loads_json_object(raw_json)
    post = normalized_post_from_apify_item(raw_item)

    if post is None:
        caption = None
        hashtags = ()
        mentions = ()
        alt = None
        post_type = None
        product_type = None
        is_sponsored = None
        timestamp = None
        owner_username = None
        owner_id = None
    else:
        caption = post.caption
        hashtags = post.hashtags
        mentions = post.mentions
        alt = post.alt
        post_type = post.type
        product_type = post.product_type
        is_sponsored = post.is_sponsored
        timestamp = post.timestamp
        owner_username = post.owner_username
        owner_id = post.owner_id

    try:
        decision = LLMDecision.model_validate_json(decision_json)
    except Exception as e:
        raise ExportError(f"Failed to parse decision_json for {post_key}: {e}") from e

    row: dict[str, Any] = {
        "post_key": _safe_excel_text(post_key),
        "url": _safe_excel_text(url),
        "owner_username": _safe_excel_text(owner_username),
        "owner_id": _safe_excel_text(owner_id),
        "caption": _safe_excel_text(caption),
        "hashtags": _safe_excel_text(_fmt_space_join(hashtags, prefix="#")),
        "mentions": _safe_excel_text(_fmt_space_join(mentions, prefix="@")),
        "alt": _safe_excel_text(alt),
        "type": _safe_excel_text(post_type),
        "product_type": _safe_excel_text(product_type),
        "is_sponsored": is_sponsored,
        "timestamp": _safe_excel_text(timestamp),
        "actor_source": _safe_excel_text(actor_source),
        "fetched_at": _safe_excel_text(fetched_at),
        "decided_at": _safe_excel_text(decided_at),
        "model": _safe_excel_text(model),
        "tokens_total": int(tokens_total) if tokens_total is not None else None,
        "overall_confidence": float(overall_confidence),
        "language_is_english": bool(decision.language.is_english),
        "language_confidence": float(decision.language.confidence),
        "topic_is_bodyweight_calisthenics": bool(decision.topic.is_bodyweight_calisthenics),
        "topic_confidence": float(decision.topic.confidence),
        "topic_notes": _safe_excel_text(decision.topic.topic_notes),
        "commercial_is_exclusively_commercial": bool(decision.commercial.is_exclusively_commercial),
        "commercial_signals": _safe_excel_text(_fmt_pipe_join(decision.commercial.signals)),
        "caption_quality_is_analyzable": bool(decision.caption_quality.is_analyzable),
        "caption_quality_issues": _safe_excel_text(_fmt_pipe_join(decision.caption_quality.issues)),
        "tags_genre": _safe_excel_text(decision.tags.genre),
        "tags_narrative_labels": _safe_excel_text(_fmt_pipe_join(decision.tags.narrative_labels)),
        "tags_discourse_moves": _safe_excel_text(_fmt_pipe_join(decision.tags.discourse_moves)),
        "tags_neoliberal_signals": _safe_excel_text(_fmt_pipe_join(decision.tags.neoliberal_signals)),
        "eligibility_reasons": _safe_excel_text(_fmt_pipe_join(decision.eligibility_reasons)),
        "selected_final": bool(selected_final),
    }
    return row, decision


def _fetch_actor_runs(store: SQLiteStateStore, *, run_id: str) -> list[dict[str, Any]]:
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

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "actor_id": _safe_excel_text(r["actor_id"]),
                "actor_run_id": _safe_excel_text(r["actor_run_id"]),
                "dataset_id": _safe_excel_text(r["dataset_id"]),
                "created_at": _safe_excel_text(r["created_at"]),
            }
        )
    return out


def export_corpus_workbook(
    config: AppConfig,
    store: SQLiteStateStore,
    out_path: str | Path,
    *,
    run_id: str,
) -> Path:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:
        raise ExportError("pandas is required for Excel export") from e

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pool_limit = int(config.targets.pool_n)
    final_n = int(config.targets.final_n)
    seed = int(config.targets.sampling_seed)

    pool_raw = _fetch_latest_posts_with_decisions(
        store,
        eligible=True,
        limit=pool_limit,
        order_asc=True,
    )

    pool_keys = [str(r["post_key"]) for r in pool_raw if str(r.get("post_key") or "").strip()]
    pool_sha = pool_keys_sha256(pool_keys)

    meta = load_final_sample_meta(store, run_id=run_id)
    if meta is not None:
        final_keys = load_final_sample_keys(store, run_id=run_id)
    else:
        final_keys, meta = ensure_final_sample(
            store,
            run_id=run_id,
            pool_keys=pool_keys,
            sampling_seed=seed,
            pool_n=pool_limit,
            final_n=final_n,
            persist=len(pool_keys) >= pool_limit,
        )

    genre_counts: Counter[str] = Counter()
    narrative_counts: Counter[str] = Counter()
    discourse_counts: Counter[str] = Counter()
    neoliberal_counts: Counter[str] = Counter()
    hashtag_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()

    pool_rows: list[dict[str, Any]] = []
    for r in pool_raw:
        post_key = str(r["post_key"])
        selected = post_key in final_keys
        row, decision = _flatten_row(
            post_key=post_key,
            url=str(r["url"]),
            actor_source=str(r["actor_source"]) if r["actor_source"] is not None else None,
            fetched_at=str(r["fetched_at"]),
            raw_json=str(r["raw_json"]),
            model=str(r["model"]),
            tokens_total=int(r["tokens_total"]) if r["tokens_total"] is not None else None,
            decided_at=str(r["decided_at"]),
            overall_confidence=float(r["overall_confidence"]),
            decision_json=str(r["decision_json"]),
            selected_final=selected,
        )
        pool_rows.append(row)

        model_counts[str(r["model"])] += 1
        genre_counts[str(decision.tags.genre)] += 1

        for lab in decision.tags.narrative_labels:
            t = (lab or "").strip()
            if t:
                narrative_counts[t] += 1

        for mv in decision.tags.discourse_moves:
            t = (mv or "").strip()
            if t:
                discourse_counts[t] += 1

        for sig in decision.tags.neoliberal_signals:
            t = (sig or "").strip()
            if t:
                neoliberal_counts[t] += 1

        hashtags_text = str(row.get("hashtags") or "").strip()
        if hashtags_text:
            for token in hashtags_text.split():
                h = token.lstrip("#").strip()
                if h:
                    hashtag_counts[h] += 1

    final_rows = [r for r in pool_rows if bool(r.get("selected_final"))]

    rejected_limit = min(5000, int(config.loop.max_raw_items))
    rejected_raw = _fetch_latest_posts_with_decisions(
        store,
        eligible=False,
        limit=rejected_limit,
        order_asc=False,
    )

    rejected_rows: list[dict[str, Any]] = []
    for r in rejected_raw:
        row, _ = _flatten_row(
            post_key=str(r["post_key"]),
            url=str(r["url"]),
            actor_source=str(r["actor_source"]) if r["actor_source"] is not None else None,
            fetched_at=str(r["fetched_at"]),
            raw_json=str(r["raw_json"]),
            model=str(r["model"]),
            tokens_total=int(r["tokens_total"]) if r["tokens_total"] is not None else None,
            decided_at=str(r["decided_at"]),
            overall_confidence=float(r["overall_confidence"]),
            decision_json=str(r["decision_json"]),
            selected_final=False,
        )
        rejected_rows.append(row)

    run = store.get_run(run_id)

    versions = run.versions if run is not None else {}
    versions_json = json.dumps(versions, ensure_ascii=False, sort_keys=True)
    config_yaml = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=True,
        allow_unicode=True,
    )

    schema_version = _db_scalar_int(store, "SELECT MAX(version) FROM schema_migrations")

    tokens_used = _db_scalar_int(
        store,
        """
        SELECT COALESCE(SUM(tokens_total), 0)
        FROM llm_decisions
        """.strip(),
    )

    meta_rows: list[dict[str, Any]] = [
        {"key": "run_id", "value": _safe_excel_text(run_id)},
        {"key": "exported_at_utc", "value": _safe_excel_text(_utc_now_iso())},
        {"key": "sqlite_schema_version", "value": int(schema_version)},
        {"key": "status_note", "value": _safe_excel_text("See sheets for outputs")},
        {"key": "targets.final_n", "value": final_n},
        {"key": "targets.pool_n", "value": pool_limit},
        {"key": "targets.sampling_seed", "value": seed},
        {"key": "counts.eligible_in_sheet", "value": len(pool_rows)},
        {"key": "counts.final_in_sheet", "value": len(final_rows)},
        {"key": "counts.rejected_in_sheet", "value": len(rejected_rows)},
        {"key": "limits.rejected_sheet_cap", "value": rejected_limit},
        {"key": "run.started_at", "value": _safe_excel_text(run.started_at if run is not None else None)},
        {"key": "run.ended_at", "value": _safe_excel_text(run.ended_at if run is not None else None)},
        {"key": "run.config_hash", "value": _safe_excel_text(run.config_hash if run is not None else None)},
        {"key": "run.sampling_seed", "value": run.sampling_seed if run is not None else None},
        {"key": "versions_json", "value": _safe_excel_text(versions_json)},
        {"key": "config_yaml", "value": _safe_excel_text(config_yaml)},
        {"key": "repro.pool_keys_sha256", "value": _safe_excel_text(pool_sha)},
        {
            "key": "repro.final_post_keys_json",
            "value": _safe_excel_text(json.dumps(sorted(final_keys), ensure_ascii=False)),
        },
        {"key": "repro.final_sample_recorded", "value": bool(meta is not None)},
        {"key": "repro.final_sample_recorded_at", "value": _safe_excel_text(meta.created_at if meta is not None else None)},
        {"key": "counts.llm_tokens_total_all_time", "value": int(tokens_used)},
        {"key": "output_path", "value": _safe_excel_text(str(out))},
    ]

    actor_runs = _fetch_actor_runs(store, run_id=run_id)

    tag_rows: list[dict[str, Any]] = []
    for genre, n in genre_counts.most_common():
        if (genre or "").strip():
            tag_rows.append({"kind": "genre", "label": _safe_excel_text(genre), "count": int(n)})

    for lab, n in narrative_counts.most_common(200):
        if (lab or "").strip():
            tag_rows.append({"kind": "narrative_label", "label": _safe_excel_text(lab), "count": int(n)})

    for mv, n in discourse_counts.most_common(200):
        if (mv or "").strip():
            tag_rows.append({"kind": "discourse_move", "label": _safe_excel_text(mv), "count": int(n)})

    for sig, n in neoliberal_counts.most_common(200):
        if (sig or "").strip():
            tag_rows.append({"kind": "neoliberal_signal", "label": _safe_excel_text(sig), "count": int(n)})

    for h, n in hashtag_counts.most_common(200):
        if (h or "").strip():
            tag_rows.append({"kind": "hashtag", "label": _safe_excel_text(h), "count": int(n)})

    for m, n in model_counts.most_common():
        if (m or "").strip():
            tag_rows.append({"kind": "model", "label": _safe_excel_text(m), "count": int(n)})

    df_final = pd.DataFrame(final_rows)
    df_pool = pd.DataFrame(pool_rows)
    df_rejected = pd.DataFrame(rejected_rows)
    df_meta = pd.DataFrame(meta_rows)
    df_actor = pd.DataFrame(actor_runs)
    df_tags = pd.DataFrame(tag_rows)

    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_final.to_excel(writer, sheet_name="final500", index=False)
            df_pool.to_excel(writer, sheet_name="eligible_pool", index=False)
            df_rejected.to_excel(writer, sheet_name="rejected", index=False)

            df_meta.to_excel(writer, sheet_name="run_metadata", index=False)
            start = len(df_meta.index) + 2
            df_actor.to_excel(writer, sheet_name="run_metadata", index=False, startrow=start)

            df_tags.to_excel(writer, sheet_name="tag_summary", index=False)

            wb = writer.book
            for name in ("final500", "eligible_pool", "rejected", "run_metadata", "tag_summary"):
                if name in wb.sheetnames:
                    ws = wb[name]
                    ws.freeze_panes = "A2"
    except Exception as e:
        raise ExportError(f"Failed to write workbook: {out}: {e}") from e

    return out

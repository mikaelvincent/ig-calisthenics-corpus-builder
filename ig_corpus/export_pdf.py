from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .codebook import CodebookData, collect_codebook_data
from .config_schema import AppConfig
from .errors import ExportError
from .storage import SQLiteStateStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _p(text: str, style: Any) -> Any:
    safe = escape(text or "").replace("\n", "<br/>")
    return style.__class__(safe, style) if hasattr(style, "__class__") else safe


def export_codebook_pdf(
    config: AppConfig,
    store: SQLiteStateStore,
    out_path: str | Path,
    *,
    run_id: str,
) -> Path:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            ListFlowable,
            ListItem,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib import colors
    except Exception as e:
        raise ExportError("reportlab is required for PDF codebook export") from e

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data: CodebookData = collect_codebook_data(config, store, run_id=run_id)

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    body_small = ParagraphStyle(
        "BodySmall",
        parent=body,
        fontSize=9,
        leading=11,
        spaceAfter=6,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=body_small,
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
    )

    def P(text: str, style: Any = body) -> Paragraph:
        return Paragraph(escape(text or "").replace("\n", "<br/>"), style)

    def bullets(items: list[str]) -> ListFlowable:
        flow_items = [ListItem(P(i, body), leftIndent=14) for i in items]
        return ListFlowable(flow_items, bulletType="bullet", leftIndent=20)

    def kv_table(rows: list[tuple[str, str]]) -> Table:
        tbl = Table(
            [[P(k, body_small), P(v, body_small)] for k, v in rows],
            colWidths=[2.1 * inch, 4.7 * inch],
            hAlign="LEFT",
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return tbl

    def top_table(title: str, pairs: list[tuple[str, int]], *, limit: int) -> list[Any]:
        story: list[Any] = [P(title, h3)]
        if not pairs:
            story.append(P("No data available.", body_small))
            return story

        rows = [["label", "count"]]
        for label, n in pairs[: max(0, int(limit))]:
            rows.append([label, str(int(n))])

        tbl = Table(rows, colWidths=[5.4 * inch, 1.4 * inch], hAlign="LEFT")
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(tbl)
        return story

    run = data.run
    run_rows: list[tuple[str, str]] = [
        ("run_id", run_id),
        ("exported_at_utc", _utc_now_iso()),
        ("config_hash", run.config_hash if run is not None else "unknown"),
        ("sampling_seed", str(run.sampling_seed) if (run is not None and run.sampling_seed is not None) else "unknown"),
        ("run_started_at", run.started_at if run is not None else "unknown"),
        ("run_ended_at", run.ended_at if (run is not None and run.ended_at is not None) else "not_recorded"),
    ]

    cfg_rows: list[tuple[str, str]] = [
        ("primary_actor", config.apify.primary_actor),
        ("fallback_actor", config.apify.fallback_actor),
        ("keyword_search", str(bool(config.apify.keyword_search))),
        ("results_limit_per_query", str(int(config.apify.results_limit_per_query))),
        ("run_batch_queries", str(int(config.apify.run_batch_queries))),
        ("openai_model_primary", config.openai.model_primary),
        ("openai_model_escalation", config.openai.model_escalation),
        ("escalation_conf_threshold", str(float(config.openai.escalation_confidence_threshold))),
        ("min_caption_chars", str(int(config.filters.min_caption_chars))),
        ("max_posts_per_user", str(int(config.filters.max_posts_per_user))),
        ("allow_reels", str(bool(config.filters.allow_reels))),
        ("reject_if_sponsored_true", str(bool(config.filters.reject_if_sponsored_true))),
        ("targets_pool_n", str(int(config.targets.pool_n))),
        ("targets_final_n", str(int(config.targets.final_n))),
    ]

    count_rows: list[tuple[str, str]] = [
        ("raw_posts_total", str(int(data.counts.raw_posts))),
        ("decision_records_total", str(int(data.counts.decision_records))),
        ("labeled_posts_distinct", str(int(data.counts.labeled_posts))),
        ("eligible_total", str(int(data.counts.eligible_total))),
        ("rejected_total", str(int(data.counts.rejected_total))),
        ("eligible_in_pool_used", str(int(data.counts.eligible_in_pool))),
        ("final_sample_n", str(int(data.counts.final_sample_n))),
    ]

    story: list[Any] = []
    story.append(P("Instagram Corpus Codebook", h1))
    story.append(P("Public posts filtered for English, calisthenics/bodyweight relevance, and caption analyzability.", body))
    story.append(Spacer(1, 10))

    story.append(P("Run summary", h2))
    story.append(kv_table(run_rows))
    story.append(Spacer(1, 8))

    if run is not None and run.versions:
        versions_lines = "\n".join(f"{k}={v}" for k, v in sorted(run.versions.items()))
        story.append(P("Environment versions", h3))
        story.append(Paragraph(escape(versions_lines).replace("\n", "<br/>"), mono))
        story.append(Spacer(1, 8))

    story.append(P("Configuration snapshot", h2))
    story.append(kv_table(cfg_rows))
    story.append(Spacer(1, 8))

    story.append(P("Corpus counts", h2))
    story.append(kv_table(count_rows))
    story.append(Spacer(1, 10))

    story.append(P("Operational rules", h2))
    story.append(P("Include posts only if all inclusion criteria are satisfied.", body))
    story.append(P("Inclusion criteria", h3))
    story.append(bullets(list(data.inclusion_rules)))
    story.append(Spacer(1, 6))

    story.append(P("Common exclusions", h3))
    story.append(bullets(list(data.exclusion_rules)))
    story.append(Spacer(1, 10))

    story.append(P("Tag fields", h2))
    story.append(P("Structured fields are assigned per post using a JSON schema.", body))
    story.append(Spacer(1, 4))

    tag_rows = [(k, v) for k, v in sorted(data.tag_field_definitions.items())]
    story.append(kv_table(tag_rows))
    story.append(Spacer(1, 8))

    story.append(P("Genre values", h3))
    story.append(bullets([f"`{g}`" for g in data.genre_values]))
    story.append(PageBreak())

    story.append(P("Summary statistics (eligible pool)", h2))
    story.append(P("Counts are computed from the eligible pool used for export (capped by pool size).", body))

    story.extend(top_table("Top hashtags", data.stats.top_hashtags, limit=25))
    story.append(Spacer(1, 10))
    story.extend(top_table("Top genres", data.stats.top_genres, limit=20))
    story.append(Spacer(1, 10))
    story.extend(top_table("Top narrative labels", data.stats.top_narrative_labels, limit=25))
    story.append(Spacer(1, 10))
    story.extend(top_table("Top neoliberal signals", data.stats.top_neoliberal_signals, limit=25))
    story.append(Spacer(1, 10))

    story.append(P("Actor runs", h2))
    if not data.actor_runs:
        story.append(P("No actor runs were recorded for this run_id.", body_small))
    else:
        runs_rows = [["actor_id", "actor_run_id", "dataset_id", "created_at"]]
        for r in data.actor_runs[:200]:
            runs_rows.append([r.actor_id, r.actor_run_id, r.dataset_id, r.created_at])
        tbl = Table(
            runs_rows,
            colWidths=[2.0 * inch, 1.4 * inch, 1.6 * inch, 1.8 * inch],
            hAlign="LEFT",
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(tbl)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Corpus Codebook",
        author="ig_corpus",
    )

    try:
        doc.build(story)
    except Exception as e:
        raise ExportError(f"Failed to write codebook PDF: {out}: {e}") from e

    return out

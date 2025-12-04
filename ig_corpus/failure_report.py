from __future__ import annotations

from typing import Any, Mapping

from .config_schema import AppConfig


def build_failure_report(
    *,
    status: str,
    config: AppConfig,
    iterations: int,
    raw_posts: int,
    decisions: int,
    eligible: int,
    recent_new_eligible_total: int | None = None,
) -> dict[str, Any]:
    st = (status or "").strip() or "unknown"
    pool_target = int(config.targets.pool_n)
    final_target = int(config.targets.final_n)

    remaining = max(0, pool_target - int(eligible))
    progress_ratio = float(eligible) / float(pool_target) if pool_target > 0 else 0.0
    if progress_ratio < 0.0:
        progress_ratio = 0.0
    if progress_ratio > 1.0:
        progress_ratio = 1.0

    details: dict[str, Any] = {
        "iterations": int(iterations),
        "raw_posts": int(raw_posts),
        "decisions": int(decisions),
        "eligible": int(eligible),
        "pool_target": pool_target,
        "final_target": final_target,
        "eligible_remaining": int(remaining),
        "pool_progress_ratio": float(progress_ratio),
    }

    recommendations: list[str] = []
    summary = f"Run stopped with status={st}."

    if st == "max_raw_items":
        details["max_raw_items"] = int(config.loop.max_raw_items)
        summary = (
            "Reached the raw post cap before the eligible pool target "
            f"({raw_posts}/{int(config.loop.max_raw_items)} raw, {eligible}/{pool_target} eligible)."
        )
        recommendations = [
            "Increase loop.max_raw_items if you can store/process more posts.",
            "Increase apify.results_limit_per_query or apify.run_batch_queries to scrape more per cycle.",
            "Relax filters.min_caption_chars if captions are frequently filtered out.",
            "Add more querying.seed_terms or loosen querying.expansion thresholds to broaden discovery.",
        ]

    elif st == "max_iterations":
        details["max_iterations"] = int(config.loop.max_iterations)
        if recent_new_eligible_total is not None:
            details["recent_new_eligible_total"] = int(recent_new_eligible_total)
            details["stagnation_window"] = int(config.loop.stagnation_window)
            details["stagnation_min_new_eligible"] = int(config.loop.stagnation_min_new_eligible)

        summary = (
            "Reached the iteration cap before the eligible pool target "
            f"({iterations}/{int(config.loop.max_iterations)} iterations, {eligible}/{pool_target} eligible)."
        )
        recommendations = [
            "Increase loop.max_iterations to allow more scraping cycles.",
            "Increase apify.results_limit_per_query to fetch more posts per query term.",
            "Tune querying.expansion to enqueue more high-signal terms.",
            "Relax filters if too restrictive (min_caption_chars, max_posts_per_user).",
        ]

    elif st == "empty_query_queue":
        details["expansion_enabled"] = bool(config.querying.expansion.enabled)
        details["seed_terms_count"] = int(len(config.querying.seed_terms))
        summary = "No query terms were available to scrape."
        recommendations = [
            "Add more querying.seed_terms.",
            "Enable querying.expansion or increase max_new_terms_per_iter.",
        ]

    return {
        "status": st,
        "summary": summary,
        "details": details,
        "recommendations": recommendations,
    }


def format_failure_report(report: Mapping[str, Any]) -> str:
    status = str(report.get("status") or "").strip() or "unknown"
    summary = str(report.get("summary") or "").strip() or f"Run stopped ({status})."

    lines: list[str] = [summary]
    recs = report.get("recommendations")
    if isinstance(recs, list) and recs:
        lines.append("Recommendations:")
        for r in recs:
            t = str(r or "").strip()
            if t:
                lines.append(f"- {t}")

    return "\n".join(lines)

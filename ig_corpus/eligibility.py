from __future__ import annotations

from .llm_schema import LLMDecision


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    out = list(existing)
    seen = {r for r in out if isinstance(r, str)}
    for item in additions:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def compute_structured_eligibility(decision: LLMDecision) -> tuple[bool, tuple[str, ...]]:
    """
    Compute eligibility deterministically from structured fields.

    Accept only if ALL are true:
    - language.is_english == True
    - topic.is_bodyweight_calisthenics == True
    - caption_quality.is_analyzable == True
    - commercial.is_exclusively_commercial == False
    """
    failures: list[str] = []

    if not bool(decision.language.is_english):
        failures.append("language_not_english")

    if not bool(decision.topic.is_bodyweight_calisthenics):
        failures.append("topic_not_bodyweight_calisthenics")

    if not bool(decision.caption_quality.is_analyzable):
        failures.append("caption_not_analyzable")

    if bool(decision.commercial.is_exclusively_commercial):
        failures.append("exclusively_commercial")

    return (len(failures) == 0), tuple(failures)


def enforce_structured_eligibility(decision: LLMDecision) -> LLMDecision:
    """
    Ensure decision.eligible is consistent with the structured eligibility rules.

    If the model sets `eligible` inconsistently with required fields, override it and
    append machine-readable markers to eligibility_reasons.
    """
    computed_eligible, failures = compute_structured_eligibility(decision)
    if bool(decision.eligible) == bool(computed_eligible):
        return decision

    marker = "eligibility_overridden_accept" if computed_eligible else "eligibility_overridden_reject"
    additions: list[str] = [marker]
    if failures:
        additions.extend([f"eligibility_rule:{k}" for k in failures])

    return decision.model_copy(
        update={
            "eligible": bool(computed_eligible),
            "eligibility_reasons": _append_unique(list(decision.eligibility_reasons), additions),
        }
    )

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from openai import OpenAI

from .config_schema import OpenAIConfig
from .errors import LLMError
from .llm_schema import DECISION_JSON_SCHEMA, DECISION_SCHEMA_NAME, LLMDecision


class _ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


_SYSTEM_INSTRUCTIONS = """\
You label Instagram posts for an English-only research corpus on calisthenics/bodyweight training.

Use ONLY the provided fields (caption/hashtags/alt/type/isSponsored/etc.). Do not assume what is in the video/image.

Return a JSON object that matches the provided schema EXACTLY.

Guidelines:
- English-only: reject if mostly non-English or too mixed to analyze.
- Topic: accept only if clearly about calisthenics / street workout / bodyweight training, skills, progressions, rehab related to bodyweight work.
  Reject gym-only weightlifting/bodybuilding/crossfit/yoga/parkour/bouldering.
- Caption quality: reject if empty/emoji-only/hashtag-only or too fragmentary to analyze.
- Commercial: reject if the post is exclusively an ad (e.g., only promotion/codes/DM to buy) with no substantive training content.
- Provide concise eligibility_reasons explaining accept/reject.
- Fill tags:
  - genre: choose the best enum value
  - narrative_labels: 1–3 short labels (or empty if none)
  - discourse_moves: common moves present (or empty)
  - neoliberal_signals: only if present (or empty)

overall_confidence is 0–1.
"""


@dataclass(frozen=True)
class PostForLLM:
    url: str
    caption: str | None = None
    hashtags: Sequence[str] | None = None
    mentions: Sequence[str] | None = None
    alt: str | None = None
    type: str | None = None
    product_type: str | None = None
    is_sponsored: bool | None = None
    timestamp: str | None = None


def _build_user_message(post: PostForLLM) -> str:
    payload = {
        "url": post.url,
        "caption": post.caption,
        "hashtags": list(post.hashtags or []),
        "mentions": list(post.mentions or []),
        "alt": post.alt,
        "type": post.type,
        "productType": post.product_type,
        "isSponsored": post.is_sponsored,
        "timestamp": post.timestamp,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _extract_output_text(response: Any) -> str:
    direct = (getattr(response, "output_text", None) or "").strip()
    if direct:
        return direct

    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

    raise LLMError("OpenAI response did not include output text")


class OpenAIPostClassifier:
    """
    Minimal OpenAI wrapper for one-post-per-call labeling with Structured Outputs.
    """

    def __init__(
        self,
        api_key: str,
        *,
        openai_cfg: OpenAIConfig,
        client: _OpenAIClient | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError("api_key must be a non-empty string")

        self._cfg = openai_cfg
        self._client: _OpenAIClient = client or OpenAI(api_key=key)

    def classify(self, post: PostForLLM) -> LLMDecision:
        if not (post.url or "").strip():
            raise ValueError("post.url must be non-empty")

        try:
            response = self._client.responses.create(
                model=self._cfg.model_primary,
                instructions=_SYSTEM_INSTRUCTIONS,
                input=[
                    {"role": "user", "content": _build_user_message(post)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": DECISION_SCHEMA_NAME,
                        "strict": True,
                        "schema": DECISION_JSON_SCHEMA,
                    }
                },
                max_output_tokens=self._cfg.max_output_tokens,
            )
        except Exception as e:
            raise LLMError(f"OpenAI call failed: {e}") from e

        raw = _extract_output_text(response)
        try:
            return LLMDecision.model_validate_json(raw)
        except Exception as e:
            raise LLMError(f"Failed to parse structured output: {e}") from e

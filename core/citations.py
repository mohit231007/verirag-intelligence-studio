"""Citation parsing shared by generation validation and diagnostics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_BRACKETED = re.compile(r"\[([^\]\r\n]{1,80})\]")
_SOURCE_ID = re.compile(r"\b(?:S|Source)\s*(\d+)\b", re.IGNORECASE)
_VALID_GROUP = re.compile(
    r"\s*(?:S|Source)\s*\d+"
    r"(?:\s*(?:,|;|&|\band\b)\s*(?:S|Source)\s*\d+)*\s*",
    re.IGNORECASE,
)
_SOURCE_ID_ONLY = re.compile(r"(?:S|Source)\s*(\d+)", re.IGNORECASE)


class StructuredAnswerError(ValueError):
    """The model did not return a safely renderable claim-to-source mapping."""


@dataclass(frozen=True, slots=True)
class StructuredAnswer:
    can_answer: bool
    answer: str = ""
    reason: str = ""


def _normalized(text: str) -> str:
    return text.translate(str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"}))


def extract_citation_ids(text: str) -> list[str]:
    """Return source numbers from bounded, bracketed citation groups.

    Accepted examples include ``[S1]``, ``[S1, S2]``, ``[Source 3]``, and
    ``【S4】``. Bare source-like text is deliberately ignored.
    """

    identifiers: list[str] = []
    for match in _BRACKETED.finditer(_normalized(text)):
        group = match.group(1)
        if not _VALID_GROUP.fullmatch(group):
            continue
        identifiers.extend(_SOURCE_ID.findall(group))
    return identifiers


def canonicalize_citations(text: str) -> str:
    """Normalize accepted citation variants to the displayed ``[S#]`` form."""

    normalized = _normalized(text)

    def replace(match: re.Match[str]) -> str:
        group = match.group(1)
        if not _VALID_GROUP.fullmatch(group):
            return match.group(0)
        return " ".join(f"[S{identifier}]" for identifier in _SOURCE_ID.findall(group))

    return _BRACKETED.sub(replace, normalized)


def parse_structured_answer(text: str, valid_ids: set[str]) -> StructuredAnswer:
    """Parse model JSON and deterministically render cited Markdown bullets."""

    candidate = text.strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[7:-3].strip()
    elif candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise StructuredAnswerError("No JSON object found") from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StructuredAnswerError("Invalid JSON") from exc
    if not isinstance(payload, dict):
        raise StructuredAnswerError("The JSON response must be an object")

    can_answer = payload.get("can_answer")
    if not isinstance(can_answer, bool):
        raise StructuredAnswerError("can_answer must be a boolean")
    if not can_answer:
        reason = " ".join(str(payload.get("reason", "Evidence is insufficient.")).split())
        return StructuredAnswer(False, reason=reason[:500])

    items = payload.get("items")
    if not isinstance(items, list) or not items or len(items) > 20:
        raise StructuredAnswerError("items must contain between 1 and 20 claims")

    rendered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise StructuredAnswerError("Each item must be an object")
        raw_claim = item.get("claim")
        if not isinstance(raw_claim, str):
            raise StructuredAnswerError("Each item needs a text claim")
        claim = " ".join(raw_claim.split()).lstrip("-*+ ")
        source_ids = item.get("source_ids")
        if not claim or not isinstance(source_ids, list) or not source_ids:
            raise StructuredAnswerError("Each item needs a claim and source IDs")

        normalized_ids: list[str] = []
        for source_id in source_ids:
            if not isinstance(source_id, str):
                raise StructuredAnswerError("Source IDs must be strings")
            match = _SOURCE_ID_ONLY.fullmatch(source_id.strip())
            if not match or match.group(1) not in valid_ids:
                raise StructuredAnswerError("An item contains an unknown source ID")
            if match.group(1) not in normalized_ids:
                normalized_ids.append(match.group(1))
        citations = " ".join(f"[S{identifier}]" for identifier in normalized_ids)
        punctuation = claim[-1] if claim.endswith((".", "!", "?")) else ""
        claim_body = claim[:-1] if punctuation else claim
        rendered.append(f"- {claim_body} {citations}{punctuation}")

    return StructuredAnswer(True, answer="\n".join(rendered))

"""Citation parsing shared by generation validation and diagnostics."""

from __future__ import annotations

import re

_BRACKETED = re.compile(r"\[([^\]\r\n]{1,80})\]")
_SOURCE_ID = re.compile(r"\b(?:S|Source)\s*(\d+)\b", re.IGNORECASE)
_VALID_GROUP = re.compile(
    r"\s*(?:S|Source)\s*\d+"
    r"(?:\s*(?:,|;|&|\band\b)\s*(?:S|Source)\s*\d+)*\s*",
    re.IGNORECASE,
)


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

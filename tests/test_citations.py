import pytest

from core.citations import (
    StructuredAnswerError,
    canonicalize_citations,
    extract_citation_ids,
    parse_structured_answer,
)


def test_extracts_bounded_citation_variants() -> None:
    assert extract_citation_ids("Claim [S1, S2] and claim 【Source 3】.") == ["1", "2", "3"]


def test_ignores_bare_or_unbounded_source_text() -> None:
    assert extract_citation_ids("See S1 and [the source is S2].") == []


def test_canonicalizes_accepted_variants() -> None:
    assert canonicalize_citations("Claim [Source 1, S2].") == "Claim [S1] [S2]."


def test_structured_answer_renders_citations_deterministically() -> None:
    raw = '{"can_answer":true,"reason":"","items":[{"claim":"Ten years experience","source_ids":["S1","S2"]}]}'
    result = parse_structured_answer(raw, {"1", "2"})
    assert result.can_answer
    assert result.answer == "- Ten years experience [S1] [S2]"


def test_structured_answer_can_report_insufficient_evidence() -> None:
    raw = '{"can_answer":false,"reason":"The CV is unrelated.","items":[]}'
    result = parse_structured_answer(raw, {"1"})
    assert not result.can_answer
    assert result.reason == "The CV is unrelated."


def test_structured_answer_rejects_unknown_source() -> None:
    raw = '{"can_answer":true,"items":[{"claim":"Claim","source_ids":["S9"]}]}'
    with pytest.raises(StructuredAnswerError, match="unknown"):
        parse_structured_answer(raw, {"1"})


def test_structured_answer_rejects_non_object_json() -> None:
    with pytest.raises(StructuredAnswerError, match="object"):
        parse_structured_answer('[{"can_answer":true}]', {"1"})

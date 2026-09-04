from core.citations import canonicalize_citations, extract_citation_ids


def test_extracts_bounded_citation_variants() -> None:
    assert extract_citation_ids("Claim [S1, S2] and claim 【Source 3】.") == ["1", "2", "3"]


def test_ignores_bare_or_unbounded_source_text() -> None:
    assert extract_citation_ids("See S1 and [the source is S2].") == []


def test_canonicalizes_accepted_variants() -> None:
    assert canonicalize_citations("Claim [Source 1, S2].") == "Claim [S1] [S2]."

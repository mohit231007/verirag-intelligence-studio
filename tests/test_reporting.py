from io import BytesIO

from pypdf import PdfReader

from core.models import DocumentChunk, QueryTrace, RetrievedChunk
from core.reporting import build_answer_pdf


def test_answer_pdf_contains_question_answer_and_evidence() -> None:
    chunk = DocumentChunk("chunk-1", "hash", "policy.txt", 3, "The window starts in October.", 0, 29)
    trace = QueryTrace(
        query="When does it start?",
        standalone_query="campaign start date",
        answer="It starts in October [S1].",
        retrieved=[RetrievedChunk(chunk, 0.84, 1)],
        confidence="High",
        total_ms=250.0,
        provider="fake",
        model="test",
    )
    output = build_answer_pdf(trace)
    reader = PdfReader(BytesIO(output))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) == 2
    assert "When does it start?" in text
    assert "The window starts in October" in text


def test_answer_pdf_normalizes_unsupported_hyphens_and_wraps_long_query() -> None:
    trace = QueryTrace(
        query="Summarize data‑science experience and measurable achievements.",
        standalone_query="Summarize " + "applied data science experience " * 15,
        answer="- Built an end‑to‑end machine‑learning product [S1].",
        provider="groq",
        model="test",
    )
    output = build_answer_pdf(trace)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(output)).pages)
    assert "data-science" in text
    assert "end-to-end machine-learning" in text
    assert "■" not in text

from core.ingestion import chunk_page, ingest_document, normalize_text
from core.models import SourcePage


def test_normalize_text_preserves_paragraph_boundaries() -> None:
    assert normalize_text("A  line.\n\n\n B\tline.") == "A line.\n\nB line."


def test_chunks_are_deterministic_and_bounded() -> None:
    text = "First sentence has useful evidence. Second sentence adds detail. " * 8
    page = SourcePage("policy.txt", 1, text)
    first = chunk_page(page, "a" * 64, chunk_size=120, overlap=20)
    second = chunk_page(page, "a" * 64, chunk_size=120, overlap=20)
    assert first == second
    assert len(first) > 1
    assert all(len(chunk.text) <= 120 for chunk in first)
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_text_ingestion_uses_content_hash_for_identity() -> None:
    result = ingest_document(
        b"Policy statement.\n\nThe campaign starts in October.",
        "policy.txt",
        max_file_bytes=1_000,
        max_pages=10,
        chunk_size=400,
        chunk_overlap=20,
    )
    assert result.pages == 1
    assert len(result.chunks) == 1
    assert result.chunks[0].document_hash == result.document_hash

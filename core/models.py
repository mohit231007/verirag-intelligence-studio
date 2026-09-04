"""Domain models shared across ingestion, retrieval, generation, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourcePage:
    source_doc: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_hash: str
    source_doc: str
    page_number: int
    text: str
    char_start: int
    char_end: int

    def metadata(self) -> dict[str, str | int]:
        data = asdict(self)
        data.pop("text")
        return data


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    similarity: float
    rank: int


@dataclass(slots=True)
class QueryTrace:
    query: str
    standalone_query: str
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    is_refusal: bool = False
    refusal_reason: str | None = None
    confidence: str = "Low"
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    provider: str = ""
    model: str = ""
    invalid_citations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_hash: str
    filename: str
    pages: int
    chunks: tuple[DocumentChunk, ...]
    warnings: tuple[str, ...] = ()

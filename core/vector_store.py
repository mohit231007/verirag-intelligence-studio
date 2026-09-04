"""Session-isolated Chroma storage with a shared, cached embedding model."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Protocol

from .models import DocumentChunk, RetrievedChunk


class EmbeddingModel(Protocol):
    def embed(self, documents: Iterable[str]) -> Iterable[Any]: ...

    def query_embed(self, query: str) -> Iterable[Any]: ...


def safe_collection_name(session_id: str) -> str:
    """Produce a valid, non-identifying Chroma collection name."""

    compact = re.sub(r"[^a-zA-Z0-9]", "", session_id)[:40]
    return f"verirag_{compact or 'session'}"


class VectorStoreManager:
    """A collection is unique per browser session to prevent cross-user data leakage."""

    def __init__(self, client: Any, embedding_model: EmbeddingModel, session_id: str):
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = safe_collection_name(session_id)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def contains_document(self, document_hash: str) -> bool:
        result = self.collection.get(where={"document_hash": document_hash}, limit=1)
        return bool(result.get("ids"))

    def add_chunks(self, chunks: Iterable[DocumentChunk], batch_size: int = 64) -> int:
        items = list(chunks)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = [vector.tolist() for vector in self.embedding_model.embed(texts)]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=embeddings,
            )
        return len(items)

    def query(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        available = self.count()
        if available == 0:
            return []
        query_vector = next(iter(self.embedding_model.query_embed(query_text))).tolist()
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for rank, (text, metadata, distance) in enumerate(
            zip(documents, metadatas, distances, strict=False),
            start=1,
        ):
            chunk = DocumentChunk(
                chunk_id=str(metadata["chunk_id"]),
                document_hash=str(metadata["document_hash"]),
                source_doc=str(metadata["source_doc"]),
                page_number=int(metadata["page_number"]),
                text=text,
                char_start=int(metadata["char_start"]),
                char_end=int(metadata["char_end"]),
            )
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(RetrievedChunk(chunk, round(similarity, 4), rank))
        return retrieved

    def count(self) -> int:
        return int(self.collection.count())

    def document_names(self) -> list[str]:
        result = self.collection.get(include=["metadatas"])
        names = {str(meta["source_doc"]) for meta in result.get("metadatas", [])}
        return sorted(names, key=str.casefold)

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

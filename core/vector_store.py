"""Session-isolated Chroma storage with dense, lexical and hybrid retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Protocol

from .models import DocumentChunk, RetrievedChunk
from .retrieval import bm25_scores, rrf_score, transparent_rerank_score


class EmbeddingModel(Protocol):
    def embed(self, documents: Iterable[str]) -> Iterable[Any]: ...

    def query_embed(self, query: str) -> Iterable[Any]: ...


def safe_collection_name(session_id: str) -> str:
    """Produce a valid, non-identifying Chroma collection name."""

    compact = re.sub(r"[^a-zA-Z0-9]", "", session_id)[:40]
    return f"verirag_{compact or 'session'}"


def _chunk_from_record(text: str, metadata: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(metadata["chunk_id"]),
        document_hash=str(metadata["document_hash"]),
        source_doc=str(metadata["source_doc"]),
        page_number=int(metadata["page_number"]),
        text=text,
        char_start=int(metadata["char_start"]),
        char_end=int(metadata["char_end"]),
    )


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

    def all_chunks(self) -> list[DocumentChunk]:
        if self.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            _chunk_from_record(text or "", metadata or {})
            for text, metadata in zip(documents, metadatas, strict=False)
            if text and metadata
        ]

    def query_dense(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
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
            chunk = _chunk_from_record(text, metadata)
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(
                RetrievedChunk(
                    chunk,
                    round(similarity, 4),
                    rank,
                    dense_score=round(similarity, 4),
                    rerank_score=round(similarity, 4),
                    retrieval_method="dense",
                )
            )
        return retrieved

    def query_lexical(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        chunks = self.all_chunks()
        if not chunks:
            return []
        scores = bm25_scores(query_text, [chunk.text for chunk in chunks])
        ordered = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, min(top_k, len(chunks)))]
        return [
            RetrievedChunk(
                chunk,
                round(score, 4),
                rank,
                lexical_score=round(score, 4),
                rerank_score=round(score, 4),
                retrieval_method="lexical",
            )
            for rank, (chunk, score) in enumerate(ordered, start=1)
        ]

    def query_hybrid(
        self,
        query_text: str,
        top_k: int,
        *,
        dense_weight: float = 0.60,
        rerank_weight: float = 0.20,
        rrf_k: int = 60,
        lexical_candidate_limit: int = 40,
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        candidate_limit = min(
            self.count(),
            max(top_k * 3, top_k, lexical_candidate_limit),
        )
        dense = self.query_dense(query_text, candidate_limit)
        lexical = self.query_lexical(query_text, candidate_limit)
        dense_map = {item.chunk.chunk_id: item for item in dense}
        lexical_map = {item.chunk.chunk_id: item for item in lexical}
        dense_rank = {item.chunk.chunk_id: item.rank for item in dense}
        lexical_rank = {item.chunk.chunk_id: item.rank for item in lexical}

        merged: list[RetrievedChunk] = []
        for chunk_id in dense_map.keys() | lexical_map.keys():
            dense_item = dense_map.get(chunk_id)
            lexical_item = lexical_map.get(chunk_id)
            item = dense_item or lexical_item
            if item is None:
                continue
            dense_score = float(dense_item.dense_score or 0.0) if dense_item else 0.0
            lexical_score = (
                float(lexical_item.lexical_score or 0.0) if lexical_item else 0.0
            )
            fused_rrf = rrf_score(
                dense_rank.get(chunk_id), lexical_rank.get(chunk_id), k=rrf_k
            )
            final = transparent_rerank_score(
                query_text,
                item.chunk.text,
                dense_score=dense_score,
                lexical_score=lexical_score,
                rrf=fused_rrf,
                dense_weight=dense_weight,
                rerank_weight=rerank_weight,
            )
            merged.append(
                RetrievedChunk(
                    item.chunk,
                    final,
                    0,
                    dense_score=round(dense_score, 4),
                    lexical_score=round(lexical_score, 4),
                    rerank_score=final,
                    retrieval_method="hybrid",
                )
            )

        merged.sort(key=lambda item: item.similarity, reverse=True)
        return [
            RetrievedChunk(
                item.chunk,
                item.similarity,
                rank,
                dense_score=item.dense_score,
                lexical_score=item.lexical_score,
                rerank_score=item.rerank_score,
                retrieval_method=item.retrieval_method,
            )
            for rank, item in enumerate(merged[:top_k], start=1)
        ]

    def query(
        self,
        query_text: str,
        top_k: int,
        *,
        mode: str = "dense",
        dense_weight: float = 0.60,
        rerank_weight: float = 0.20,
        rrf_k: int = 60,
        lexical_candidate_limit: int = 40,
    ) -> list[RetrievedChunk]:
        if mode == "dense":
            return self.query_dense(query_text, top_k)
        if mode == "lexical":
            return self.query_lexical(query_text, top_k)
        if mode == "hybrid":
            return self.query_hybrid(
                query_text,
                top_k,
                dense_weight=dense_weight,
                rerank_weight=rerank_weight,
                rrf_k=rrf_k,
                lexical_candidate_limit=lexical_candidate_limit,
            )
        raise ValueError(f"Unsupported retrieval mode: {mode}")

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

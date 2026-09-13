from core.models import DocumentChunk
from core.retrieval import bm25_scores, query_term_coverage, rrf_score, transparent_rerank_score
from core.vector_store import VectorStoreManager


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeEmbeddings:
    def embed(self, documents):
        vectors = []
        for document in documents:
            text = document.lower()
            vectors.append(Vector([1.0 if "refund" in text else 0.2, 1.0 if "hotel" in text else 0.1]))
        return vectors

    def query_embed(self, query):
        text = query.lower()
        return iter([Vector([1.0 if "refund" in text else 0.2, 1.0 if "hotel" in text else 0.1])])


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, documents, metadatas, embeddings):
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=True):
            self.rows[item_id] = (document, metadata, embedding)

    def get(self, where=None, limit=None, include=None):
        rows = list(self.rows.items())
        if where:
            rows = [(key, row) for key, row in rows if all(row[1].get(k) == v for k, v in where.items())]
        if limit:
            rows = rows[:limit]
        return {
            "ids": [key for key, _ in rows],
            "documents": [row[0] for _, row in rows],
            "metadatas": [row[1] for _, row in rows],
        }

    def query(self, query_embeddings, n_results, include):
        rows = list(self.rows.values())[:n_results]
        distances = []
        for _, _, embedding in rows:
            # Keep deterministic but not identical dense scores.
            distances.append(0.05 if embedding[0] >= 1.0 else 0.6)
        return {
            "documents": [[row[0] for row in rows]],
            "metadatas": [[row[1] for row in rows]],
            "distances": [distances],
        }

    def count(self):
        return len(self.rows)


class FakeClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, metadata):
        return self.collections.setdefault(name, FakeCollection())

    def delete_collection(self, name):
        self.collections.pop(name, None)


def _chunk(chunk_id, text):
    return DocumentChunk(chunk_id, "hash", f"{chunk_id}.txt", 1, text, 0, len(text))


def test_bm25_prioritizes_exact_lexical_evidence() -> None:
    docs = [
        "Refunds above USD 2000 require manager approval.",
        "Hotel reimbursement is capped at USD 220 per night.",
    ]
    scores = bm25_scores("refund manager approval", docs)
    assert scores[0] > scores[1]
    assert scores[0] == 1.0


def test_transparent_reranker_rewards_query_coverage() -> None:
    strong = transparent_rerank_score(
        "refund manager approval",
        "Refund requests require manager approval.",
        dense_score=0.7,
        lexical_score=0.8,
        rrf=rrf_score(1, 1),
    )
    weak = transparent_rerank_score(
        "refund manager approval",
        "Hotel rates are capped.",
        dense_score=0.7,
        lexical_score=0.1,
        rrf=rrf_score(1, 8),
    )
    assert strong > weak
    assert query_term_coverage("refund manager approval", "refund manager approval") == 1.0


def test_vector_store_supports_dense_lexical_and_hybrid_modes() -> None:
    store = VectorStoreManager(FakeClient(), FakeEmbeddings(), "hybrid-test")
    refund = _chunk("refund", "Refunds above USD 2000 require Store Operations Manager approval.")
    hotel = _chunk("hotel", "Domestic hotel reimbursement is capped at USD 220 per night.")
    store.add_chunks([refund, hotel])

    dense = store.query("refund manager approval", 2, mode="dense")
    lexical = store.query("refund manager approval", 2, mode="lexical")
    hybrid = store.query("refund manager approval", 2, mode="hybrid")

    assert dense[0].retrieval_method == "dense"
    assert lexical[0].chunk.chunk_id == "refund"
    assert lexical[0].retrieval_method == "lexical"
    assert hybrid[0].chunk.chunk_id == "refund"
    assert hybrid[0].retrieval_method == "hybrid"
    assert hybrid[0].dense_score is not None
    assert hybrid[0].lexical_score is not None
    assert hybrid[0].rerank_score is not None

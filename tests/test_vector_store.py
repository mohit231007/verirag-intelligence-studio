from core.models import DocumentChunk
from core.vector_store import VectorStoreManager, safe_collection_name


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeEmbeddings:
    def embed(self, documents):
        return [Vector([1.0, 0.0]) for _ in documents]

    def query_embed(self, query):
        return iter([Vector([1.0, 0.0])])


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
        return {"ids": [key for key, _ in rows], "metadatas": [row[1] for _, row in rows]}

    def query(self, query_embeddings, n_results, include):
        rows = list(self.rows.values())[:n_results]
        return {
            "documents": [[row[0] for row in rows]],
            "metadatas": [[row[1] for row in rows]],
            "distances": [[0.1 for _ in rows]],
        }

    def count(self):
        return len(self.rows)


class FakeClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, metadata):
        return self.collections.setdefault(name, FakeCollection())

    def delete_collection(self, name):
        self.collections.pop(name)


def test_collection_name_is_sanitized() -> None:
    assert safe_collection_name("abc-123! private") == "verirag_abc123private"


def test_add_query_list_and_clear_round_trip() -> None:
    store = VectorStoreManager(FakeClient(), FakeEmbeddings(), "session-1")
    chunk = DocumentChunk("id-1", "hash-1", "Policy.txt", 2, "Evidence", 0, 8)
    assert store.add_chunks([chunk]) == 1
    assert store.contains_document("hash-1")
    assert store.count() == 1
    assert store.document_names() == ["Policy.txt"]
    result = store.query("evidence", 4)
    assert result[0].chunk == chunk
    assert result[0].similarity == 0.9
    store.clear()
    assert store.count() == 0

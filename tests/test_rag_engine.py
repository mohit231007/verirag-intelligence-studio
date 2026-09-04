from core.config import AppConfig
from core.models import DocumentChunk, RetrievedChunk
from core.rag_engine import CITATION_FAILURE, INSUFFICIENT_EVIDENCE, RAGEngine


class FakeStore:
    def __init__(self, retrieved):
        self.retrieved = retrieved

    def count(self):
        return len(self.retrieved)

    def query(self, query_text, top_k):
        return self.retrieved[:top_k]


class FakeProvider:
    name = "fake"
    model = "deterministic"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def complete(self, system_prompt, user_prompt, *, temperature=0.0):
        self.calls += 1
        return self.answer


def evidence(score=0.8):
    chunk = DocumentChunk("hash_p1_c1", "hash", "policy.txt", 1, "Campaign starts in October.", 0, 27)
    return RetrievedChunk(chunk, score, 1)


def test_low_similarity_refuses_without_generation() -> None:
    provider = FakeProvider("should not be called")
    engine = RAGEngine(FakeStore([evidence(0.2)]), provider, AppConfig())
    trace = engine.execute("When does it start?")
    assert trace.answer == INSUFFICIENT_EVIDENCE
    assert trace.is_refusal
    assert provider.calls == 0


def test_valid_source_id_allows_answer() -> None:
    provider = FakeProvider("The campaign starts in October [S1].")
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("When does it start?")
    assert not trace.is_refusal
    assert trace.confidence == "High"
    assert provider.calls == 1


def test_missing_or_invented_citation_fails_closed() -> None:
    for answer in ("It starts in October.", "It starts in October [S9]."):
        provider = FakeProvider(answer)
        engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
        trace = engine.execute("When does it start?")
        assert trace.answer == CITATION_FAILURE
        assert trace.is_refusal

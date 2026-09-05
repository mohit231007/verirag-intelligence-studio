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

    def __init__(self, answer, repair_answer=None):
        self.answer = answer
        self.repair_answer = repair_answer if repair_answer is not None else answer
        self.calls = 0

    def complete(self, system_prompt, user_prompt, *, temperature=0.0):
        self.calls += 1
        return self.answer if self.calls == 1 else self.repair_answer


def evidence(score=0.8):
    chunk = DocumentChunk("hash_p1_c1", "hash", "policy.txt", 1, "Campaign starts in October.", 0, 27)
    return RetrievedChunk(chunk, score, 1)


def second_evidence(score=0.45):
    chunk = DocumentChunk("hash_p2_c1", "hash", "policy.txt", 2, "Campaign ends in November.", 0, 26)
    return RetrievedChunk(chunk, score, 2)


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
        assert trace.citation_repair_attempted


def test_citation_variants_are_canonicalized() -> None:
    provider = FakeProvider("The campaign starts in October [Source 1].")
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("When does it start?")
    assert trace.answer == "The campaign starts in October [S1]."
    assert not trace.is_refusal


def test_invalid_first_draft_is_repaired_once() -> None:
    provider = FakeProvider(
        "The campaign starts in October.",
        '{"can_answer":true,"reason":"","items":'
        '[{"claim":"The campaign starts in October.","source_ids":["S1"]}]}',
    )
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("When does it start?")
    assert trace.answer == "- The campaign starts in October [S1]."
    assert trace.citation_repair_attempted
    assert provider.calls == 2


def test_partially_cited_draft_is_repaired_once() -> None:
    provider = FakeProvider(
        "It starts in October. It ends in November [S1].",
        '{"can_answer":true,"reason":"","items":'
        '[{"claim":"It starts in October and ends in November.","source_ids":["S1"]}]}',
    )
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("What is the campaign window?")
    assert not trace.is_refusal
    assert trace.metrics["citation_coverage"] == 1.0
    assert provider.calls == 2


def test_repair_can_classify_semantically_insufficient_evidence() -> None:
    provider = FakeProvider(
        "The evidence does not describe data-science experience.",
        '{"can_answer":false,"reason":"The supplied CV is about power operations.","items":[]}',
    )
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("Summarize data-science experience")
    assert trace.answer == INSUFFICIENT_EVIDENCE
    assert trace.refusal_reason == "model_insufficient_evidence"
    assert trace.citation_validation_error is None


def test_first_generation_can_return_structured_answer_without_repair() -> None:
    provider = FakeProvider(
        '{"can_answer":true,"reason":"","items":'
        '[{"claim":"The campaign starts in October.","source_ids":["S1"]}]}'
    )
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("When does it start?")
    assert trace.answer == "- The campaign starts in October [S1]."
    assert not trace.citation_repair_attempted
    assert provider.calls == 1


def test_first_generation_can_classify_insufficient_evidence() -> None:
    provider = FakeProvider(
        '{"can_answer":false,"reason":"The evidence is unrelated.","items":[]}'
    )
    engine = RAGEngine(FakeStore([evidence()]), provider, AppConfig())
    trace = engine.execute("Summarize data-science experience")
    assert trace.refusal_reason == "model_insufficient_evidence"
    assert not trace.citation_repair_attempted
    assert provider.calls == 1


def test_configured_similarity_gate_is_the_only_evidence_floor() -> None:
    provider = FakeProvider("It starts in October [S1] and ends in November [S2].")
    config = AppConfig(top_k=2, similarity_threshold=0.4)
    engine = RAGEngine(FakeStore([evidence(), second_evidence()]), provider, config)
    trace = engine.execute("What is the campaign window?")
    assert len(trace.retrieved) == 2
    assert not trace.is_refusal

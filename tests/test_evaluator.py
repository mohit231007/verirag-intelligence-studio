import pytest

from core.evaluator import answer_relevance, citation_coverage, deterministic_metrics, judge_faithfulness
from core.models import QueryTrace
from core.providers import ProviderError


def test_citation_coverage_counts_claims() -> None:
    answer = "The window starts in October [S1]. It ends in November [S2]."
    assert citation_coverage(answer) == 1.0
    assert citation_coverage("The window starts in October. It ends in November [S2].") == 0.5
    assert citation_coverage("## Summary\n- It starts in October [Source 1].") == 1.0
    assert citation_coverage("**Applied Data Science Experience**\n- It starts [S1].") == 1.0


def test_answer_relevance_is_explicitly_lexical() -> None:
    assert answer_relevance("frozen food window", "The frozen food campaign window is October.") == 1.0


def test_refusal_without_chunks_has_perfect_context_precision_proxy() -> None:
    trace = QueryTrace("pet policy", "pet policy", "No evidence", is_refusal=True)
    assert deterministic_metrics(trace, 0.4)["context_precision_proxy"] == 1.0


def test_metrics_audit_rejected_generated_answer() -> None:
    trace = QueryTrace(
        "campaign start",
        "campaign start",
        "Citation validation failed",
        is_refusal=True,
        refusal_reason="citation_validation",
        generated_answer="The campaign starts in October.",
    )
    metrics = deterministic_metrics(trace, 0.4)
    assert metrics["citation_coverage"] == 0.0
    assert metrics["answer_relevance_proxy"] == 0.5


class JudgeProvider:
    name = "fake"
    model = "judge"

    def __init__(self, response):
        self.response = response

    def complete(self, system_prompt, user_prompt, *, temperature=0.0):
        return self.response


def test_judge_parses_bounded_json_result() -> None:
    trace = QueryTrace("Question", "Question", "Answer")
    result = judge_faithfulness(
        trace,
        JudgeProvider('{"score": 1.5, "unsupported_claims": ["claim"], "reasoning": "checked"}'),
    )
    assert result.score == 1.0
    assert result.unsupported_claims == ("claim",)


def test_judge_rejects_non_json_response() -> None:
    with pytest.raises(ProviderError, match="invalid JSON"):
        judge_faithfulness(QueryTrace("Q", "Q", "A"), JudgeProvider("not json"))


def test_judge_marks_evidence_refusal_not_applicable() -> None:
    trace = QueryTrace("Q", "Q", "No evidence", is_refusal=True, refusal_reason="similarity_threshold")
    result = judge_faithfulness(trace, JudgeProvider("not used"))
    assert result.score is None

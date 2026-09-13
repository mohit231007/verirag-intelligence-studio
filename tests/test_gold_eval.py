from core.gold_eval import (
    GoldDataError,
    GoldExample,
    brier_score,
    citation_gold_metrics,
    compare_runs,
    evaluate_run,
    expected_calibration_error,
    parse_gold_jsonl,
    retrieval_metrics,
    token_f1,
)
from core.models import DocumentChunk, QueryTrace, RetrievedChunk


def _retrieved(chunk_id: str, source: str, similarity: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        DocumentChunk(
            chunk_id=chunk_id,
            document_hash="hash",
            source_doc=source,
            page_number=1,
            text="Gold evidence text",
            char_start=0,
            char_end=18,
        ),
        similarity,
        rank,
    )


def test_parse_gold_jsonl_supports_conversation_labels() -> None:
    rows = """{"case_id":"c1","query":"What restrictions apply to it?","expected_answerable":true,"expected_source_docs":["policy.txt"],"expected_standalone_query":"What restrictions apply to the incentive?","history":[{"role":"user","content":"What is the incentive?"}]}
{"case_id":"c2","query":"What is the CEO name?","expected_answerable":false}
"""
    examples = parse_gold_jsonl(rows)
    assert len(examples) == 2
    assert examples[0].history[0]["role"] == "user"
    assert examples[1].expected_answerable is False


def test_parse_gold_jsonl_rejects_answerable_without_any_gold_target() -> None:
    try:
        parse_gold_jsonl(
            '{"case_id":"c1","query":"Q","expected_answerable":true}'
        )
    except GoldDataError as exc:
        assert "expected_answer or expected evidence" in str(exc)
    else:
        raise AssertionError("expected GoldDataError")


def test_retrieval_and_citation_metrics_use_gold_evidence() -> None:
    example = GoldExample(
        case_id="c1",
        query="Q",
        expected_answerable=True,
        expected_chunk_ids=("gold",),
    )
    trace = QueryTrace(
        "Q",
        "Q",
        "- Correct answer [S2].",
        retrieved=[
            _retrieved("distractor", "other.txt", 0.9, 1),
            _retrieved("gold", "policy.txt", 0.8, 2),
        ],
        generated_answer="- Correct answer [S2].",
    )
    recall, reciprocal_rank = retrieval_metrics(example, trace)
    precision, citation_recall, citation_f1 = citation_gold_metrics(example, trace)
    assert recall == 1.0
    assert reciprocal_rank == 0.5
    assert precision == 1.0
    assert citation_recall == 1.0
    assert citation_f1 == 1.0


def test_token_f1_is_deterministic_not_semantic() -> None:
    assert token_f1("campaign starts 15 October", "campaign starts 15 October") == 1.0
    assert token_f1("red", "blue") == 0.0


def test_calibration_metrics() -> None:
    confidences = [0.9, 0.8, 0.2, 0.1]
    outcomes = [True, True, False, False]
    assert brier_score(confidences, outcomes) == 0.025
    assert expected_calibration_error(confidences, outcomes, bins=2) == 0.15


def test_evaluate_run_reports_confusion_failures_and_regression() -> None:
    examples = [
        GoldExample(
            case_id="a",
            query="Answerable",
            expected_answerable=True,
            expected_answer="The answer is 42.",
            expected_source_docs=("policy.txt",),
        ),
        GoldExample(
            case_id="b",
            query="Unsupported",
            expected_answerable=False,
        ),
    ]
    good = [
        QueryTrace(
            "Answerable",
            "Answerable",
            "The answer is 42. [S1]",
            retrieved=[_retrieved("gold", "policy.txt", 0.9, 1)],
            generated_answer="The answer is 42. [S1]",
            confidence="High",
            confidence_score=0.9,
        ),
        QueryTrace(
            "Unsupported",
            "Unsupported",
            "No evidence",
            is_refusal=True,
            refusal_reason="similarity_threshold",
        ),
    ]
    baseline = evaluate_run(examples, good, variant="baseline", answer_f1_threshold=0.5)
    assert baseline.confusion == {"tp": 1, "tn": 1, "fp": 0, "fn": 0}
    assert baseline.metrics["answerability_accuracy"] == 1.0

    bad = [
        QueryTrace(
            "Answerable",
            "Answerable",
            "No evidence",
            is_refusal=True,
            refusal_reason="similarity_threshold",
        ),
        QueryTrace(
            "Unsupported",
            "Unsupported",
            "Fabricated answer [S1].",
            retrieved=[_retrieved("wrong", "other.txt", 0.8, 1)],
            generated_answer="Fabricated answer [S1].",
            confidence="High",
            confidence_score=0.9,
        ),
    ]
    candidate = evaluate_run(examples, bad, variant="candidate")
    assert candidate.failure_counts["WRONG_REFUSAL"] == 1
    assert candidate.failure_counts["WRONG_ANSWERABILITY"] == 1
    comparison = compare_runs(baseline, candidate)
    accuracy = next(row for row in comparison if row["metric"] == "overall_accuracy")
    assert accuracy["regression"] is True

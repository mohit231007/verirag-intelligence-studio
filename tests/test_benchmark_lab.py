from core.benchmark_lab import red_blue_summary, regression_gate, threshold_sweep
from core.builtin_benchmark import BUILTIN_CASE_COUNT, builtin_gold_jsonl
from core.config import AppConfig
from core.gold_eval import BenchmarkRun, CaseEvaluation, GoldExample, parse_gold_jsonl
from core.models import DocumentChunk, RetrievedChunk
from core.rag_engine import RAGEngine


class FakeProvider:
    name = "fake"
    model = "fake"

    def complete(self, system_prompt, user_prompt, *, temperature=0.0):
        return ""


class FakeStore:
    def __init__(self):
        self.mapping = {
            "known policy": [
                RetrievedChunk(
                    DocumentChunk("c1", "h1", "policy.txt", 1, "Known policy evidence.", 0, 22),
                    0.85,
                    1,
                )
            ],
            "unknown question": [
                RetrievedChunk(
                    DocumentChunk("c2", "h2", "other.txt", 1, "Weak unrelated evidence.", 0, 22),
                    0.20,
                    1,
                )
            ],
        }

    def count(self):
        return 2

    def query(self, query_text, top_k, **kwargs):
        return self.mapping.get(query_text, [])[:top_k]


def _case(case_id, expected, predicted, correct, tags=()):
    return CaseEvaluation(
        case_id=case_id,
        query=case_id,
        expected_answerable=expected,
        predicted_answerable=predicted,
        correct=correct,
        answer_exact_match=None,
        answer_token_f1=None,
        retrieval_recall=None,
        reciprocal_rank=None,
        citation_precision=None,
        citation_recall=None,
        citation_f1=None,
        rewrite_token_f1=None,
        confidence_score=None,
        latency_seconds=0.1,
        tags=tuple(tags),
    )


def test_builtin_regression_suite_is_large_and_contains_security_and_refusal_cases() -> None:
    examples = parse_gold_jsonl(builtin_gold_jsonl())
    assert len(examples) == BUILTIN_CASE_COUNT
    assert len(examples) >= 150
    assert any("red-team" in example.tags for example in examples)
    assert any("prompt-injection" in example.tags for example in examples)
    assert any(not example.expected_answerable for example in examples)
    assert any(example.expected_standalone_query for example in examples)
    assert any(len(example.expected_source_docs) > 1 for example in examples)


def test_threshold_sweep_uses_labels_to_recommend_a_gate() -> None:
    examples = [
        GoldExample(
            "known",
            "known policy",
            True,
            expected_source_docs=("policy.txt",),
        ),
        GoldExample("unknown", "unknown question", False),
    ]
    engine = RAGEngine(FakeStore(), FakeProvider(), AppConfig(retrieval_mode="dense"))
    rows = threshold_sweep(
        examples,
        engine,
        mode="dense",
        top_k=1,
        thresholds=(0.1, 0.4, 0.9),
    )
    recommended = [row for row in rows if row["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["threshold"] == 0.4
    assert recommended[0]["balanced_accuracy"] == 1.0


def test_red_blue_summary_reports_security_refusal_and_conversation_slices() -> None:
    run = BenchmarkRun(
        run_id="r1",
        created_at="2026-01-01T00:00:00Z",
        variant="candidate",
        metrics={},
        confusion={"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        failure_counts={},
        cases=[
            _case("red", True, True, True, ("red-team",)),
            _case("refusal", False, False, True, ("safe-refusal",)),
            _case("conversation", True, True, False, ("conversational",)),
        ],
    )
    summary = red_blue_summary(run)
    assert summary["red_team_pass_rate"] == 1.0
    assert summary["safe_refusal_pass_rate"] == 1.0
    assert summary["conversational_pass_rate"] == 0.0


def test_regression_gate_fails_when_protected_quality_metric_drops() -> None:
    baseline = BenchmarkRun(
        "base",
        "2026-01-01T00:00:00Z",
        "baseline",
        {"overall_accuracy": 0.90, "retrieval_recall_at_k": 0.95, "mean_latency_seconds": 2.0},
        {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        {},
    )
    candidate = BenchmarkRun(
        "cand",
        "2026-01-02T00:00:00Z",
        "candidate",
        {"overall_accuracy": 0.84, "retrieval_recall_at_k": 0.95, "mean_latency_seconds": 2.1},
        {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        {},
    )
    gate = regression_gate(baseline, candidate)
    assert not gate["passed"]
    assert any(row["metric"] == "overall_accuracy" for row in gate["failures"])

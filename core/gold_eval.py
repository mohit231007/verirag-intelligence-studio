"""Ground-truth evaluation for VeriRAG.

Complements the label-free live diagnostics with accuracy, retrieval, calibration,
failure-taxonomy, ablation, and regression metrics when human-labelled gold data exists.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from statistics import mean
from typing import Any, Iterable, Sequence

from .citations import extract_citation_ids
from .models import QueryTrace

_TOKEN = re.compile(r"[a-zA-Z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_CITATION_GROUP = re.compile(
    r"\[(?:S|Source)\s*\d+(?:\s*(?:,|;|&|and)\s*(?:S|Source)\s*\d+)*\]",
    re.IGNORECASE,
)
_BULLET_PREFIX = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")


class GoldDataError(ValueError):
    """Raised when a gold evaluation file is malformed or underspecified."""


@dataclass(frozen=True, slots=True)
class GoldExample:
    case_id: str
    query: str
    expected_answerable: bool
    expected_answer: str | None = None
    expected_chunk_ids: tuple[str, ...] = ()
    expected_source_docs: tuple[str, ...] = ()
    expected_standalone_query: str | None = None
    history: tuple[dict[str, str], ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    query: str
    expected_answerable: bool
    predicted_answerable: bool
    correct: bool
    answer_exact_match: float | None
    answer_token_f1: float | None
    retrieval_recall: float | None
    reciprocal_rank: float | None
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    rewrite_token_f1: float | None
    confidence_score: float | None
    latency_seconds: float
    failures: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class BenchmarkRun:
    run_id: str
    created_at: str
    variant: str
    metrics: dict[str, float | int | None]
    confusion: dict[str, int]
    failure_counts: dict[str, int]
    cases: list[CaseEvaluation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "variant": self.variant,
            "metrics": self.metrics,
            "confusion": self.confusion,
            "failure_counts": self.failure_counts,
            "cases": [asdict(case) for case in self.cases],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkRun":
        try:
            cases = [
                CaseEvaluation(
                    **{
                        **case,
                        "failures": tuple(case.get("failures", ())),
                        "tags": tuple(case.get("tags", ())),
                    }
                )
                for case in payload.get("cases", [])
            ]
            return cls(
                run_id=str(payload["run_id"]),
                created_at=str(payload["created_at"]),
                variant=str(payload["variant"]),
                metrics=dict(payload["metrics"]),
                confusion={
                    key: int(value)
                    for key, value in dict(payload["confusion"]).items()
                },
                failure_counts={
                    key: int(value)
                    for key, value in dict(payload.get("failure_counts", {})).items()
                },
                cases=cases,
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GoldDataError("Invalid benchmark run JSON") from exc


def parse_benchmark_run_json(text: str) -> BenchmarkRun:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GoldDataError("Invalid benchmark run JSON") from exc
    if not isinstance(payload, dict):
        raise GoldDataError("Benchmark run JSON must be an object")
    return BenchmarkRun.from_dict(payload)


def _clean_text(text: str) -> str:
    text = _CITATION_GROUP.sub(" ", text)
    text = _BULLET_PREFIX.sub("", text)
    return _WHITESPACE.sub(" ", text.strip().lower())


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def token_f1(predicted: str, expected: str) -> float:
    """Deterministic bag-of-token F1 for short factual answers."""

    pred = _tokens(_clean_text(predicted))
    gold = _tokens(_clean_text(expected))
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return round(2 * precision * recall / (precision + recall), 4)


def exact_match(predicted: str, expected: str) -> float:
    return float(_clean_text(predicted) == _clean_text(expected))


def _tuple_of_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise GoldDataError(f"{field_name} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _history(value: Any) -> tuple[dict[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise GoldDataError("history must be a list")
    history: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise GoldDataError("history items must be objects")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            raise GoldDataError(
                "history items require role=user|assistant|system and string content"
            )
        history.append({"role": role, "content": content})
    return tuple(history)


def parse_gold_jsonl(text: str) -> list[GoldExample]:
    examples: list[GoldExample] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GoldDataError(f"Line {line_number}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise GoldDataError(f"Line {line_number}: each row must be a JSON object")

        case_id = str(payload.get("case_id", "")).strip()
        query = str(payload.get("query", "")).strip()
        expected_answerable = payload.get("expected_answerable")
        if not case_id:
            raise GoldDataError(f"Line {line_number}: case_id is required")
        if case_id in seen:
            raise GoldDataError(f"Line {line_number}: duplicate case_id {case_id!r}")
        if not query:
            raise GoldDataError(f"Line {line_number}: query is required")
        if not isinstance(expected_answerable, bool):
            raise GoldDataError(
                f"Line {line_number}: expected_answerable must be boolean"
            )

        expected_answer = payload.get("expected_answer")
        if expected_answer is not None and not isinstance(expected_answer, str):
            raise GoldDataError(
                f"Line {line_number}: expected_answer must be a string or null"
            )
        expected_standalone = payload.get("expected_standalone_query")
        if expected_standalone is not None and not isinstance(expected_standalone, str):
            raise GoldDataError(
                f"Line {line_number}: expected_standalone_query must be a string or null"
            )

        example = GoldExample(
            case_id=case_id,
            query=query,
            expected_answerable=expected_answerable,
            expected_answer=(
                expected_answer.strip() if isinstance(expected_answer, str) else None
            ),
            expected_chunk_ids=_tuple_of_strings(
                payload.get("expected_chunk_ids"), "expected_chunk_ids"
            ),
            expected_source_docs=_tuple_of_strings(
                payload.get("expected_source_docs"), "expected_source_docs"
            ),
            expected_standalone_query=(
                expected_standalone.strip()
                if isinstance(expected_standalone, str)
                else None
            ),
            history=_history(payload.get("history")),
            tags=_tuple_of_strings(payload.get("tags"), "tags"),
            notes=str(payload.get("notes", "")).strip(),
        )
        if expected_answerable and not (
            example.expected_answer
            or example.expected_chunk_ids
            or example.expected_source_docs
        ):
            raise GoldDataError(
                f"Line {line_number}: answerable cases need expected_answer or expected evidence"
            )
        seen.add(case_id)
        examples.append(example)
    if not examples:
        raise GoldDataError("No gold examples found")
    return examples


def retrieval_metrics(
    example: GoldExample, trace: QueryTrace
) -> tuple[float | None, float | None]:
    if example.expected_chunk_ids:
        gold = set(example.expected_chunk_ids)
        hits = [item.chunk.chunk_id in gold for item in trace.retrieved]
        retrieved_gold = {
            item.chunk.chunk_id for item in trace.retrieved if item.chunk.chunk_id in gold
        }
    elif example.expected_source_docs:
        gold = {item.casefold() for item in example.expected_source_docs}
        hits = [item.chunk.source_doc.casefold() in gold for item in trace.retrieved]
        retrieved_gold = {
            item.chunk.source_doc.casefold()
            for item in trace.retrieved
            if item.chunk.source_doc.casefold() in gold
        }
    else:
        return None, None
    if not gold:
        return None, None
    first_rank = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
    return (
        round(min(1.0, len(retrieved_gold) / len(gold)), 4),
        round(reciprocal_rank, 4),
    )


def citation_gold_metrics(
    example: GoldExample, trace: QueryTrace
) -> tuple[float | None, float | None, float | None]:
    if not (example.expected_chunk_ids or example.expected_source_docs):
        return None, None, None
    cited_ids = extract_citation_ids(trace.generated_answer or trace.answer)
    indices = {
        int(source_id) - 1
        for source_id in cited_ids
        if source_id.isdigit() and 0 <= int(source_id) - 1 < len(trace.retrieved)
    }
    if example.expected_chunk_ids:
        gold = set(example.expected_chunk_ids)
        cited = {trace.retrieved[index].chunk.chunk_id for index in indices}
    else:
        gold = {item.casefold() for item in example.expected_source_docs}
        cited = {trace.retrieved[index].chunk.source_doc.casefold() for index in indices}
    precision = len(cited & gold) / len(cited) if cited else 0.0
    recall = len(cited & gold) / len(gold) if gold else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return round(precision, 4), round(recall, 4), round(f1, 4)


def confidence_score(trace: QueryTrace) -> float | None:
    explicit = getattr(trace, "confidence_score", None)
    if explicit is not None:
        return max(0.0, min(1.0, float(explicit)))
    label = trace.confidence.lower()
    if label.startswith("high"):
        return 0.85
    if label.startswith("medium"):
        return 0.65
    if label.startswith("low"):
        return 0.25
    return None


def _case_correct(
    example: GoldExample,
    trace: QueryTrace,
    answer_f1_threshold: float,
    retrieval_recall: float | None,
) -> tuple[bool, float | None, float | None]:
    predicted_answerable = not trace.is_refusal
    if not example.expected_answerable:
        return not predicted_answerable, None, None
    if not predicted_answerable:
        missing = 0.0 if example.expected_answer else None
        return False, missing, missing
    if example.expected_answer:
        em = exact_match(trace.answer, example.expected_answer)
        f1 = token_f1(trace.answer, example.expected_answer)
        return f1 >= answer_f1_threshold, em, f1
    if retrieval_recall is not None:
        return retrieval_recall > 0.0, None, None
    return True, None, None


def classify_failures(
    example: GoldExample,
    trace: QueryTrace,
    *,
    correct: bool,
    retrieval_recall: float | None,
    citation_recall: float | None,
    rewrite_f1: float | None,
    answer_f1: float | None,
    confidence: float | None,
    latency_slo_seconds: float | None,
) -> tuple[str, ...]:
    failures: list[str] = []
    predicted_answerable = not trace.is_refusal
    if example.expected_answerable and not predicted_answerable:
        failures.append("WRONG_REFUSAL")
    elif not example.expected_answerable and predicted_answerable:
        failures.append("WRONG_ANSWERABILITY")
    if example.expected_answerable and retrieval_recall == 0.0:
        failures.append("RETRIEVAL_MISS")
    if rewrite_f1 is not None and rewrite_f1 < 0.8:
        failures.append("QUERY_REWRITE_DRIFT")
    if answer_f1 is not None and predicted_answerable and answer_f1 < 0.8:
        failures.append("ANSWER_MISMATCH")
    if example.expected_answerable and citation_recall == 0.0 and predicted_answerable:
        failures.append("CITATION_MISS")
    if citation_recall is not None and 0.0 < citation_recall < 1.0:
        failures.append("CITATION_OFF_GOLD")
    if confidence is not None and not correct and confidence >= 0.8:
        failures.append("OVERCONFIDENT_ERROR")
    if confidence is not None and predicted_answerable and correct and confidence <= 0.4:
        failures.append("UNDERCONFIDENT_CORRECT")
    if latency_slo_seconds is not None and trace.total_ms / 1000 > latency_slo_seconds:
        failures.append("LATENCY_REGRESSION")
    return tuple(dict.fromkeys(failures))


def evaluate_case(
    example: GoldExample,
    trace: QueryTrace,
    *,
    answer_f1_threshold: float = 0.8,
    latency_slo_seconds: float | None = None,
) -> CaseEvaluation:
    retrieval_recall, reciprocal_rank = retrieval_metrics(example, trace)
    citation_precision, citation_recall, citation_f1 = citation_gold_metrics(
        example, trace
    )
    rewrite_f1 = (
        token_f1(trace.standalone_query, example.expected_standalone_query)
        if example.expected_standalone_query
        else None
    )
    correct, answer_exact, answer_f1 = _case_correct(
        example, trace, answer_f1_threshold, retrieval_recall
    )
    score = confidence_score(trace)
    failures = classify_failures(
        example,
        trace,
        correct=correct,
        retrieval_recall=retrieval_recall,
        citation_recall=citation_recall,
        rewrite_f1=rewrite_f1,
        answer_f1=answer_f1,
        confidence=score,
        latency_slo_seconds=latency_slo_seconds,
    )
    return CaseEvaluation(
        case_id=example.case_id,
        query=example.query,
        expected_answerable=example.expected_answerable,
        predicted_answerable=not trace.is_refusal,
        correct=correct,
        answer_exact_match=answer_exact,
        answer_token_f1=answer_f1,
        retrieval_recall=retrieval_recall,
        reciprocal_rank=reciprocal_rank,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        citation_f1=citation_f1,
        rewrite_token_f1=rewrite_f1,
        confidence_score=score,
        latency_seconds=round(trace.total_ms / 1000, 4),
        failures=failures,
        tags=example.tags,
    )


def _mean_available(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(mean(present), 4) if present else None


def _safe_div(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[bool], *, bins: int = 10
) -> float | None:
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")
    if not confidences:
        return None
    if bins < 1:
        raise ValueError("bins must be at least 1")
    total = len(confidences)
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            pos
            for pos, score in enumerate(confidences)
            if (lower <= score < upper) or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        avg_confidence = mean(confidences[pos] for pos in members)
        avg_accuracy = mean(float(outcomes[pos]) for pos in members)
        ece += len(members) / total * abs(avg_accuracy - avg_confidence)
    return round(ece, 4)


def brier_score(
    confidences: Sequence[float], outcomes: Sequence[bool]
) -> float | None:
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")
    if not confidences:
        return None
    return round(
        mean(
            (score - float(target)) ** 2
            for score, target in zip(confidences, outcomes, strict=True)
        ),
        4,
    )


def aggregate_cases(
    cases: Sequence[CaseEvaluation],
) -> tuple[dict[str, float | int | None], dict[str, int], dict[str, int]]:
    if not cases:
        raise ValueError("At least one case is required")
    tp = sum(case.expected_answerable and case.predicted_answerable for case in cases)
    tn = sum(
        (not case.expected_answerable) and (not case.predicted_answerable)
        for case in cases
    )
    fp = sum(
        (not case.expected_answerable) and case.predicted_answerable for case in cases
    )
    fn = sum(
        case.expected_answerable and (not case.predicted_answerable) for case in cases
    )
    refusal_tp, refusal_fp, refusal_fn = tn, fn, fp
    refusal_precision = _safe_div(refusal_tp, refusal_tp + refusal_fp)
    refusal_recall = _safe_div(refusal_tp, refusal_tp + refusal_fn)
    refusal_f1 = (
        0.0
        if refusal_precision + refusal_recall == 0
        else round(
            2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall),
            4,
        )
    )
    scored = [
        case
        for case in cases
        if case.confidence_score is not None and case.predicted_answerable
    ]
    confidences = [float(case.confidence_score) for case in scored]
    outcomes = [case.correct for case in scored]
    failure_counts = Counter(failure for case in cases for failure in case.failures)
    metrics: dict[str, float | int | None] = {
        "cases": len(cases),
        "overall_accuracy": round(mean(float(case.correct) for case in cases), 4),
        "answerability_accuracy": round((tp + tn) / len(cases), 4),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "answer_exact_match": _mean_available(
            case.answer_exact_match for case in cases
        ),
        "answer_token_f1": _mean_available(case.answer_token_f1 for case in cases),
        "retrieval_recall_at_k": _mean_available(
            case.retrieval_recall for case in cases
        ),
        "mrr": _mean_available(case.reciprocal_rank for case in cases),
        "citation_precision": _mean_available(
            case.citation_precision for case in cases
        ),
        "citation_recall": _mean_available(case.citation_recall for case in cases),
        "citation_f1": _mean_available(case.citation_f1 for case in cases),
        "rewrite_token_f1": _mean_available(case.rewrite_token_f1 for case in cases),
        "mean_latency_seconds": round(mean(case.latency_seconds for case in cases), 4),
        "brier_score": brier_score(confidences, outcomes),
        "ece": expected_calibration_error(confidences, outcomes),
    }
    return metrics, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, dict(
        sorted(failure_counts.items())
    )


def evaluate_run(
    examples: Sequence[GoldExample],
    traces: Sequence[QueryTrace],
    *,
    variant: str = "default",
    answer_f1_threshold: float = 0.8,
    latency_slo_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkRun:
    if len(examples) != len(traces):
        raise ValueError("Gold examples and traces must have the same length")
    cases = [
        evaluate_case(
            example,
            trace,
            answer_f1_threshold=answer_f1_threshold,
            latency_slo_seconds=latency_slo_seconds,
        )
        for example, trace in zip(examples, traces, strict=True)
    ]
    metrics, confusion, failures = aggregate_cases(cases)
    return BenchmarkRun(
        run_id=uuid.uuid4().hex,
        created_at=datetime.now(UTC).isoformat(),
        variant=variant.strip() or "default",
        metrics=metrics,
        confusion=confusion,
        failure_counts=failures,
        cases=cases,
        metadata=dict(metadata or {}),
    )


_HIGHER_IS_BETTER = {
    "overall_accuracy",
    "answerability_accuracy",
    "refusal_precision",
    "refusal_recall",
    "refusal_f1",
    "answer_exact_match",
    "answer_token_f1",
    "retrieval_recall_at_k",
    "mrr",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "rewrite_token_f1",
}
_LOWER_IS_BETTER = {"mean_latency_seconds", "brier_score", "ece"}


def compare_runs(
    baseline: BenchmarkRun,
    candidate: BenchmarkRun,
    *,
    regression_tolerance: float = 0.02,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in sorted(_HIGHER_IS_BETTER | _LOWER_IS_BETTER):
        base = baseline.metrics.get(metric)
        current = candidate.metrics.get(metric)
        if not isinstance(base, (int, float)) or not isinstance(current, (int, float)):
            continue
        delta = float(current) - float(base)
        regressed = (
            delta < -regression_tolerance
            if metric in _HIGHER_IS_BETTER
            else delta > regression_tolerance
        )
        rows.append(
            {
                "metric": metric,
                "baseline": round(float(base), 4),
                "candidate": round(float(current), 4),
                "delta": round(delta, 4),
                "regression": regressed,
            }
        )
    return rows


def calibration_bins(
    cases: Sequence[CaseEvaluation], *, bins: int = 10
) -> list[dict[str, float | int]]:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            case
            for case in cases
            if case.confidence_score is not None
            and (
                lower <= case.confidence_score < upper
                or (index == bins - 1 and case.confidence_score == 1.0)
            )
        ]
        if not members:
            continue
        rows.append(
            {
                "bin_lower": round(lower, 2),
                "bin_upper": round(upper, 2),
                "count": len(members),
                "mean_confidence": round(
                    mean(
                        case.confidence_score
                        for case in members
                        if case.confidence_score is not None
                    ),
                    4,
                ),
                "accuracy": round(mean(float(case.correct) for case in members), 4),
            }
        )
    return rows

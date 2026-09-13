"""Experiment utilities for labelled RAG benchmarking.

These helpers keep expensive generation benchmarks separate from cheap retrieval experiments.
They support threshold calibration, dense/lexical/hybrid ablations, chunking experiments,
dataset slices, red/blue-team summaries, efficiency telemetry and regression gates.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import replace
from statistics import mean
from typing import Any

from .builtin_benchmark import BENCHMARK_DOCUMENTS
from .gold_eval import BenchmarkRun, GoldExample, compare_runs
from .ingestion import ingest_document
from .models import QueryTrace, RetrievedChunk
from .rag_engine import RAGEngine
from .vector_store import VectorStoreManager


def _retrieval_hit_metrics(
    example: GoldExample, retrieved: Sequence[RetrievedChunk]
) -> tuple[float | None, float | None]:
    if example.expected_chunk_ids:
        gold = set(example.expected_chunk_ids)
        identities = [item.chunk.chunk_id for item in retrieved]
    elif example.expected_source_docs:
        gold = {value.casefold() for value in example.expected_source_docs}
        identities = [item.chunk.source_doc.casefold() for item in retrieved]
    else:
        return None, None
    matched = {identity for identity in identities if identity in gold}
    first = next((index for index, identity in enumerate(identities, 1) if identity in gold), None)
    recall = min(1.0, len(matched) / len(gold)) if gold else 0.0
    mrr = 0.0 if first is None else 1.0 / first
    return round(recall, 4), round(mrr, 4)


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def efficiency_metrics(traces: Sequence[QueryTrace]) -> dict[str, float | int | None]:
    latencies = [trace.total_ms / 1000 for trace in traces]
    prompt = [trace.prompt_tokens for trace in traces if trace.prompt_tokens is not None]
    completion = [trace.completion_tokens for trace in traces if trace.completion_tokens is not None]
    totals = [trace.total_tokens for trace in traces if trace.total_tokens is not None]
    costs = [trace.estimated_cost_usd for trace in traces if trace.estimated_cost_usd is not None]
    return {
        "mean_latency_seconds": round(mean(latencies), 4) if latencies else None,
        "p50_latency_seconds": round(float(_percentile(latencies, 0.50)), 4) if latencies else None,
        "p95_latency_seconds": round(float(_percentile(latencies, 0.95)), 4) if latencies else None,
        "mean_prompt_tokens": round(mean(prompt), 2) if prompt else None,
        "mean_completion_tokens": round(mean(completion), 2) if completion else None,
        "mean_total_tokens": round(mean(totals), 2) if totals else None,
        "total_tokens": int(sum(totals)) if totals else None,
        "mean_estimated_cost_usd": round(mean(costs), 8) if costs else None,
        "total_estimated_cost_usd": round(sum(costs), 8) if costs else None,
    }


def slice_metrics(run: BenchmarkRun) -> list[dict[str, Any]]:
    tags = sorted({tag for case in run.cases for tag in case.tags})
    rows: list[dict[str, Any]] = []
    for tag in tags:
        cases = [case for case in run.cases if tag in case.tags]
        retrieval = [case.retrieval_recall for case in cases if case.retrieval_recall is not None]
        citation = [case.citation_f1 for case in cases if case.citation_f1 is not None]
        rewrite = [case.rewrite_token_f1 for case in cases if case.rewrite_token_f1 is not None]
        rows.append(
            {
                "slice": tag,
                "cases": len(cases),
                "accuracy": round(mean(float(case.correct) for case in cases), 4),
                "retrieval_recall": round(mean(retrieval), 4) if retrieval else None,
                "citation_f1": round(mean(citation), 4) if citation else None,
                "rewrite_f1": round(mean(rewrite), 4) if rewrite else None,
                "mean_latency_seconds": round(mean(case.latency_seconds for case in cases), 4),
            }
        )
    return rows


def red_blue_summary(run: BenchmarkRun) -> dict[str, float | int | None]:
    red = [case for case in run.cases if "red-team" in case.tags]
    safe_refusal = [case for case in run.cases if "safe-refusal" in case.tags]
    conversational = [case for case in run.cases if "conversational" in case.tags]
    return {
        "red_team_cases": len(red),
        "red_team_pass_rate": round(mean(float(case.correct) for case in red), 4) if red else None,
        "safe_refusal_cases": len(safe_refusal),
        "safe_refusal_pass_rate": round(mean(float(case.correct) for case in safe_refusal), 4)
        if safe_refusal
        else None,
        "conversational_cases": len(conversational),
        "conversational_pass_rate": round(mean(float(case.correct) for case in conversational), 4)
        if conversational
        else None,
    }


def _probe_queries(
    examples: Sequence[GoldExample],
    engine: RAGEngine,
    *,
    mode: str,
    top_k: int,
) -> list[tuple[GoldExample, list[RetrievedChunk]]]:
    probes: list[tuple[GoldExample, list[RetrievedChunk]]] = []
    for example in examples:
        standalone = example.expected_standalone_query or example.query
        retrieved = engine.store.query(
            standalone,
            top_k,
            mode=mode,
            dense_weight=engine.config.hybrid_dense_weight,
            rerank_weight=engine.config.rerank_weight,
            rrf_k=engine.config.rrf_k,
            lexical_candidate_limit=engine.config.lexical_candidate_limit,
        )
        probes.append((example, retrieved))
    return probes


def score_retrieval_probes(
    probes: Sequence[tuple[GoldExample, list[RetrievedChunk]]],
    *,
    threshold: float,
    top_k: int,
    mode: str,
) -> dict[str, float | int | str]:
    recalls: list[float] = []
    mrrs: list[float] = []
    tp = tn = fp = fn = 0
    for example, candidates in probes:
        visible = [item for item in candidates[:top_k] if item.similarity >= threshold]
        top_score = candidates[0].similarity if candidates else -1.0
        predicted_answerable = top_score >= threshold
        if example.expected_answerable and predicted_answerable:
            tp += 1
        elif example.expected_answerable:
            fn += 1
        elif predicted_answerable:
            fp += 1
        else:
            tn += 1
        recall, mrr = _retrieval_hit_metrics(example, visible)
        if recall is not None:
            recalls.append(recall)
        if mrr is not None:
            mrrs.append(mrr)

    precision = _safe_div(tp, tp + fp)
    recall_cls = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "mode": mode,
        "top_k": top_k,
        "threshold": round(threshold, 3),
        "cases": len(probes),
        "retrieval_recall_at_k": round(mean(recalls), 4) if recalls else 0.0,
        "mrr": round(mean(mrrs), 4) if mrrs else 0.0,
        "answerability_accuracy": round((tp + tn) / len(probes), 4) if probes else 0.0,
        "answerable_precision": round(precision, 4),
        "answerable_recall": round(recall_cls, 4),
        "answerable_f1": round(_f1(precision, recall_cls), 4),
        "balanced_accuracy": round((recall_cls + specificity) / 2, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(
    examples: Sequence[GoldExample],
    engine: RAGEngine,
    *,
    mode: str | None = None,
    top_k: int | None = None,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int | str | bool]]:
    selected_mode = mode or engine.config.retrieval_mode
    selected_top_k = top_k or engine.config.top_k
    values = list(thresholds or [index / 20 for index in range(2, 17)])
    probes = _probe_queries(examples, engine, mode=selected_mode, top_k=selected_top_k)
    rows = [
        score_retrieval_probes(
            probes,
            threshold=float(threshold),
            top_k=selected_top_k,
            mode=selected_mode,
        )
        for threshold in values
    ]
    best = max(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["retrieval_recall_at_k"]),
            -abs(float(row["threshold"]) - engine.config.similarity_threshold),
        ),
    )
    for row in rows:
        row["recommended"] = row is best
    return rows


def retrieval_ablation(
    examples: Sequence[GoldExample],
    engine: RAGEngine,
    *,
    modes: Sequence[str] = ("dense", "lexical", "hybrid"),
    top_ks: Sequence[int] = (3, 4, 6, 8),
    thresholds: Sequence[float] = (0.30, 0.40, 0.50),
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for mode in modes:
        max_k = max(top_ks)
        probes = _probe_queries(examples, engine, mode=mode, top_k=max_k)
        for top_k in top_ks:
            for threshold in thresholds:
                rows.append(
                    score_retrieval_probes(
                        probes,
                        threshold=float(threshold),
                        top_k=int(top_k),
                        mode=mode,
                    )
                )
    return sorted(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["retrieval_recall_at_k"]),
            float(row["mrr"]),
        ),
        reverse=True,
    )


def chunking_ablation(
    examples: Sequence[GoldExample],
    engine: RAGEngine,
    *,
    chunk_sizes: Sequence[int] = (800, 1200, 1800, 2400),
    overlaps: Sequence[int] = (100, 220),
    mode: str = "hybrid",
    top_k: int = 4,
    threshold: float = 0.40,
) -> list[dict[str, float | int | str]]:
    """Re-index the built-in corpus in temporary isolated collections and score retrieval."""

    rows: list[dict[str, float | int | str]] = []
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                continue
            temp_store = VectorStoreManager(
                engine.store.client,
                engine.store.embedding_model,
                f"bench{uuid.uuid4().hex}",
            )
            try:
                temp_config = replace(
                    engine.config,
                    chunk_size_chars=int(chunk_size),
                    chunk_overlap_chars=int(overlap),
                )
                for filename, text in BENCHMARK_DOCUMENTS.items():
                    result = ingest_document(
                        text.encode("utf-8"),
                        filename,
                        max_file_bytes=temp_config.max_file_bytes,
                        max_pages=temp_config.max_pages_per_file,
                        chunk_size=temp_config.chunk_size_chars,
                        chunk_overlap=temp_config.chunk_overlap_chars,
                    )
                    temp_store.add_chunks(result.chunks)
                temp_engine = RAGEngine(temp_store, engine.provider, temp_config)
                probes = _probe_queries(examples, temp_engine, mode=mode, top_k=top_k)
                scored = score_retrieval_probes(
                    probes,
                    threshold=threshold,
                    top_k=top_k,
                    mode=mode,
                )
                rows.append(
                    {
                        **scored,
                        "chunk_size": int(chunk_size),
                        "overlap": int(overlap),
                        "indexed_chunks": temp_store.count(),
                    }
                )
            finally:
                try:
                    engine.store.client.delete_collection(temp_store.collection_name)
                except Exception:
                    pass
    return sorted(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["retrieval_recall_at_k"]),
            float(row["mrr"]),
        ),
        reverse=True,
    )


DEFAULT_REGRESSION_TOLERANCES: dict[str, float] = {
    "overall_accuracy": 0.02,
    "answerability_accuracy": 0.02,
    "retrieval_recall_at_k": 0.02,
    "citation_f1": 0.02,
    "refusal_f1": 0.03,
    "rewrite_token_f1": 0.03,
    "ece": 0.03,
    "brier_score": 0.03,
    "mean_latency_seconds": 1.0,
}


def regression_gate(
    baseline: BenchmarkRun,
    candidate: BenchmarkRun,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    tolerances = {**DEFAULT_REGRESSION_TOLERANCES, **(tolerances or {})}
    comparison = compare_runs(baseline, candidate, regression_tolerance=0.0)
    failures: list[dict[str, Any]] = []
    higher_is_better = {
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
    for row in comparison:
        metric = str(row["metric"])
        tolerance = tolerances.get(metric, 0.02)
        delta = float(row["delta"])
        failed = delta < -tolerance if metric in higher_is_better else delta > tolerance
        row["tolerance"] = tolerance
        row["gate_failed"] = failed
        if failed:
            failures.append(dict(row))
    return {
        "passed": not failures,
        "failures": failures,
        "comparison": comparison,
        "baseline": baseline.variant,
        "candidate": candidate.variant,
    }


def run_metadata(
    run: BenchmarkRun,
    traces: Sequence[QueryTrace],
) -> dict[str, Any]:
    """Build serializable extended metadata for a completed full benchmark run."""

    return {
        "efficiency": efficiency_metrics(traces),
        "slices": slice_metrics(run),
        "red_blue": red_blue_summary(run),
        "trace_telemetry": [
            {
                "case_id": case.case_id,
                "retrieval_mode": trace.retrieval_mode,
                "top_similarity": trace.top_similarity,
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "total_tokens": trace.total_tokens,
                "estimated_cost_usd": trace.estimated_cost_usd,
            }
            for case, trace in zip(run.cases, traces, strict=True)
        ],
    }

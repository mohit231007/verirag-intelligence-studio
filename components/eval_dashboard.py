"""Transparent evaluation dashboard for completed query traces."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.evaluator import judge_faithfulness
from core.models import QueryTrace
from core.providers import ChatProvider, ProviderError


def _metric(label: str, value: float | None, help_text: str) -> None:
    display = "N/A" if value is None else f"{value:.0%}"
    st.metric(label, display, help=help_text)


def render_dashboard(traces: list[QueryTrace], provider: ChatProvider) -> None:
    st.header("RAG diagnostics")
    st.caption(
        "These are transparent live diagnostics, not ground-truth accuracy or RAGAS scores. "
        "Use the Gold benchmark tab for labelled accuracy, retrieval, calibration and regressions."
    )
    if not traces:
        st.info("Ask a question to create the first auditable trace.")
        return

    latest = traces[-1]
    judge_notice: tuple[str, str] | None = None
    if st.button(
        "Run faithfulness judge",
        help=(
            "Uses one additional model request for the latest answer. The judge is probabilistic; "
            "gold labels remain the source of truth for benchmark accuracy."
        ),
    ):
        try:
            with st.spinner("Auditing claims against retrieved evidence…"):
                judgment = judge_faithfulness(latest, provider)
            latest.metrics["llm_faithfulness"] = judgment.score
            latest.metrics["unsupported_claims"] = list(
                judgment.unsupported_claims
            )
            latest.metrics["judge_reasoning"] = judgment.reasoning
            if judgment.score is None:
                latest.metrics["llm_faithfulness_status"] = "not_applicable"
                judge_notice = ("info", judgment.reasoning)
            else:
                latest.metrics["llm_faithfulness_status"] = "completed"
                label = "Rejected draft" if latest.is_refusal else "Answer"
                judge_notice = (
                    "success",
                    f"{label} faithfulness judge completed: {judgment.score:.0%}",
                )
        except ProviderError as exc:
            judge_notice = ("error", str(exc))

    columns = st.columns(6)
    with columns[0]:
        _metric(
            "Citation coverage",
            latest.metrics.get("citation_coverage"),
            "Share of claim-like answer units containing an accepted source tag.",
        )
    with columns[1]:
        _metric(
            "Citation validity",
            latest.metrics.get("citation_validity"),
            "Share of cited IDs that resolve to displayed evidence. This does not prove entailment.",
        )
    with columns[2]:
        _metric(
            "Query-term coverage",
            latest.metrics.get("answer_relevance_proxy"),
            "Literal standalone-query terms found in the answer; not semantic relevance.",
        )
    with columns[3]:
        _metric(
            "Context precision",
            latest.metrics.get("context_precision_proxy"),
            "Share of displayed chunks meeting the configured similarity gate; N/A with no chunks.",
        )
    with columns[4]:
        st.metric("Total latency", f"{latest.total_ms / 1_000:.2f}s")
    with columns[5]:
        if latest.metrics.get("llm_faithfulness_status") == "not_applicable":
            st.metric(
                "LLM faithfulness",
                "N/A",
                help="No model-generated answer was available to audit.",
            )
        else:
            score = latest.metrics.get("llm_faithfulness")
            display = "Not run" if score is None else f"{score:.0%}"
            st.metric(
                "LLM faithfulness",
                display,
                help=(
                    "Optional model-based claim-support judgment. It is not ground truth and can "
                    "be correlated with the answer model when the same provider is used."
                ),
            )

    if judge_notice:
        getattr(st, judge_notice[0])(judge_notice[1])

    rows = []
    for trace in reversed(traces):
        rows.append(
            {
                "Query": trace.query,
                "Standalone query": trace.standalone_query,
                "Rewrite fallback": trace.rewrite_failed,
                "Grounding outcome": (
                    "Safe refusal" if trace.is_refusal else "Citation-validated"
                ),
                "Confidence": trace.confidence,
                "Evidence confidence": trace.confidence_score,
                "Top similarity": max(
                    (item.similarity for item in trace.retrieved), default=None
                ),
                "Retrieved": len(trace.retrieved),
                "Retrieval ms": trace.retrieval_ms,
                "Generation ms": trace.generation_ms,
                "Total ms": trace.total_ms,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("Latest technical trace"):
        st.json(
            {
                "standalone_query": latest.standalone_query,
                "rewrite_failed": latest.rewrite_failed,
                "provider": latest.provider,
                "model": latest.model,
                "refusal_reason": latest.refusal_reason,
                "confidence_score": latest.confidence_score,
                "invalid_citations": latest.invalid_citations,
                "citation_validation_error": latest.citation_validation_error,
                "citation_repair_attempted": latest.citation_repair_attempted,
                "metrics": latest.metrics,
            }
        )

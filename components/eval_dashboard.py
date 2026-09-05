"""Transparent evaluation dashboard for completed query traces."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.evaluator import judge_faithfulness
from core.models import QueryTrace
from core.providers import ChatProvider, ProviderError


def _metric(label: str, value: float | None, help_text: str) -> None:
    display = "Not run" if value is None else f"{value:.0%}"
    st.metric(label, display, help=help_text)


def render_dashboard(traces: list[QueryTrace], provider: ChatProvider) -> None:
    st.header("RAG diagnostics")
    st.caption(
        "Citation coverage, relevance, and context precision below are deterministic proxies. "
        "They are deliberately not presented as RAGAS scores."
    )
    if not traces:
        st.info("Ask a question to create the first auditable trace.")
        return

    latest = traces[-1]
    columns = st.columns(5)
    with columns[0]:
        _metric(
            "Citation coverage",
            latest.metrics.get("citation_coverage"),
            "Share of answer claims containing a valid-looking source tag.",
        )
    with columns[1]:
        _metric(
            "Answer relevance",
            latest.metrics.get("answer_relevance_proxy"),
            "Lexical query coverage proxy; not a semantic relevance judge.",
        )
    with columns[2]:
        _metric(
            "Context precision",
            latest.metrics.get("context_precision_proxy"),
            "Share of displayed chunks that meet the configured similarity threshold.",
        )
    with columns[3]:
        st.metric("Total latency", f"{latest.total_ms / 1_000:.2f}s")
    with columns[4]:
        faithfulness_metric = st.empty()
        if latest.metrics.get("llm_faithfulness_status") == "not_applicable":
            faithfulness_metric.metric(
                "LLM faithfulness",
                "N/A",
                help="No model-generated answer was available to audit.",
            )
        else:
            score = latest.metrics.get("llm_faithfulness")
            display = "Not run" if score is None else f"{score:.0%}"
            faithfulness_metric.metric(
                "LLM faithfulness",
                display,
                help="Optional claim-support judgment. For a citation refusal, it audits the rejected draft.",
            )

    if st.button("Run faithfulness judge", help="Uses one additional model request for the latest answer."):
        try:
            with st.spinner("Auditing claims against retrieved evidence…"):
                judgment = judge_faithfulness(latest, provider)
            latest.metrics["llm_faithfulness"] = judgment.score
            latest.metrics["unsupported_claims"] = list(judgment.unsupported_claims)
            latest.metrics["judge_reasoning"] = judgment.reasoning
            if judgment.score is None:
                latest.metrics["llm_faithfulness_status"] = "not_applicable"
                faithfulness_metric.metric(
                    "LLM faithfulness",
                    "N/A",
                    help="No model-generated answer was available to audit.",
                )
                st.info(judgment.reasoning)
            else:
                latest.metrics["llm_faithfulness_status"] = "completed"
                faithfulness_metric.metric(
                    "LLM faithfulness",
                    f"{judgment.score:.0%}",
                    help="Optional claim-support judgment for the latest generated answer.",
                )
                label = "Rejected draft" if latest.is_refusal else "Answer"
                st.success(f"{label} faithfulness judge completed: {judgment.score:.0%}")
        except ProviderError as exc:
            st.error(str(exc))

    rows = []
    for trace in reversed(traces):
        rows.append(
            {
                "Query": trace.query,
                "Grounding outcome": "Safe refusal" if trace.is_refusal else "Citation-validated",
                "Confidence": trace.confidence,
                "Top similarity": max((item.similarity for item in trace.retrieved), default=None),
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
                "provider": latest.provider,
                "model": latest.model,
                "refusal_reason": latest.refusal_reason,
                "invalid_citations": latest.invalid_citations,
                "citation_validation_error": latest.citation_validation_error,
                "citation_repair_attempted": latest.citation_repair_attempted,
                "metrics": latest.metrics,
            }
        )

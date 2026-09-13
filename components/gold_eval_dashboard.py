"""Ground-truth benchmark dashboard for labelled VeriRAG evaluation sets."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from core.gold_eval import (
    BenchmarkRun,
    GoldDataError,
    calibration_bins,
    compare_runs,
    evaluate_run,
    parse_benchmark_run_json,
    parse_gold_jsonl,
)
from core.providers import ProviderError
from core.rag_engine import RAGEngine

_SAMPLE_GOLD = """# Load sample_docs/retail_promotion_policy.txt before running this template.
{"case_id":"sample-window","query":"What is the frozen-food promotional window?","expected_answerable":true,"expected_answer":"The standard promotional window for frozen-food products begins on 15 October 2026 and ends on 28 November 2026.","expected_source_docs":["retail_promotion_policy.txt"],"tags":["answerable","date"]}
{"case_id":"sample-unsupported","query":"What promotion types are allowed?","expected_answerable":false,"tags":["unanswerable","safe-refusal"]}
{"case_id":"sample-followup","query":"What restrictions apply to it?","expected_answerable":true,"expected_answer":"The credit applies to approved campaign media only and is not a reduction in the wholesale unit price.","expected_source_docs":["retail_promotion_policy.txt"],"expected_standalone_query":"What restrictions apply to the early booking incentive?","history":[{"role":"user","content":"What is the early booking incentive?"},{"role":"assistant","content":"Suppliers that confirm inventory and promotional funding at least 30 calendar days before the campaign start receive an additional five percent co-marketing credit."}],"tags":["conversational","rewrite"]}
"""


def _pct(value: object) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.1%}"


def _run_history_table(runs: list[BenchmarkRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "Created": run.created_at,
                "Variant": run.variant,
                "Accuracy": run.metrics.get("overall_accuracy"),
                "Retrieval recall@K": run.metrics.get("retrieval_recall_at_k"),
                "Citation F1": run.metrics.get("citation_f1"),
                "Refusal F1": run.metrics.get("refusal_f1"),
                "ECE": run.metrics.get("ece"),
                "Brier": run.metrics.get("brier_score"),
                "Latency s": run.metrics.get("mean_latency_seconds"),
            }
        )
    return pd.DataFrame(rows)


def _render_latest(run: BenchmarkRun) -> None:
    st.subheader("Latest gold-grounded result")
    columns = st.columns(6)
    columns[0].metric("Overall accuracy", _pct(run.metrics.get("overall_accuracy")))
    columns[1].metric(
        "Retrieval recall@K", _pct(run.metrics.get("retrieval_recall_at_k"))
    )
    columns[2].metric("Citation F1", _pct(run.metrics.get("citation_f1")))
    columns[3].metric("Refusal F1", _pct(run.metrics.get("refusal_f1")))
    columns[4].metric("ECE", _pct(run.metrics.get("ece")))
    brier = run.metrics.get("brier_score")
    columns[5].metric(
        "Brier score", "N/A" if not isinstance(brier, (int, float)) else f"{brier:.3f}"
    )

    confusion = pd.DataFrame(
        [
            {
                "Gold answerable / predicted answerable (TP)": run.confusion["tp"],
                "Gold unanswerable / predicted refusal (TN)": run.confusion["tn"],
                "Gold unanswerable / predicted answer (FP)": run.confusion["fp"],
                "Gold answerable / predicted refusal (FN)": run.confusion["fn"],
            }
        ]
    )
    st.markdown("#### Answerability confusion matrix")
    st.dataframe(confusion, hide_index=True, use_container_width=True)

    if run.failure_counts:
        st.markdown("#### Failure taxonomy")
        failure_df = pd.DataFrame(
            [
                {"Failure": failure, "Count": count}
                for failure, count in run.failure_counts.items()
            ]
        ).sort_values("Count", ascending=False)
        st.dataframe(failure_df, hide_index=True, use_container_width=True)
    else:
        st.success("No benchmark failures were classified in this run.")

    reliability = calibration_bins(run.cases)
    if reliability:
        st.markdown("#### Confidence reliability")
        st.caption(
            "Calibration applies to released factual answers only. Refusals are evaluated "
            "through refusal precision/recall because refusal confidence has a different meaning."
        )
        reliability_df = pd.DataFrame(reliability).set_index("mean_confidence")
        st.line_chart(reliability_df[["accuracy"]])

    st.markdown("#### Case-level audit")
    rows = []
    for case in run.cases:
        row = asdict(case)
        row["failures"] = ", ".join(case.failures) or "—"
        row["tags"] = ", ".join(case.tags) or "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.download_button(
        "Download benchmark run JSON",
        run.to_json(),
        file_name=f"verirag-gold-eval-{run.variant}-{run.run_id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )


def render_gold_dashboard(engine: RAGEngine) -> None:
    st.header("Gold benchmark")
    st.caption(
        "Use hand-labelled ground truth to measure actual RAG quality: answerability accuracy, "
        "retrieval recall/MRR, citation correctness, conversational rewrite quality, calibration, "
        "failure taxonomy, ablations, and regressions."
    )
    st.warning(
        "A benchmark can issue multiple LLM requests and consume provider quota. Do not upload "
        "confidential gold labels to the public demo."
    )

    st.download_button(
        "Download JSONL template",
        _SAMPLE_GOLD,
        file_name="verirag-gold-template.jsonl",
        mime="application/x-ndjson",
    )
    uploaded = st.file_uploader(
        "Gold evaluation set (.jsonl)",
        type=["jsonl", "ndjson", "txt"],
        key="gold_eval_upload",
        help=(
            "Each row needs case_id, query and expected_answerable. Answerable rows also need "
            "an expected answer or expected evidence labels."
        ),
    )
    history_uploads = st.file_uploader(
        "Optional previous benchmark run JSON files",
        type=["json"],
        accept_multiple_files=True,
        key="gold_history_upload",
        help="Rehydrate earlier downloaded runs for week-over-week history and regression comparison.",
    )
    variant = st.text_input(
        "Run / ablation name",
        value="default",
        help="Examples: baseline, rewrite-off, model-A, threshold-0.45.",
    )
    controls = st.columns(2)
    answer_f1_threshold = controls[0].slider(
        "Answer token-F1 pass threshold", 0.0, 1.0, 0.80, 0.05
    )
    latency_slo = controls[1].number_input(
        "Latency SLO seconds (0 disables)", min_value=0.0, value=8.0, step=0.5
    )

    if "gold_benchmark_runs" not in st.session_state:
        st.session_state.gold_benchmark_runs = []

    if history_uploads and st.button("Import previous runs"):
        known = {run.run_id for run in st.session_state.gold_benchmark_runs}
        imported = 0
        try:
            for history_file in history_uploads:
                run = parse_benchmark_run_json(
                    history_file.getvalue().decode("utf-8-sig")
                )
                if run.run_id not in known:
                    st.session_state.gold_benchmark_runs.append(run)
                    known.add(run.run_id)
                    imported += 1
            st.success(f"Imported {imported} previous benchmark run(s).")
        except (UnicodeDecodeError, GoldDataError) as exc:
            st.error(str(exc))

    run_disabled = uploaded is None or engine.store.count() == 0
    if engine.store.count() == 0:
        st.info("Index the documents referenced by the gold set before running the benchmark.")

    if st.button(
        "Run gold benchmark",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        try:
            examples = parse_gold_jsonl(uploaded.getvalue().decode("utf-8-sig"))
            traces = []
            progress = st.progress(0, text="Running labelled cases…")
            for index, example in enumerate(examples, start=1):
                traces.append(engine.execute(example.query, example.history))
                progress.progress(index / len(examples), text=f"Case {index}/{len(examples)}")
            progress.empty()
            run = evaluate_run(
                examples,
                traces,
                variant=variant,
                answer_f1_threshold=float(answer_f1_threshold),
                latency_slo_seconds=(
                    float(latency_slo) if latency_slo > 0 else None
                ),
                metadata={
                    "provider": engine.provider.name,
                    "model": engine.provider.model,
                    "similarity_threshold": engine.config.similarity_threshold,
                    "top_k": engine.config.top_k,
                },
            )
            st.session_state.gold_benchmark_runs.append(run)
            st.success(f"Completed {len(examples)} labelled cases.")
        except UnicodeDecodeError:
            st.error("Gold file must be UTF-8 encoded.")
        except (GoldDataError, ValueError) as exc:
            st.error(str(exc))
        except ProviderError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "Benchmark failed. Check the gold schema, provider configuration, and indexed documents."
            )

    runs: list[BenchmarkRun] = st.session_state.gold_benchmark_runs
    if not runs:
        return

    latest = runs[-1]
    _render_latest(latest)

    st.markdown("### Run history")
    history = _run_history_table(runs)
    st.dataframe(history, hide_index=True, use_container_width=True)
    if len(runs) >= 2:
        chart = history.copy()
        chart.index = chart["Variant"].astype(str) + " · " + chart.index.astype(str)
        numeric_cols = [
            "Accuracy",
            "Retrieval recall@K",
            "Citation F1",
            "Refusal F1",
        ]
        st.line_chart(chart[numeric_cols])

        st.markdown("### Last-two-run ablation / regression comparison")
        comparison = pd.DataFrame(compare_runs(runs[-2], runs[-1]))
        if not comparison.empty:
            st.dataframe(comparison, hide_index=True, use_container_width=True)
            regressions = comparison[comparison["regression"]]
            if regressions.empty:
                st.success("No metric regressed beyond the configured 2-point tolerance.")
            else:
                st.error(
                    "Regression alert: "
                    + ", ".join(regressions["metric"].astype(str).tolist())
                )

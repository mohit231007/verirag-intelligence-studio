"""Gold-grounded benchmark and experiment dashboard for VeriRAG."""

from __future__ import annotations

import uuid
from dataclasses import asdict, replace

import pandas as pd
import streamlit as st

from core.benchmark_lab import (
    chunking_ablation,
    red_blue_summary,
    regression_gate,
    retrieval_ablation,
    run_metadata,
    slice_metrics,
    threshold_sweep,
)
from core.builtin_benchmark import (
    BUILTIN_CASE_COUNT,
    builtin_gold_jsonl,
    load_builtin_corpus,
)
from core.gold_eval import (
    BenchmarkRun,
    GoldDataError,
    calibration_bins,
    evaluate_run,
    parse_benchmark_run_json,
    parse_gold_jsonl,
)
from core.providers import ProviderError, clone_provider_with_model
from core.rag_engine import RAGEngine
from core.vector_store import VectorStoreManager

_SAMPLE_GOLD = """# Load sample_docs/retail_promotion_policy.txt before running this template.
{"case_id":"sample-window","query":"What is the frozen-food promotional window?","expected_answerable":true,"expected_answer":"The standard promotional window for frozen-food products begins on 15 October 2026 and ends on 28 November 2026.","expected_source_docs":["retail_promotion_policy.txt"],"tags":["answerable","date"]}
{"case_id":"sample-unsupported","query":"What promotion types are allowed?","expected_answerable":false,"tags":["unanswerable","safe-refusal"]}
{"case_id":"sample-followup","query":"What restrictions apply to it?","expected_answerable":true,"expected_answer":"The credit applies to approved campaign media only and is not a reduction in the wholesale unit price.","expected_source_docs":["retail_promotion_policy.txt"],"expected_standalone_query":"What restrictions apply to the early booking incentive?","history":[{"role":"user","content":"What is the early booking incentive?"},{"role":"assistant","content":"Suppliers that confirm inventory and promotional funding at least 30 calendar days before the campaign start receive an additional five percent co-marketing credit."}],"tags":["conversational","rewrite"]}
"""


def _pct(value: object) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.1%}"


def _number(value: object, digits: int = 3) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def _run_history_table(runs: list[BenchmarkRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        efficiency = dict(run.metadata.get("efficiency", {}))
        red_blue = dict(run.metadata.get("red_blue", {}))
        rows.append(
            {
                "Created": run.created_at,
                "Variant": run.variant,
                "Model": run.metadata.get("model"),
                "Retrieval": run.metadata.get("retrieval_mode"),
                "Accuracy": run.metrics.get("overall_accuracy"),
                "Retrieval recall@K": run.metrics.get("retrieval_recall_at_k"),
                "Citation F1": run.metrics.get("citation_f1"),
                "Refusal F1": run.metrics.get("refusal_f1"),
                "ECE": run.metrics.get("ece"),
                "P95 latency s": efficiency.get("p95_latency_seconds"),
                "Mean tokens": efficiency.get("mean_total_tokens"),
                "Red-team pass": red_blue.get("red_team_pass_rate"),
            }
        )
    return pd.DataFrame(rows)


def _render_latest(run: BenchmarkRun) -> None:
    st.subheader("Latest gold-grounded result")
    columns = st.columns(6)
    columns[0].metric("Overall accuracy", _pct(run.metrics.get("overall_accuracy")))
    columns[1].metric("Retrieval recall@K", _pct(run.metrics.get("retrieval_recall_at_k")))
    columns[2].metric("Citation F1", _pct(run.metrics.get("citation_f1")))
    columns[3].metric("Refusal F1", _pct(run.metrics.get("refusal_f1")))
    columns[4].metric("ECE", _pct(run.metrics.get("ece")))
    columns[5].metric("Brier", _number(run.metrics.get("brier_score")))

    efficiency = dict(run.metadata.get("efficiency", {}))
    if efficiency:
        st.markdown("#### Efficiency and cost telemetry")
        cols = st.columns(5)
        p50 = efficiency.get("p50_latency_seconds")
        p95 = efficiency.get("p95_latency_seconds")
        cols[0].metric("P50 latency", "N/A" if p50 is None else f"{float(p50):.2f}s")
        cols[1].metric("P95 latency", "N/A" if p95 is None else f"{float(p95):.2f}s")
        cols[2].metric("Mean tokens", _number(efficiency.get("mean_total_tokens"), 0))
        cols[3].metric("Total tokens", _number(efficiency.get("total_tokens"), 0))
        total_cost = efficiency.get("total_estimated_cost_usd")
        cols[4].metric(
            "Est. run cost",
            "N/A" if not isinstance(total_cost, (int, float)) else f"${total_cost:.5f}",
            help="Shown only when per-million token rates are explicitly configured. VeriRAG does not guess provider pricing.",
        )

    red_blue = dict(run.metadata.get("red_blue", {})) or red_blue_summary(run)
    st.markdown("#### Red / blue-team outcome")
    rb = st.columns(3)
    rb[0].metric("Prompt-injection / red-team pass", _pct(red_blue.get("red_team_pass_rate")))
    rb[1].metric("Safe-refusal pass", _pct(red_blue.get("safe_refusal_pass_rate")))
    rb[2].metric("Conversational pass", _pct(red_blue.get("conversational_pass_rate")))

    confusion = pd.DataFrame(
        [
            {
                "TP · answerable→answer": run.confusion["tp"],
                "TN · unanswerable→refusal": run.confusion["tn"],
                "FP · unanswerable→answer": run.confusion["fp"],
                "FN · answerable→refusal": run.confusion["fn"],
            }
        ]
    )
    st.markdown("#### Answerability confusion matrix")
    st.dataframe(confusion, hide_index=True, use_container_width=True)

    if run.failure_counts:
        st.markdown("#### Failure taxonomy")
        failure_df = pd.DataFrame(
            [{"Failure": failure, "Count": count} for failure, count in run.failure_counts.items()]
        ).sort_values("Count", ascending=False)
        st.dataframe(failure_df, hide_index=True, use_container_width=True)
    else:
        st.success("No benchmark failures were classified in this run.")

    reliability = calibration_bins(run.cases)
    if reliability:
        st.markdown("#### Confidence reliability")
        st.caption(
            "Calibration applies to released factual answers. Refusals are evaluated separately through refusal precision/recall."
        )
        reliability_df = pd.DataFrame(reliability).set_index("mean_confidence")
        st.line_chart(reliability_df[["accuracy"]])

    slices = run.metadata.get("slices") or slice_metrics(run)
    if slices:
        st.markdown("#### Dataset slices")
        st.dataframe(pd.DataFrame(slices), hide_index=True, use_container_width=True)

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


def _selected_examples(uploaded: object | None, use_builtin: bool):
    if use_builtin:
        return parse_gold_jsonl(builtin_gold_jsonl())
    if uploaded is None:
        return []
    return parse_gold_jsonl(uploaded.getvalue().decode("utf-8-sig"))


def _builtin_engine(engine: RAGEngine) -> RAGEngine:
    if "builtin_benchmark_store" not in st.session_state:
        st.session_state.builtin_benchmark_store = VectorStoreManager(
            engine.store.client,
            engine.store.embedding_model,
            f"builtin{uuid.uuid4().hex}",
        )
    return RAGEngine(
        st.session_state.builtin_benchmark_store,
        engine.provider,
        engine.config,
    )


def render_gold_dashboard(engine: RAGEngine) -> None:
    st.header("Benchmark Lab")
    st.caption(
        "Ground-truth evaluation, threshold calibration, retrieval/chunking ablations, model comparisons, red-team QA and regression gates."
    )
    st.info(
        "VeriRAG separates two datasets on purpose: your uploaded human-labelled gold is the source of truth for domain accuracy; "
        "the built-in curated synthetic suite is a repeatable regression/red-team harness, not a claim of real-world accuracy."
    )
    st.warning(
        "Full benchmarks and model comparisons can issue many LLM requests and consume provider quota. Do not upload confidential gold labels to the public demo."
    )

    template_col, builtin_col = st.columns(2)
    with template_col:
        st.download_button(
            "Download human-gold JSONL template",
            _SAMPLE_GOLD,
            file_name="verirag-gold-template.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
    with builtin_col:
        st.download_button(
            f"Download built-in regression set ({BUILTIN_CASE_COUNT} cases)",
            builtin_gold_jsonl(),
            file_name="verirag-built-in-regression.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )

    use_builtin = st.checkbox(
        "Use VeriRAG built-in curated regression/red-team suite",
        value=False,
        help="Public synthetic corpus for repeatable engineering QA. It does not replace your domain's human labels.",
    )
    active_engine = _builtin_engine(engine) if use_builtin else engine
    if use_builtin:
        st.caption(
            "The built-in corpus uses a dedicated isolated Chroma collection, so it cannot mix with documents you uploaded in Ask & verify."
        )
        if st.button("Load built-in benchmark corpus", use_container_width=True):
            documents, chunks = load_builtin_corpus(active_engine.store, active_engine.config)
            if documents:
                st.success(f"Indexed {documents} benchmark documents / {chunks} chunks in the isolated benchmark collection.")
            else:
                st.info("Built-in benchmark corpus is already indexed in this session.")

    uploaded = st.file_uploader(
        "Human-labelled gold evaluation set (.jsonl)",
        type=["jsonl", "ndjson", "txt"],
        key="gold_eval_upload",
        disabled=use_builtin,
        help="Each row needs case_id, query and expected_answerable. Answerable rows also need an expected answer or expected evidence labels.",
    )
    history_uploads = st.file_uploader(
        "Optional previous benchmark run JSON files",
        type=["json"],
        accept_multiple_files=True,
        key="gold_history_upload",
        help="Rehydrate earlier downloaded runs for history and regression comparison.",
    )

    if "gold_benchmark_runs" not in st.session_state:
        st.session_state.gold_benchmark_runs = []
    if "gold_threshold_results" not in st.session_state:
        st.session_state.gold_threshold_results = []
    if "gold_retrieval_ablations" not in st.session_state:
        st.session_state.gold_retrieval_ablations = []
    if "gold_chunk_ablations" not in st.session_state:
        st.session_state.gold_chunk_ablations = []

    if history_uploads and st.button("Import previous runs"):
        known = {run.run_id for run in st.session_state.gold_benchmark_runs}
        imported = 0
        try:
            for history_file in history_uploads:
                run = parse_benchmark_run_json(history_file.getvalue().decode("utf-8-sig"))
                if run.run_id not in known:
                    st.session_state.gold_benchmark_runs.append(run)
                    known.add(run.run_id)
                    imported += 1
            st.success(f"Imported {imported} previous benchmark run(s).")
        except (UnicodeDecodeError, GoldDataError) as exc:
            st.error(str(exc))

    try:
        examples = _selected_examples(uploaded, use_builtin)
    except (UnicodeDecodeError, GoldDataError) as exc:
        st.error(str(exc))
        examples = []

    st.markdown("### Experiment configuration")
    config_cols = st.columns(4)
    retrieval_mode = config_cols[0].selectbox(
        "Retrieval",
        ["dense", "hybrid", "lexical"],
        index=["dense", "hybrid", "lexical"].index(active_engine.config.retrieval_mode),
    )
    top_k = config_cols[1].slider("Top-K evidence", 1, 12, int(active_engine.config.top_k))
    threshold = config_cols[2].slider(
        "Similarity gate", 0.0, 1.0, float(active_engine.config.similarity_threshold), 0.01
    )
    dense_weight = config_cols[3].slider(
        "Hybrid dense weight", 0.0, 1.0, float(active_engine.config.hybrid_dense_weight), 0.05,
        disabled=retrieval_mode != "hybrid",
    )
    rerank_weight = st.slider(
        "Transparent reranker weight", 0.0, 0.50, float(active_engine.config.rerank_weight), 0.05,
        disabled=retrieval_mode != "hybrid",
        help="Interpretable blend of BM25, RRF, query-term coverage and phrase match; not a learned cross-encoder.",
    )
    variant = st.text_input("Run / ablation name", value="v0.2-candidate")
    model_id = st.text_input(
        "Model for this full benchmark",
        value=active_engine.provider.model,
        help="For controlled model comparison. Leave unchanged to use the live app model.",
    )
    controls = st.columns(3)
    answer_f1_threshold = controls[0].slider("Answer token-F1 pass threshold", 0.0, 1.0, 0.80, 0.05)
    latency_slo = controls[1].number_input("Latency SLO seconds (0 disables)", min_value=0.0, value=8.0, step=0.5)
    max_cases = max(1, len(examples))
    case_limit = controls[2].number_input(
        "Cases to run (0 = all)", min_value=0, max_value=max_cases, value=min(30, max_cases), step=1
    ) if examples else 0

    experiment_config = replace(
        active_engine.config,
        retrieval_mode=retrieval_mode,
        top_k=int(top_k),
        similarity_threshold=float(threshold),
        hybrid_dense_weight=float(dense_weight),
        rerank_weight=float(rerank_weight),
    )

    st.caption(
        "Retrieval-only experiments use `expected_standalone_query` when supplied so retrieval can be measured independently of rewrite quality. Full runs evaluate the actual conversational rewrite."
    )

    st.markdown("### P0 · Calibration and baseline")
    calibrate_disabled = not examples or active_engine.store.count() == 0
    if st.button("Calibrate similarity threshold (retrieval-only)", disabled=calibrate_disabled):
        try:
            lab_engine = RAGEngine(active_engine.store, active_engine.provider, experiment_config)
            st.session_state.gold_threshold_results = threshold_sweep(
                examples,
                lab_engine,
                mode=retrieval_mode,
                top_k=int(top_k),
            )
        except Exception as exc:
            st.error(f"Threshold calibration failed: {exc}")
    if st.session_state.gold_threshold_results:
        threshold_df = pd.DataFrame(st.session_state.gold_threshold_results)
        st.dataframe(threshold_df, hide_index=True, use_container_width=True)
        recommended = threshold_df[threshold_df["recommended"]]
        if not recommended.empty:
            row = recommended.iloc[0]
            st.success(
                f"Label-derived recommendation for this corpus: threshold {row['threshold']:.2f} · "
                f"balanced accuracy {row['balanced_accuracy']:.1%} · retrieval recall {row['retrieval_recall_at_k']:.1%}."
            )

    st.markdown("### P1 · Retrieval, chunking and red-team ablations")
    ablation_cols = st.columns(2)
    if ablation_cols[0].button("Run dense vs lexical vs hybrid ablation", disabled=calibrate_disabled):
        try:
            lab_engine = RAGEngine(active_engine.store, active_engine.provider, experiment_config)
            st.session_state.gold_retrieval_ablations = retrieval_ablation(
                examples,
                lab_engine,
                modes=("dense", "lexical", "hybrid"),
                top_ks=(3, 4, 6, 8),
                thresholds=(0.30, 0.40, 0.50),
            )
        except Exception as exc:
            st.error(f"Retrieval ablation failed: {exc}")
    if ablation_cols[1].button(
        "Run built-in chunk-size / overlap ablation",
        disabled=not (use_builtin and examples),
        help="Re-indexes the small public benchmark corpus in temporary isolated collections.",
    ):
        try:
            lab_engine = RAGEngine(active_engine.store, active_engine.provider, experiment_config)
            st.session_state.gold_chunk_ablations = chunking_ablation(
                examples,
                lab_engine,
                chunk_sizes=(800, 1200, 1800, 2400),
                overlaps=(100, 220),
                mode=retrieval_mode,
                top_k=int(top_k),
                threshold=float(threshold),
            )
        except Exception as exc:
            st.error(f"Chunking ablation failed: {exc}")

    if st.session_state.gold_retrieval_ablations:
        st.markdown("#### Retrieval ablation ranking")
        st.dataframe(pd.DataFrame(st.session_state.gold_retrieval_ablations), hide_index=True, use_container_width=True)
    if st.session_state.gold_chunk_ablations:
        st.markdown("#### Chunking ablation ranking")
        st.dataframe(pd.DataFrame(st.session_state.gold_chunk_ablations), hide_index=True, use_container_width=True)

    st.markdown("### Full gold benchmark")
    if active_engine.store.count() == 0:
        st.info("Index the documents referenced by the gold set before running the full benchmark.")
    run_disabled = not examples or active_engine.store.count() == 0
    if st.button("Run full labelled benchmark", type="primary", disabled=run_disabled, use_container_width=True):
        selected = examples if int(case_limit) == 0 else examples[: int(case_limit)]
        try:
            provider = active_engine.provider if model_id.strip() == active_engine.provider.model else clone_provider_with_model(active_engine.provider, model_id)
            lab_engine = RAGEngine(active_engine.store, provider, experiment_config)
            traces = []
            progress = st.progress(0, text="Running labelled cases…")
            for index, example in enumerate(selected, start=1):
                traces.append(lab_engine.execute(example.query, example.history))
                progress.progress(index / len(selected), text=f"Case {index}/{len(selected)}")
            progress.empty()
            run = evaluate_run(
                selected,
                traces,
                variant=variant,
                answer_f1_threshold=float(answer_f1_threshold),
                latency_slo_seconds=float(latency_slo) if latency_slo > 0 else None,
                metadata={
                    "provider": provider.name,
                    "model": provider.model,
                    "similarity_threshold": experiment_config.similarity_threshold,
                    "top_k": experiment_config.top_k,
                    "retrieval_mode": experiment_config.retrieval_mode,
                    "hybrid_dense_weight": experiment_config.hybrid_dense_weight,
                    "rerank_weight": experiment_config.rerank_weight,
                    "dataset": "built-in-curated-synthetic" if use_builtin else "user-human-gold",
                },
            )
            run.metadata.update(run_metadata(run, traces))
            st.session_state.gold_benchmark_runs.append(run)
            st.success(f"Completed {len(selected)} labelled cases.")
        except UnicodeDecodeError:
            st.error("Gold file must be UTF-8 encoded.")
        except (GoldDataError, ValueError) as exc:
            st.error(str(exc))
        except ProviderError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")

    st.markdown("### P2 · Model comparison")
    comparison_models = st.text_area(
        "Model IDs (one per line)",
        value=active_engine.provider.model,
        help="Uses the already configured provider credentials. Only model IDs enabled for that provider will work.",
    )
    compare_limit = st.slider("Cases per model", 1, max(1, min(max_cases, 40)), min(10, max(1, max_cases))) if examples else 1
    if st.button("Run model comparison", disabled=run_disabled):
        models = [line.strip() for line in comparison_models.splitlines() if line.strip()]
        if len(models) < 2:
            st.warning("Enter at least two model IDs to run a model comparison.")
        else:
            selected = examples[:compare_limit]
            progress = st.progress(0, text="Running model comparison…")
            try:
                for model_index, candidate_model in enumerate(models, start=1):
                    provider = clone_provider_with_model(active_engine.provider, candidate_model)
                    lab_engine = RAGEngine(active_engine.store, provider, experiment_config)
                    traces = [lab_engine.execute(example.query, example.history) for example in selected]
                    run = evaluate_run(
                        selected,
                        traces,
                        variant=f"model::{candidate_model}",
                        answer_f1_threshold=float(answer_f1_threshold),
                        latency_slo_seconds=float(latency_slo) if latency_slo > 0 else None,
                        metadata={
                            "provider": provider.name,
                            "model": provider.model,
                            "similarity_threshold": experiment_config.similarity_threshold,
                            "top_k": experiment_config.top_k,
                            "retrieval_mode": experiment_config.retrieval_mode,
                            "dataset": "built-in-curated-synthetic" if use_builtin else "user-human-gold",
                        },
                    )
                    run.metadata.update(run_metadata(run, traces))
                    st.session_state.gold_benchmark_runs.append(run)
                    progress.progress(model_index / len(models), text=f"Model {model_index}/{len(models)}")
                progress.empty()
                st.success(f"Completed {len(models)} model variants × {len(selected)} cases.")
            except (ProviderError, ValueError) as exc:
                progress.empty()
                st.error(str(exc))

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
        numeric_cols = ["Accuracy", "Retrieval recall@K", "Citation F1", "Refusal F1"]
        st.line_chart(chart[numeric_cols])

        st.markdown("### Regression gate · previous run → latest run")
        gate = regression_gate(runs[-2], runs[-1])
        comparison = pd.DataFrame(gate["comparison"])
        if not comparison.empty:
            st.dataframe(comparison, hide_index=True, use_container_width=True)
        if gate["passed"]:
            st.success("Regression gate passed: no protected metric exceeded its allowed degradation tolerance.")
        else:
            st.error(
                "Regression gate failed: "
                + ", ".join(str(row["metric"]) for row in gate["failures"])
            )

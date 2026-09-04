"""VeriRAG Studio Streamlit entrypoint."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.config import Settings
from fastembed import TextEmbedding

from components.eval_dashboard import render_dashboard
from components.evidence_panel import render_evidence
from core.config import AppConfig, load_config
from core.ingestion import DocumentParseError, ingest_document
from core.models import QueryTrace
from core.providers import ProviderError, build_provider
from core.rag_engine import RAGEngine
from core.reporting import build_answer_pdf
from core.security import UploadValidationError
from core.vector_store import VectorStoreManager

st.set_page_config(
    page_title="VeriRAG Studio",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .hero {padding: 1.8rem 2rem; margin-bottom: 1.2rem; border: 1px solid #214158;
             border-radius: 22px; background: linear-gradient(125deg, #0d2131 0%, #113c41 100%);}
      .hero h1 {font-size: clamp(2rem, 4vw, 3.7rem); line-height: 1.03; margin: .45rem 0 .7rem;}
      .hero p {color: #b9d7d2; font-size: 1.05rem; margin: 0;}
      .eyebrow {color: #59e0c2; letter-spacing: .15em; font-size: .72rem; font-weight: 800;}
      [data-testid="stMetric"] {background: #102330; border: 1px solid #203e50; padding: .75rem;
                               border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedding_model(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.Client(Settings(anonymized_telemetry=False))


def initialize_state(config: AppConfig) -> None:
    defaults = {
        "session_id": uuid.uuid4().hex,
        "messages": [],
        "traces": [],
        "processed_hashes": set(),
        "ingestion_warnings": [],
        "persona": "Executive",
        "threshold": config.similarity_threshold,
        "top_k": config.top_k,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_store(config: AppConfig) -> VectorStoreManager:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager(
            get_chroma_client(),
            get_embedding_model(config.embedding_model),
            st.session_state.session_id,
        )
    return st.session_state.vector_store


def process_file(name: str, content: bytes, store: VectorStoreManager, config: AppConfig) -> tuple[bool, str]:
    result = ingest_document(
        content,
        name,
        max_file_bytes=config.max_file_bytes,
        max_pages=config.max_pages_per_file,
        chunk_size=config.chunk_size_chars,
        chunk_overlap=config.chunk_overlap_chars,
    )
    if store.contains_document(result.document_hash):
        return False, f"{result.filename} was already indexed in this session."
    if store.count() + len(result.chunks) > config.max_chunks_per_session:
        raise UploadValidationError(
            f"This upload would exceed the {config.max_chunks_per_session:,}-chunk session limit."
        )
    store.add_chunks(result.chunks)
    st.session_state.processed_hashes.add(result.document_hash)
    st.session_state.ingestion_warnings.extend(result.warnings)
    return True, f"Indexed {result.filename}: {result.pages} page(s), {len(result.chunks)} chunks."


def reset_session(store: VectorStoreManager) -> None:
    store.clear()
    st.session_state.messages = []
    st.session_state.traces = []
    st.session_state.processed_hashes = set()
    st.session_state.ingestion_warnings = []


def trace_download(trace: QueryTrace) -> str:
    payload = asdict(trace)
    return json.dumps(payload, indent=2, ensure_ascii=False)


config = load_config()
initialize_state(config)
store = get_store(config)
provider = build_provider(config)

st.markdown(
    """
    <div class="hero">
      <span class="eyebrow">ENTERPRISE DOCUMENT INTELLIGENCE</span>
      <h1>Ask the document. Inspect the proof.</h1>
      <p>Session-isolated retrieval, evidence-gated answers, and auditable source traces.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## VeriRAG Studio")
    st.caption("A portfolio-grade, bounded RAG reference implementation")
    st.session_state.persona = st.radio(
        "View",
        ["Executive", "Technical"],
        horizontal=True,
        help="Technical view reveals retrieval and model diagnostics.",
    )

    st.markdown("### Knowledge collection")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT, Markdown, or CSV",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help=f"Up to {config.max_files} files; {config.max_file_bytes // (1024 * 1024)} MB each.",
    )
    upload_disabled = not uploads or len(uploads) > config.max_files
    if uploads and len(uploads) > config.max_files:
        st.error(f"Select at most {config.max_files} files at a time.")
    if st.button(
        "Process selected files", type="primary", use_container_width=True, disabled=upload_disabled
    ):
        for uploaded in uploads or []:
            try:
                added, message = process_file(uploaded.name, uploaded.getvalue(), store, config)
                (st.success if added else st.info)(message)
            except (UploadValidationError, DocumentParseError) as exc:
                st.error(str(exc))
            except Exception:
                st.error(f"Could not index {uploaded.name}. The file may be malformed or unsupported.")

    sample_path = Path(__file__).parent / "sample_docs" / "retail_promotion_policy.txt"
    if st.button("Load sample policy", use_container_width=True):
        try:
            added, message = process_file(sample_path.name, sample_path.read_bytes(), store, config)
            (st.success if added else st.info)(message)
        except (UploadValidationError, DocumentParseError) as exc:
            st.error(str(exc))

    st.metric("Indexed chunks", f"{store.count():,}")
    names = store.document_names()
    if names:
        st.caption("Documents: " + ", ".join(names))

    with st.expander("Retrieval controls"):
        st.session_state.top_k = st.slider("Evidence chunks", 1, 8, int(st.session_state.top_k))
        st.session_state.threshold = st.slider(
            "Similarity gate",
            0.0,
            1.0,
            float(st.session_state.threshold),
            0.01,
            help="Calibrate this on a labelled evaluation set before production use.",
        )

    st.caption(f"Provider: {config.provider} · Model: {provider.model}")
    if st.button("Reset this session", use_container_width=True):
        reset_session(store)
        st.rerun()

runtime_config = replace(
    config,
    top_k=int(st.session_state.top_k),
    similarity_threshold=float(st.session_state.threshold),
)
engine = RAGEngine(store, provider, runtime_config)

chat_tab, diagnostics_tab, about_tab = st.tabs(["Ask & verify", "Diagnostics", "Architecture"])

with chat_tab:
    main_column, evidence_column = st.columns([1.65, 1], gap="large")
    with main_column:
        if not config.provider_ready:
            st.warning(
                "Generation is not configured. Add `GROQ_API_KEY` to Streamlit secrets or set "
                "`VERIRAG_PROVIDER=ollama` for local inference."
            )
        if not st.session_state.messages:
            st.info("Load the sample policy or upload a document, then ask a question grounded in it.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(
            "Ask a question about the indexed documents…",
            disabled=not config.provider_ready,
        )
        if prompt:
            previous = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving evidence and validating the answer…"):
                        trace = engine.execute(prompt, previous)
                    st.markdown(trace.answer)
                st.session_state.messages.append({"role": "assistant", "content": trace.answer})
                st.session_state.traces.append(trace)
                st.rerun()
            except ProviderError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The query could not be completed. Review provider configuration and try again.")

        if st.session_state.traces:
            latest: QueryTrace = st.session_state.traces[-1]
            status = "Safe refusal" if latest.is_refusal else f"{latest.confidence} confidence"
            st.caption(f"Outcome: {status} · {latest.total_ms / 1_000:.2f}s total")
            download_columns = st.columns(2)
            with download_columns[0]:
                st.download_button(
                    "Download answer PDF",
                    build_answer_pdf(latest),
                    file_name="verirag-answer-report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with download_columns[1]:
                st.download_button(
                    "Download trace JSON",
                    trace_download(latest),
                    file_name="verirag-query-trace.json",
                    mime="application/json",
                    use_container_width=True,
                )
            if st.session_state.persona == "Technical":
                with st.expander("Technical details", expanded=True):
                    st.json(
                        {
                            "standalone_query": latest.standalone_query,
                            "retrieval_ms": latest.retrieval_ms,
                            "generation_ms": latest.generation_ms,
                            "provider": latest.provider,
                            "model": latest.model,
                            "refusal_reason": latest.refusal_reason,
                            "citation_validation_error": latest.citation_validation_error,
                            "citation_repair_attempted": latest.citation_repair_attempted,
                            "metrics": latest.metrics,
                        }
                    )

    with evidence_column:
        if st.session_state.traces:
            render_evidence(st.session_state.traces[-1])
        else:
            st.subheader("Evidence")
            st.caption("Retrieved passages will appear here with source, page, chunk, and similarity.")

with diagnostics_tab:
    render_dashboard(st.session_state.traces, provider)

with about_tab:
    st.header("How the answer path is controlled")
    st.markdown(
        """
        1. Files are validated, normalized, split at semantic boundaries, and assigned deterministic IDs.
        2. A session-specific Chroma collection prevents document mixing between visitors.
        3. Retrieval must cross the configured cosine-similarity gate before generation is allowed.
        4. Document text is fenced as untrusted evidence and cannot redefine system instructions.
        5. Generated citations are normalized and validated; one bounded repair is attempted before a safe refusal.
        6. Every completed query retains a local trace with evidence and latency diagnostics.

        This design reduces unsupported answers; it does not claim that any probabilistic model can guarantee zero hallucinations.
        """
    )

if st.session_state.ingestion_warnings:
    with st.expander("Ingestion warnings"):
        for warning in st.session_state.ingestion_warnings:
            st.warning(warning)

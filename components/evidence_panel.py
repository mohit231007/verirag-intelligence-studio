"""Evidence rendering kept separate from application state orchestration."""

from __future__ import annotations

import streamlit as st

from core.models import QueryTrace


def render_evidence(trace: QueryTrace) -> None:
    st.subheader("Evidence")
    if not trace.retrieved:
        st.caption("No passages crossed the evidence gate.")
        return

    for source_number, item in enumerate(trace.retrieved, start=1):
        label = (
            f"S{source_number} · {item.chunk.source_doc} · "
            f"page {item.chunk.page_number} · {item.similarity:.1%}"
        )
        with st.expander(label, expanded=source_number == 1):
            st.markdown(item.chunk.text)
            st.caption(f"Chunk `{item.chunk.chunk_id}` · rank {item.rank}")

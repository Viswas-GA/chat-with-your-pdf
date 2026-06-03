"""
Chat with Your PDF — beginner RAG app.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag.pdf_processor import process_pdf
from rag.qa_chain import answer_question
from rag.vector_store import VectorStore
from rag.visualize import build_embedding_figure

load_dotenv()

st.set_page_config(page_title="Chat with Your PDF", page_icon="📄", layout="wide")

st.title("📄 Chat with Your PDF")
st.caption("Upload a PDF, ask questions in plain English — RAG finds relevant sections and answers.")

# --- Sidebar: learning map + settings ---
with st.sidebar:
    st.header("How RAG works here")
    st.markdown(
        """
        1. **Load PDF** — extract text per page  
        2. **Chunk** — split into overlapping pieces  
        3. **Embed** — turn chunks into vectors (local model)  
        4. **Store** — save vectors in ChromaDB  
        5. **Retrieve** — find chunks similar to your question  
        6. **Generate** — LLM answers using those chunks only  
        """
    )
    st.divider()
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)
    show_sources = st.checkbox("Show source chunks", value=True)

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI mode: set `OPENAI_API_KEY` in `.env` (paid).")
    else:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        st.success(f"Free mode: Ollama (`{model}`)")
        st.caption("Install [Ollama](https://ollama.com/download), then run `ollama pull " + model + "`")

# --- Session state ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_name" not in st.session_state:
    st.session_state.indexed_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "last_retrieved_ids" not in st.session_state:
    st.session_state.last_retrieved_ids = set()


def _render_chat(top_k: int, show_sources: bool) -> None:
    st.subheader(f"Chatting with: {st.session_state.indexed_name}")
    st.caption(f"{st.session_state.chunk_count} chunks in the vector store")

    question = st.chat_input("Ask a question about your PDF…")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for src in msg["sources"]:
                        st.markdown(f"**Page {src['page']}** (similarity distance: {src['distance']:.3f})")
                        st.text(src["text"][:500] + ("…" if len(src["text"]) > 500 else ""))

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant sections…"):
                hits = st.session_state.vector_store.search(question, top_k=top_k)

            st.session_state.last_retrieved_ids = {
                str(h["chunk_id"]) for h in hits if h.get("chunk_id") is not None
            }

            try:
                with st.spinner("Generating answer…"):
                    answer = answer_question(question, hits)
            except ValueError as exc:
                answer = f"⚠️ {exc}"
                hits = []

            st.markdown(answer)

            if show_sources and hits:
                with st.expander("Sources used"):
                    for hit in hits:
                        st.markdown(f"**Page {hit['page']}** (distance: {hit['distance']:.3f})")
                        st.text(hit["text"][:500] + ("…" if len(hit["text"]) > 500 else ""))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": hits if show_sources else None,
            }
        )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_retrieved_ids = set()
        st.rerun()


# --- Upload & index ---
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    if st.session_state.indexed_name != uploaded.name:
        with st.spinner("Indexing PDF (chunk → embed → store)…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                chunks = process_pdf(tmp_path)
                store = VectorStore()
                count = store.add_chunks(chunks)
                st.session_state.vector_store = store
                st.session_state.indexed_name = uploaded.name
                st.session_state.chunk_count = count
                st.success(f"Indexed **{uploaded.name}** — {count} chunks ready.")
            except ValueError as exc:
                st.error(str(exc))
            finally:
                os.unlink(tmp_path)

# --- Chat ---
if st.session_state.vector_store is None:
    st.info("Upload a PDF to get started.")
else:
    tab_chat, tab_map = st.tabs(["💬 Chat", "🗺️ Embedding map"])

    with tab_map:
        st.markdown(
            "Each dot is one **chunk** stored in ChromaDB (384 numbers squashed to 2D). "
            "**Closer dots** ≈ similar meaning. Color = PDF page."
        )
        method = st.radio(
            "Projection method",
            ["pca", "tsne"],
            horizontal=True,
            format_func=lambda x: "PCA (fast)" if x == "pca" else "t-SNE (slower, nicer clusters)",
        )
        points = st.session_state.vector_store.get_all_points()
        fig = build_embedding_figure(
            points,
            method=method,
            highlight_ids=st.session_state.last_retrieved_ids,
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.session_state.last_retrieved_ids:
            st.caption("🔴 Larger red dots = chunks retrieved for your last question.")
        st.caption(f"{len(points)} vectors in ChromaDB · hover a dot to read the chunk text.")

    with tab_chat:
        _render_chat(top_k, show_sources)


"""Step 4 of RAG: store chunk embeddings in a vector database (ChromaDB)."""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

from rag.embeddings import embed_texts
from rag.pdf_processor import Chunk


class VectorStore:
    """In-memory Chroma collection for one uploaded PDF session."""

    def __init__(self, collection_name: str | None = None) -> None:
        name = collection_name or f"pdf_{uuid.uuid4().hex[:8]}"
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed all chunks and store them. Returns number of chunks indexed."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        self._collection.add(
            ids=[str(c.chunk_id) for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"page": c.page, "chunk_id": c.chunk_id}
                for c in chunks
            ],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """
        Step 5 (retrieval): find chunks most similar to the user's question.

        Returns dicts with keys: text, page, distance.
        """
        query_embedding = embed_texts([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        if not results["documents"] or not results["documents"][0]:
            return hits

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "page": meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "distance": dist,
                }
            )
        return hits

    def get_all_points(self) -> list[dict[str, Any]]:
        """Return every stored chunk with its embedding (for visualization)."""
        results = self._collection.get(
            include=["embeddings", "documents", "metadatas"],
        )

        points: list[dict[str, Any]] = []
        ids = results.get("ids")
        embeddings = results.get("embeddings")
        documents = results.get("documents")
        metadatas = results.get("metadatas")

        if not ids or embeddings is None:
            return points

        if documents is None:
            documents = [""] * len(ids)
        if metadatas is None:
            metadatas = [{}] * len(ids)

        for point_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            if emb is None:
                continue
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            points.append(
                {
                    "id": point_id,
                    "embedding": emb,
                    "text": doc or "",
                    "page": (meta or {}).get("page", "?"),
                    "chunk_id": (meta or {}).get("chunk_id", point_id),
                }
            )
        return points

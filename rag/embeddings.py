"""Step 3 of RAG: turn text chunks into vectors (embeddings)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load a small, free embedding model (runs locally, no API key).

    all-MiniLM-L6-v2 maps sentences to 384-dimensional vectors.
    Similar meaning → vectors close together in space.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into embedding vectors."""
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()

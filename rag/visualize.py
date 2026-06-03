"""Visualize ChromaDB embeddings as an interactive 2D map (for learning RAG)."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_to_2d(embeddings: list[list[float]], method: str = "pca") -> np.ndarray:
    """
    Each chunk is a 384-number vector — humans can't see that.
    PCA / t-SNE squashes it to 2D (x, y) for plotting.
    Nearby points ≈ similar meaning.
    """
    X = np.array(embeddings, dtype=float)
    if len(X) == 0:
        return np.empty((0, 2))
    if len(X) == 1:
        return np.array([[0.0, 0.0]])

    n_components = 2
    if method == "tsne":
        perplexity = min(30, len(X) - 1)
        if perplexity >= 1:
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42,
                init="pca",
            )
            return reducer.fit_transform(X)

    reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)


def build_embedding_figure(
    points: list[dict[str, Any]],
    method: str = "pca",
    highlight_ids: set[str] | None = None,
) -> go.Figure:
    """Build a Plotly scatter plot of chunk embeddings."""
    if not points:
        fig = go.Figure()
        fig.update_layout(title="No embeddings to show")
        return fig

    embeddings = [p["embedding"] for p in points]
    coords = reduce_to_2d(embeddings, method=method)

    labels = []
    for p in points:
        preview = p["text"][:80].replace("\n", " ")
        if len(p["text"]) > 80:
            preview += "…"
        labels.append(f"Page {p['page']} · chunk {p['chunk_id']}<br>{preview}")

    sizes = [14 if p["id"] in (highlight_ids or set()) else 8 for p in points]

    df_points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "page": f"Page {p['page']}",
            "label": labels[i],
            "size": sizes[i],
            "highlight": p["id"] in (highlight_ids or set()),
        }
        for i, p in enumerate(points)
    ]

    import pandas as pd

    df = pd.DataFrame(df_points)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="page",
        hover_name="label",
        size="size",
        size_max=18,
        title=f"Embedding map ({method.upper()}) — closer dots ≈ more similar text",
        labels={"x": "Dimension 1", "y": "Dimension 2"},
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig.update_layout(
        height=520,
        legend=dict(title="PDF page"),
        hovermode="closest",
    )
    return fig

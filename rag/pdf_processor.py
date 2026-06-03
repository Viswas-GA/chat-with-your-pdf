"""Step 1 & 2 of RAG: load a PDF and split it into searchable chunks."""

from __future__ import annotations

from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    """One piece of text from the PDF, with its page number for citations."""

    text: str
    page: int
    chunk_id: int


def load_pdf_text(file_path: str) -> list[tuple[int, str]]:
    """Read every page; return (page_number, text) pairs (1-based pages)."""
    reader = PdfReader(file_path)
    pages: list[tuple[int, str]] = []

    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((index, text))

    return pages


def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Split pages into overlapping chunks.

    - chunk_size: target characters per chunk (roughly ~150–200 tokens).
    - chunk_overlap: shared text between chunks so sentences at boundaries
      are not lost during retrieval.
    """
    chunks: list[Chunk] = []
    chunk_id = 0

    for page_num, page_text in pages:
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            piece = page_text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, chunk_id=chunk_id))
                chunk_id += 1
            if end >= len(page_text):
                break
            start = end - chunk_overlap

    return chunks


def process_pdf(file_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """Full PDF → chunks pipeline."""
    pages = load_pdf_text(file_path)
    if not pages:
        raise ValueError("No text could be extracted from this PDF. It may be scanned/image-only.")
    return chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

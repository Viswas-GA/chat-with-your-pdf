"""Step 6 of RAG: send retrieved chunks + question to an LLM for the final answer."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context from a PDF document.

Rules:
- Answer based on the context. If the context does not contain enough information, say so clearly.
- Be concise and accurate.
- When relevant, mention which page the information came from (page numbers are in the context headers).
- Do not invent facts not supported by the context."""


def build_context_block(retrieved: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    parts: list[str] = []
    for i, hit in enumerate(retrieved, start=1):
        page = hit.get("page", "?")
        parts.append(f"--- Context {i} (page {page}) ---\n{hit['text']}")
    return "\n\n".join(parts)


def _ollama_host() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return base.rstrip("/").removesuffix("/v1")


def check_ollama_running() -> None:
    """Raise a clear error if Ollama is not reachable."""
    url = f"{_ollama_host()}/api/tags"
    try:
        urllib.request.urlopen(url, timeout=3)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(
            "Ollama is not running. This project uses Ollama for free local answers.\n\n"
            "1. Install: https://ollama.com/download\n"
            "2. In a terminal run: ollama pull llama3.2\n"
            "3. Keep Ollama open, then try again."
        ) from exc


def get_llm_client_and_model() -> tuple[OpenAI, str]:
    """
    Default: Ollama (100% free, runs on your PC, no API key).

    Optional: set LLM_PROVIDER=openai and OPENAI_API_KEY in .env (paid).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is missing. "
                "Use Ollama instead (free): set LLM_PROVIDER=ollama in .env"
            )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAI(api_key=api_key), model

    check_ollama_running()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    # Ollama does not need a real key; the client requires some string.
    return OpenAI(base_url=base_url, api_key="ollama"), model


def answer_question(
    question: str,
    retrieved: list[dict[str, Any]],
    model: str | None = None,
) -> str:
    """Augmented generation: question + retrieved context → LLM answer."""
    if not retrieved:
        return "No relevant passages were found in the document. Try rephrasing your question."

    context = build_context_block(retrieved)
    client, default_model = get_llm_client_and_model()
    chat_model = model or default_model

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context from the PDF:\n\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer using only the context above:"
                    ),
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:
        err = str(exc).lower()
        if "connection" in err or "refused" in err:
            raise ValueError(
                f"Could not reach the LLM ({chat_model}). "
                "If using Ollama, run: ollama pull " + chat_model
            ) from exc
        if "not found" in err or "404" in err:
            raise ValueError(
                f"Model '{chat_model}' is not installed. Run: ollama pull {chat_model}"
            ) from exc
        raise

    return response.choices[0].message.content or ""

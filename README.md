# Chat with Your PDF

A beginner-friendly **RAG** (Retrieval-Augmented Generation) project. Upload a PDF, ask questions in plain English, and get answers grounded in the document.

## What you'll learn

| RAG step | What happens | File |
|----------|--------------|------|
| 1. Load | Extract text from each PDF page | `rag/pdf_processor.py` |
| 2. Chunk | Split text into overlapping pieces | `rag/pdf_processor.py` |
| 3. Embed | Convert chunks to vectors | `rag/embeddings.py` |
| 4. Store | Save vectors in ChromaDB | `rag/vector_store.py` |
| 5. Retrieve | Find chunks similar to your question | `rag/vector_store.py` |
| 6. Generate | LLM answers using retrieved context | `rag/qa_chain.py` |

## Quick start

### 1. Create a virtual environment

```powershell
cd d:\Rag-chatwithpdf
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB).

### 2. Install Ollama (free — no OpenAI account)

This app uses **Ollama** for answers: runs on your PC, no API key, no payment.

1. Download and install: https://ollama.com/download  
2. Open a terminal and pull a small model (one time, ~2 GB):

```powershell
ollama pull llama3.2
```

3. Optional config:

```powershell
copy .env.example .env
```

Default `.env` already sets `LLM_PROVIDER=ollama`. You do **not** need https://platform.openai.com/api-keys unless you choose paid OpenAI later.

**What is free in this project**

| Part | Cost |
|------|------|
| PDF chunking + embeddings | Free (local) |
| Vector search (ChromaDB) | Free (local) |
| Answers (Ollama) | Free (local) |
| OpenAI (optional) | Paid — only if you set `LLM_PROVIDER=openai` |

### 3. Run the app

```powershell
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

### 4. Try it

1. Upload a text-based PDF (e.g. a tutorial, resume, or paper).
2. Wait for "Indexed … chunks ready."
3. Ask: *"What is a list comprehension?"* (if your PDF is a Python tutorial).

## Project structure

```
Rag-chatwithpdf/
├── app.py                 # Streamlit UI
├── rag/
│   ├── pdf_processor.py   # Load + chunk PDF
│   ├── embeddings.py      # Local sentence embeddings
│   ├── vector_store.py    # ChromaDB search
│   └── qa_chain.py        # LLM with context
├── requirements.txt
├── .env.example
└── README.md
```

## Example RAG flow

```
Question: "What is a list comprehension?"
        ↓
Embed the question → vector
        ↓
Search ChromaDB for nearest chunk vectors
        ↓
Top 4 chunks (with page numbers) → prompt context
        ↓
Ollama (or optional GPT) reads context + question → answer
```

## Tips for learning

- **Change `chunk_size`** in `process_pdf()` and see how answers change.
- **Change `top_k`** in the sidebar — more chunks = more context but more noise.
- **Read "Sources used"** — see exactly which PDF passages drove the answer.
- **Scanned PDFs** won't work well; you need selectable text (or OCR later).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No text could be extracted" | PDF is image-only; use a text PDF or add OCR later |
| "Ollama is not running" | Install Ollama, run `ollama pull llama3.2`, keep app open |
| "Model … not installed" | Run `ollama pull llama3.2` (or your `OLLAMA_MODEL`) |
| "OPENAI_API_KEY is not set" | Only if using OpenAI; otherwise use `LLM_PROVIDER=ollama` |
| Slow first question | Embedding model loads once; later queries are faster |
| Wrong answers | Try rephrasing; increase `top_k`; use a cleaner PDF |

## Next steps (when you're ready)

- Persist ChromaDB to disk instead of in-memory
- Add conversation memory (multi-turn chat)
- Support Ollama for a fully local LLM
- Add OCR for scanned PDFs (`pytesseract` + `pdf2image`)

Happy learning!

<img width="1919" height="901" alt="image" src="https://github.com/user-attachments/assets/ca73de9e-647b-411c-84ad-8b4f8c1bbc66" />
<img width="1439" height="841" alt="image" src="https://github.com/user-attachments/assets/bd8c8180-5095-44cc-8967-0f2f24225de2" />



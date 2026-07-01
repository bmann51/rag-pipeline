# StackAI RAG Assessment

Implementation of a Retrieval-Augmented Generation (RAG) system for the StackAI Forward Deployed Engineer take-home assessment.

## Features

- PDF ingestion with page-aware chunking
- Semantic and keyword retrieval
- Mistral LLM integration
- FastAPI backend
- Simple chat UI

## Implemented So Far: PDF Ingestion

The ingestion backend is implemented with FastAPI and stores parsed data locally (no vector DB).

### Endpoint

- Method: `POST`
- Path: `/ingestion/pdfs`
- Payload: `multipart/form-data` with one or more `files`
- Accepted type: `.pdf` only
- Response groups files into `ingested`, `skipped` (for duplicates), and `failed`
- Reset endpoint: `DELETE /ingestion/reset` clears uploaded files and JSONL records

### Ingestion Flow

1. Validate files (extension and max size limit).
2. Parse PDF pages with `pypdf`.
3. Extract page text and normalize whitespace.
4. Chunk text with overlap using a page-aware algorithm.
5. Persist:
	 - Original file in `data/uploads/`
	 - Document metadata in `data/records/documents.jsonl`
	 - Chunk records in `data/records/chunks.jsonl`

### OCR Fallback For Scanned/Fuzzy PDFs

If native PDF extraction returns no text (common for scanned or low-quality PDFs), ingestion falls back to Mistral OCR.

- Trigger condition: all pages are empty after `pypdf` extraction.
- OCR provider: Mistral OCR model (default `mistral-ocr-latest`).
- Required env var: `MISTRAL_API_KEY`.

Optional env vars:

- `OCR_FALLBACK_ENABLED=true|false` (default `true`)
- `MISTRAL_OCR_MODEL` (default `mistral-ocr-latest`)

### Chunking Considerations

The chunking algorithm is custom and designed for retrieval quality while keeping implementation simple:

- Uses paragraph-first packing to preserve local context.
- Splits oversized text near sentence boundaries (`.`, `!`, `?`, newline) when possible.
- Falls back to whitespace or hard cuts if needed.
- Adds configurable overlap to reduce boundary information loss.
- Drops very small fragments by default (`min_chunk_chars`) to avoid noisy retrieval units.
- Keeps page provenance (`page_start`, `page_end`) for citation-friendly downstream results.

Default parameters are configurable in `app/config.py`:

- `chunk_size_chars=1200`
- `chunk_overlap_chars=180`
- `min_chunk_chars=250`

### Why This Strategy

- Better than fixed-size raw slicing for preserving semantic coherence.
- Lightweight and deterministic (good for debugging and explainability).
- Compatible with both keyword and semantic retrieval in later pipeline stages.

### File Layout

- API entrypoint: `app/main.py`
- PDF extraction: `app/ingestion/pdf_ingestor.py`
- Chunking logic: `app/ingestion/chunker.py`
- Local persistence: `app/storage/document_store.py`
- Request/response schemas: `app/schemas.py`

## Running (uv Native)

1. Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

2. Install dependencies from lockfile/project metadata:

```bash
uv sync --active
```

3. Run the API through uv:

```bash
uv run --active uvicorn app.main:app --reload
```

4. Test ingestion:

```bash
curl -X POST "http://127.0.0.1:8000/ingestion/pdfs" \
	-F "files=@/absolute/path/to/file1.pdf" \
	-F "files=@/absolute/path/to/file2.pdf"

# Optional: clear all ingested data
curl -X DELETE "http://127.0.0.1:8000/ingestion/reset"
```

## Notes

- Current implementation focuses on ingestion only.
- Query transformation, retrieval fusion, reranking, generation, UI, and safety policies are next.
- Dependencies are managed with `uv` via `pyproject.toml` and `uv.lock`.


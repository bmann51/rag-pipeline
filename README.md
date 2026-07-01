# StackAI RAG Assessment

Backend implementation for PDF ingestion and hybrid retrieval (keyword + semantic) for the StackAI FDE take-home assessment.

## Current Scope

- FastAPI backend only
- PDF ingestion with dedupe and OCR fallback
- Local JSONL storage (no vector database)
- Intent-gated query processing
- Hybrid retrieval: BM25-style keyword + embedding semantic search
- Score fusion and conservative reranking tie-breaker
- Embedding persistence and warmup endpoint

Out of scope:

- Chat UI
- Generation/answer synthesis

## API Endpoints

- `GET /health`
- `POST /ingestion/pdfs`
- `DELETE /ingestion/reset`
- `POST /query`
- `POST /embeddings/warmup`

## Ingestion

`POST /ingestion/pdfs`

- Accepts `multipart/form-data` with one or more `files`
- Accepts `.pdf` only
- Validates max file size
- Deduplicates by SHA-256
- Extracts text with `pypdf`
- Falls back to Mistral OCR when extraction is empty
- Chunks text with overlap and page provenance
- Persists:
  - Uploaded files in `data/uploads/`
  - Documents in `data/records/documents.jsonl`
  - Chunks in `data/records/chunks.jsonl`

`DELETE /ingestion/reset`

- Clears uploads, documents, chunks, and cached embeddings

## Retrieval Pipeline

`POST /query`

1. Normalize and classify intent.
2. Skip retrieval for non-search conversational input.
3. Rewrite query when useful (filler stripping, acronym expansion, topic extraction).
4. Run keyword retrieval across query variants.
5. Run semantic retrieval with cached chunk embeddings.
6. Fuse normalized keyword and semantic scores.
7. Apply conservative short-query source-consistency tie-breaker when conditions are met.
8. Apply evidence gating:
   - relevance threshold
   - query-term coverage threshold

Response statuses currently used:

- `search_not_required`
- `retrieval_complete`
- `insufficient_evidence`

Diagnostics include:

- `intent`
- `reason`
- `rewrite_notes`
- `topic_query`
- `retrieval_queries`

## Embeddings and Rate Limits

Semantic retrieval uses Mistral embeddings.

Required:

- `MISTRAL_API_KEY`

Optional:

- `MISTRAL_EMBEDDING_MODEL` (default `mistral-embed`)

Persistence:

- Chunk embeddings are stored in `data/records/chunk_embeddings.jsonl`
- Missing chunk embeddings are generated and appended during query-time retrieval

Warmup endpoint:

- `POST /embeddings/warmup`
- Optional query params:
  - `force_rebuild=true`
  - `max_chunks=<N>`

Rate-limit resilience:

- Request pacing (`embedding_min_request_interval_seconds`)
- 429 retries with exponential backoff (`embedding_max_retries_on_rate_limit`, `embedding_retry_base_delay_seconds`)

## Run Locally (uv)

1. Create and activate environment:

```bash
uv venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
uv sync --active
```

3. Start API:

```bash
uv run --active uvicorn app.main:app --reload
```

## Quick Test Commands

Ingest:

```bash
curl -X POST "http://127.0.0.1:8000/ingestion/pdfs" \
  -F "files=@/absolute/path/to/file1.pdf" \
  -F "files=@/absolute/path/to/file2.pdf"
```

Warm embeddings:

```bash
curl -X POST "http://127.0.0.1:8000/embeddings/warmup"
```

Query examples:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"hello"}'

curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"gradient descent","top_k":7}'

curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the document say about gradient descent?","top_k":7}'
```

Reset:

```bash
curl -X DELETE "http://127.0.0.1:8000/ingestion/reset"
```

## Key Files

- API orchestration: `app/main.py`
- Query processing: `app/retrieval/query_processor.py`
- Keyword retrieval: `app/retrieval/keyword_search.py`
- Semantic retrieval: `app/retrieval/semantic_search.py`
- Embedding client: `app/retrieval/embeddings.py`
- PDF ingestion: `app/ingestion/pdf_ingestor.py`
- Chunking: `app/ingestion/chunker.py`
- Document store: `app/storage/document_store.py`
- Embedding store: `app/storage/embedding_store.py`
- Schemas: `app/schemas.py`
- Settings: `app/config.py`


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
- Optional grounded answer generation with Mistral

Out of scope:

- Chat UI
- Multi-turn chat memory/agentic workflows

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
9. Optionally generate a grounded answer with citations from retrieved chunks.

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

When generation is enabled and succeeds, query responses also include:

- `generated_answer`
- `cited_chunk_ids`

## Embeddings and Rate Limits

Semantic retrieval uses Mistral embeddings.

Required:

- `MISTRAL_API_KEY`

Optional:

- `MISTRAL_EMBEDDING_MODEL` (default `mistral-embed`)
- `MISTRAL_CHAT_MODEL` (default `mistral-small-latest`)

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

Generation behavior:

- Disabled by default (`generation_enabled=false`).
- Runs only for `retrieval_complete` responses.
- Skips generation when retrieval confidence is too low.
- Falls back to retrieval-only output if generation fails (non-fatal).

## Validation Status (Current)

What has been validated so far:

- Retrieval pipeline is stable across intent-gated, in-domain, and out-of-domain queries.
- Hybrid retrieval (keyword + semantic + fusion) is returning expected `status` values.
- Embedding persistence and warmup are working (`chunk_embeddings.jsonl` reuse verified).
- Optional generation path is active and returns grounded answers with chunk citations on many factual queries.
- Citation IDs in responses are validated against retrieved chunk IDs.

Known gaps observed:

- Answer generation is not yet fully consistent on all answerable questions.
- Some answerable prompts still return retrieval evidence but no generated answer when citation formatting from the model is weak.
- For ambiguous short phrases, top-source selection can occasionally drift across documents depending on wording.

## Remaining Testing (Next Round)

When additional PDFs/domains are loaded, run these checks:

1. Corpus coverage checks
  - Build a small answer key per new document (10-20 factual questions with expected phrases).
  - Measure: status accuracy, evidence hit rate, answer hit rate, citation validity.
2. Cross-document ambiguity checks
  - Test short overlapping terms that may appear in multiple documents.
  - Confirm top results come from intended source when query context implies one document/domain.
3. Citation robustness checks
  - Verify generated answers contain valid `cited_chunk_ids` mapped to returned chunks.
  - Track how often generation is skipped due to missing/invalid citations.
4. Out-of-corpus safety checks
  - Validate that unsupported questions continue to return `insufficient_evidence`.
5. Scale/performance checks
  - Re-run warmup + query batches after adding files to confirm no rate-limit regressions.
  - Confirm generation remains non-fatal under transient model/API failures.

Suggested acceptance bar for next pass:

- `status` correctness: >= 95%
- evidence contains expected facts (for answerable set): >= 90%
- generated answer contains expected facts (for answerable set): >= 80%
- citation validity (generated answers): 100%

Reusable scorecard command:

```bash
/Users/brianmann/git/stackai/.venv/bin/python scripts/qa_scorecard.py
```

Optional:

- `--cases path/to/cases.json` to run a custom answer-key set
- `--save reports/qa_scorecard.json` to save full results

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


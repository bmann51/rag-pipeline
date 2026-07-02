from uuid import uuid4
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.ingestion.pdf_ingestor import build_chunks, extract_pdf_pages
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.keyword_search import KeywordSearcher
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.semantic_search import SemanticSearcher
from app.retrieval.answer_generator import AnswerGenerator
from app.schemas import (
    ClearIngestionResponse,
    ChunkRecord,
    DocumentRecord,
    FileIngestionError,
    FileIngestionResult,
    FileIngestionSkipped,
    IngestionResponse,
    QueryDiagnostics,
    RetrievedChunk,
    QueryRequest,
    QueryResponse,
)
from app.storage.chunk_reader import ChunkReader
from app.storage.document_store import DocumentStore
from app.storage.embedding_store import EmbeddingStore

settings = get_settings()
store = DocumentStore(upload_dir=settings.upload_dir, record_dir=settings.record_dir)
chunk_reader = ChunkReader(record_dir=settings.record_dir)
embedding_store = EmbeddingStore(record_dir=settings.record_dir)
keyword_searcher = KeywordSearcher()
embedding_client = EmbeddingClient(
    api_key=settings.mistral_api_key,
    model=settings.mistral_embedding_model,
    batch_size=settings.embedding_batch_size,
    min_request_interval_seconds=settings.embedding_min_request_interval_seconds,
    max_retries_on_rate_limit=settings.embedding_max_retries_on_rate_limit,
    retry_base_delay_seconds=settings.embedding_retry_base_delay_seconds,
)
semantic_searcher = SemanticSearcher(embedding_client)
query_processor = QueryProcessor()
answer_generator = AnswerGenerator(
    api_key=settings.mistral_api_key,
    model=settings.mistral_chat_model,
    temperature=settings.generation_temperature,
    max_tokens=settings.generation_max_tokens,
    max_chunks=settings.generation_max_chunks,
    max_chars_per_chunk=settings.generation_max_chars_per_chunk,
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
ui_dir = Path(__file__).resolve().parent.parent / "ui"
app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    min_score = min(raw_scores.values())
    if max_score == min_score:
        return {key: 1.0 for key in raw_scores}

    scale = max_score - min_score
    return {key: (value - min_score) / scale for key, value in raw_scores.items()}


def _query_term_coverage(query_terms: set[str], chunk_text: str) -> float:
    if not query_terms:
        return 0.0

    chunk_terms = set(KeywordSearcher.tokenize(chunk_text))
    matched = query_terms.intersection(chunk_terms)
    return len(matched) / len(query_terms)


def _collect_keyword_hits(
    queries: list[str],
    *,
    chunks: list[ChunkRecord],
    top_k: int,
) -> dict[str, tuple[ChunkRecord, float]]:
    best_hits: dict[str, tuple[ChunkRecord, float]] = {}
    for query in queries:
        for hit in keyword_searcher.search(query, chunks, top_k=top_k):
            existing = best_hits.get(hit.chunk.chunk_id)
            if existing is None or hit.score > existing[1]:
                best_hits[hit.chunk.chunk_id] = (hit.chunk, hit.score)
    return best_hits


def _apply_source_consistency_bonus(
    *,
    fused_scores: dict[str, float],
    by_chunk_id: dict[str, ChunkRecord],
    query_term_count: int,
) -> str | None:
    if not settings.source_consistency_bonus_enabled:
        return None
    if query_term_count == 0 or query_term_count > settings.source_consistency_max_query_terms:
        return None

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    if len(ranked_ids) < 2:
        return None

    window_size = min(settings.source_consistency_top_window, len(ranked_ids))
    window_ids = ranked_ids[:window_size]
    source_counts = Counter(by_chunk_id[chunk_id].source_file for chunk_id in window_ids)
    if len(source_counts) <= 1:
        return None

    dominant_source, dominant_count = source_counts.most_common(1)[0]
    share = dominant_count / window_size
    if share < settings.source_consistency_min_share:
        return None

    for chunk_id in window_ids:
        if by_chunk_id[chunk_id].source_file == dominant_source:
            fused_scores[chunk_id] += settings.source_consistency_bonus

    return dominant_source


@app.on_event("startup")
def startup() -> None:
    store.ensure_dirs()
    embedding_store.ensure_file()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embeddings/warmup")
def warmup_embeddings(force_rebuild: bool = False, max_chunks: int | None = None) -> dict[str, int | bool | str]:
    if not settings.semantic_search_enabled:
        raise HTTPException(status_code=400, detail="Semantic search is disabled.")

    chunks = chunk_reader.load_chunks()
    if not chunks:
        return {
            "status": "no_chunks",
            "force_rebuild": force_rebuild,
            "total_chunks": 0,
            "existing_embeddings": 0,
            "embedded_now": 0,
            "cached_after": 0,
        }

    if max_chunks is not None and max_chunks < 1:
        raise HTTPException(status_code=400, detail="max_chunks must be >= 1 when provided.")

    if force_rebuild:
        embedding_store.embeddings_file.write_text("", encoding="utf-8")
        cached_embeddings: dict[str, list[float]] = {}
    else:
        cached_embeddings = embedding_store.load_embeddings()

    missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in cached_embeddings]
    if max_chunks is not None:
        missing_chunks = missing_chunks[:max_chunks]

    if not missing_chunks:
        return {
            "status": "already_warm",
            "force_rebuild": force_rebuild,
            "total_chunks": len(chunks),
            "existing_embeddings": len(cached_embeddings),
            "embedded_now": 0,
            "cached_after": len(cached_embeddings),
        }

    vectors = embedding_client.embed_texts([chunk.text for chunk in missing_chunks])
    new_embeddings = {
        chunk.chunk_id: vector for chunk, vector in zip(missing_chunks, vectors)
    }
    embedding_store.append_embeddings(new_embeddings)

    return {
        "status": "warmup_complete",
        "force_rebuild": force_rebuild,
        "total_chunks": len(chunks),
        "existing_embeddings": len(cached_embeddings),
        "embedded_now": len(new_embeddings),
        "cached_after": len(cached_embeddings) + len(new_embeddings),
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    original_query = request.query
    chunks = chunk_reader.load_chunks()
    processed = query_processor.process_query(
        original_query,
        chunks=chunks,
        rewrite_enabled=settings.query_rewrite_enabled,
        intent_gate_enabled=settings.intent_gate_enabled,
    )

    normalized_query = processed.normalized_query

    if len(normalized_query) < settings.query_min_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Query must be at least {settings.query_min_chars} characters after normalization.",
        )

    if len(normalized_query) > settings.query_max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Query must be at most {settings.query_max_chars} characters.",
        )

    processed_query = processed.processed_query
    topic_query = processed.topic_query
    retrieval_queries = processed.retrieval_queries or [processed_query]
    rewrite_applied = processed.rewrite_applied
    policy_flag = processed.policy_flag
    answer_intent = processed.answer_intent

    _DISCLAIMERS = {
        "legal_topic": (
            "This response is for informational purposes only and does not constitute legal advice. "
            "Consult a qualified attorney for guidance specific to your situation."
        ),
        "medical_topic": (
            "This response is for informational purposes only and does not constitute medical advice. "
            "Consult a qualified healthcare professional for guidance specific to your situation."
        ),
    }

    if policy_flag == "pii_detected":
        diagnostics = QueryDiagnostics(
            intent=processed.intent,
            search_required=False,
            rewrite_applied=rewrite_applied,
            reason="Query refused: possible personally identifiable information detected. Please rephrase without personal data.",
            rewrite_notes=processed.rewrite_notes,
            topic_query=topic_query,
            retrieval_queries=[],
            policy_flag=policy_flag,
            answer_intent=answer_intent,
        )
        return QueryResponse(
            original_query=original_query,
            processed_query=processed_query,
            top_k=request.top_k or settings.query_top_k,
            status="query_refused",
            diagnostics=diagnostics,
            retrieved_chunks=[],
            total_chunks_searched=0,
        )

    disclaimer: str | None = _DISCLAIMERS.get(policy_flag) if policy_flag else None

    if not processed.search_required:
        diagnostics = QueryDiagnostics(
            intent=processed.intent,
            search_required=False,
            rewrite_applied=rewrite_applied,
            reason=processed.reason,
            rewrite_notes=processed.rewrite_notes,
            topic_query=topic_query,
            retrieval_queries=retrieval_queries,
            policy_flag=policy_flag,
            answer_intent=answer_intent,
        )
        return QueryResponse(
            original_query=original_query,
            processed_query=processed_query,
            top_k=request.top_k or settings.query_top_k,
            status="search_not_required",
            diagnostics=diagnostics,
            retrieved_chunks=[],
            total_chunks_searched=0,
            disclaimer=disclaimer,
        )

    diagnostics = QueryDiagnostics(
        intent=processed.intent,
        search_required=True,
        rewrite_applied=rewrite_applied,
        reason=processed.reason,
        rewrite_notes=processed.rewrite_notes,
        topic_query=topic_query,
        retrieval_queries=retrieval_queries,
        policy_flag=policy_flag,
        answer_intent=answer_intent,
    )
    top_k = request.top_k or settings.query_top_k
    candidate_k = max(top_k, settings.semantic_candidate_k)
    coverage_query = topic_query or processed_query
    query_terms = set(KeywordSearcher.tokenize(coverage_query))
    keyword_hit_map = _collect_keyword_hits(retrieval_queries, chunks=chunks, top_k=candidate_k)
    keyword_hits = list(keyword_hit_map.values())

    semantic_hits = []
    semantic_error: str | None = None
    if settings.semantic_search_enabled:
        cached_embeddings = embedding_store.load_embeddings()
        new_embeddings: dict[str, list[float]] = {}
        semantic_hit_map: dict[str, tuple[ChunkRecord, float]] = {}
        keyword_priority_ids = [
            chunk.chunk_id
            for chunk, _score in sorted(keyword_hits, key=lambda item: item[1], reverse=True)
        ]
        semantic_queries = [processed_query]
        if topic_query and topic_query != processed_query and len(topic_query.split()) >= 2:
            semantic_queries.append(topic_query)
        try:
            for semantic_query in semantic_queries:
                query_hits, generated_embeddings = await semantic_searcher.search_async(
                    semantic_query,
                    chunks,
                    top_k=candidate_k,
                    cached_embeddings=cached_embeddings,
                    max_new_chunk_embeddings=settings.semantic_max_new_chunk_embeddings_per_query,
                    missing_priority_chunk_ids=keyword_priority_ids,
                )
                if generated_embeddings:
                    cached_embeddings.update(generated_embeddings)
                    new_embeddings.update(generated_embeddings)

                for hit in query_hits:
                    existing = semantic_hit_map.get(hit.chunk.chunk_id)
                    if existing is None or hit.score > existing[1]:
                        semantic_hit_map[hit.chunk.chunk_id] = (hit.chunk, hit.score)

            semantic_hits = list(semantic_hit_map.values())
            if new_embeddings:
                embedding_store.append_embeddings(new_embeddings)
        except Exception as exc:
            semantic_error = str(exc)

    keyword_scores = {chunk.chunk_id: score for chunk, score in keyword_hits}
    semantic_scores = {chunk.chunk_id: score for chunk, score in semantic_hits}
    norm_keyword = _normalize_scores(keyword_scores)
    norm_semantic = _normalize_scores(semantic_scores)

    by_chunk_id: dict[str, ChunkRecord] = {}
    for chunk, _score in keyword_hits:
        by_chunk_id[chunk.chunk_id] = chunk
    for chunk, _score in semantic_hits:
        by_chunk_id[chunk.chunk_id] = chunk

    fused_scores: dict[str, float] = {}
    for chunk_id in by_chunk_id:
        has_keyword = chunk_id in norm_keyword
        has_semantic = chunk_id in norm_semantic
        if has_keyword and has_semantic:
            fused_scores[chunk_id] = (
                settings.keyword_weight * norm_keyword[chunk_id]
                + settings.semantic_weight * norm_semantic[chunk_id]
            )
        elif has_keyword:
            fused_scores[chunk_id] = norm_keyword[chunk_id]
        elif has_semantic:
            fused_scores[chunk_id] = norm_semantic[chunk_id]

    dominant_source = _apply_source_consistency_bonus(
        fused_scores=fused_scores,
        by_chunk_id=by_chunk_id,
        query_term_count=len(query_terms),
    )
    if dominant_source:
        diagnostics.rewrite_notes.append(
            "Applied short-query source consistency tie-breaker."
        )

    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    thresholded_chunk_ids = []
    for chunk_id in ranked_chunk_ids:
        score_ok = fused_scores[chunk_id] >= settings.relevance_threshold
        coverage = _query_term_coverage(query_terms, by_chunk_id[chunk_id].text)
        coverage_ok = coverage >= settings.min_query_term_coverage
        if score_ok and coverage_ok:
            thresholded_chunk_ids.append(chunk_id)
        if len(thresholded_chunk_ids) >= top_k:
            break

    if not thresholded_chunk_ids:
        diagnostics.reason = (
            "No retrieved chunks met the evidence threshold. "
            "Please provide a more specific question or ingest more relevant documents."
        )
        return QueryResponse(
            original_query=original_query,
            processed_query=processed_query,
            top_k=top_k,
            status="insufficient_evidence",
            diagnostics=diagnostics,
            retrieved_chunks=[],
            total_chunks_searched=len(chunks),
            disclaimer=disclaimer,
        )

    retrieved_chunks: list[RetrievedChunk] = []
    for chunk_id in thresholded_chunk_ids:
        chunk = by_chunk_id[chunk_id]
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                relevance_score=fused_scores[chunk_id],
                keyword_score=keyword_scores.get(chunk_id),
                semantic_score=semantic_scores.get(chunk_id),
            )
        )

    if semantic_error:
        diagnostics.reason = (
            "Query passed intent gate. Keyword retrieval completed; semantic retrieval unavailable: "
            f"{semantic_error}"
        )
    else:
        diagnostics.reason = "Query passed intent gate. Keyword and semantic retrieval completed."

    generated_answer: str | None = None
    cited_chunk_ids: list[str] = []
    if settings.generation_enabled:
        top_relevance_score = max(chunk.relevance_score for chunk in retrieved_chunks)
        avg_relevance_score = sum(chunk.relevance_score for chunk in retrieved_chunks) / len(retrieved_chunks)
        low_confidence = (
            top_relevance_score < settings.generation_min_top_relevance_score
            or avg_relevance_score < settings.generation_min_avg_relevance_score
        )
        if low_confidence:
            diagnostics.rewrite_notes.append(
                "Skipped answer generation due to low retrieval confidence."
            )
        else:
            generation_result = await answer_generator.generate_answer_async(
                query=original_query,
                retrieved_chunks=retrieved_chunks,
                answer_intent=answer_intent,
            )
            if generation_result.error:
                diagnostics.rewrite_notes.append(
                    f"Answer generation unavailable: {generation_result.error}"
                )
            else:
                generated_answer = generation_result.answer
                cited_chunk_ids = generation_result.cited_chunk_ids

    return QueryResponse(
        original_query=original_query,
        processed_query=processed_query,
        top_k=top_k,
        status="retrieval_complete",
        diagnostics=diagnostics,
        retrieved_chunks=retrieved_chunks,
        total_chunks_searched=len(chunks),
        generated_answer=generated_answer,
        cited_chunk_ids=cited_chunk_ids,
        disclaimer=disclaimer,
    )


@app.post("/ingestion/pdfs", response_model=IngestionResponse)
async def ingest_pdfs(files: list[UploadFile] = File(...)) -> IngestionResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    max_size = settings.max_upload_size_mb * 1024 * 1024
    ingested: list[FileIngestionResult] = []
    skipped: list[FileIngestionSkipped] = []
    failed: list[FileIngestionError] = []
    total_chunks_written = 0

    for file in files:
        filename = file.filename or "unnamed.pdf"
        if not filename.lower().endswith(".pdf"):
            failed.append(FileIngestionError(filename=filename, reason="Only PDF files are allowed."))
            continue

        file_bytes = await file.read()
        if len(file_bytes) > max_size:
            failed.append(
                FileIngestionError(
                    filename=filename,
                    reason=f"File exceeds size limit of {settings.max_upload_size_mb} MB.",
                )
            )
            continue

        file_hash = store.content_hash(file_bytes)
        existing = store.find_document_by_sha256(file_hash)
        if existing is not None:
            skipped.append(
                FileIngestionSkipped(
                    filename=filename,
                    reason=f"Duplicate file already ingested as document_id={existing.document_id}.",
                )
            )
            continue

        try:
            pages = extract_pdf_pages(
                file_bytes,
                ocr_fallback_enabled=settings.ocr_fallback_enabled,
                mistral_api_key=settings.mistral_api_key,
                mistral_ocr_model=settings.mistral_ocr_model,
            )
            slices = build_chunks(
                pages,
                chunk_size_chars=settings.chunk_size_chars,
                chunk_overlap_chars=settings.chunk_overlap_chars,
                min_chunk_chars=settings.min_chunk_chars,
            )

            document_id, stored_path = store.save_upload(filename, file_bytes)
            now = store.utc_now()

            chunk_records: list[ChunkRecord] = []
            for slice_item in slices:
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=str(uuid4()),
                        document_id=document_id,
                        source_file=stored_path.name,
                        page_start=slice_item.page_start,
                        page_end=slice_item.page_end,
                        text=slice_item.text,
                        char_count=len(slice_item.text),
                        metadata={"ingestion_strategy": "page_aware_overlap"},
                        created_at=now,
                    )
                )

            document_record = DocumentRecord(
                document_id=document_id,
                source_file=stored_path.name,
                original_filename=filename,
                page_count=len(pages),
                chunk_count=len(chunk_records),
                sha256=file_hash,
                created_at=now,
            )

            store.append_document(document_record)
            store.append_chunks(chunk_records)
            keyword_searcher.invalidate_cache()

            ingested.append(
                FileIngestionResult(
                    filename=filename,
                    document_id=document_id,
                    page_count=len(pages),
                    chunk_count=len(chunk_records),
                    sha256=document_record.sha256,
                )
            )
            total_chunks_written += len(chunk_records)
        except ValueError as exc:
            failed.append(FileIngestionError(filename=filename, reason=str(exc)))
        except Exception as exc:  # defensive catch for malformed PDFs
            failed.append(FileIngestionError(filename=filename, reason=f"Unexpected ingestion error: {exc}"))

    return IngestionResponse(
        ingested=ingested,
        skipped=skipped,
        failed=failed,
        total_chunks_written=total_chunks_written,
    )


@app.delete("/ingestion/reset", response_model=ClearIngestionResponse)
def clear_ingestion() -> ClearIngestionResponse:
    deleted_upload_entries, cleared_documents, cleared_chunks = store.clear_ingested_data()
    embedding_store.embeddings_file.write_text("", encoding="utf-8")
    keyword_searcher.invalidate_cache()
    return ClearIngestionResponse(
        deleted_upload_entries=deleted_upload_entries,
        cleared_documents=cleared_documents,
        cleared_chunks=cleared_chunks,
    )

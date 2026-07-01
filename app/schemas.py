from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    char_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DocumentRecord(BaseModel):
    document_id: str
    source_file: str
    original_filename: str
    page_count: int
    chunk_count: int
    sha256: str
    created_at: datetime


class FileIngestionResult(BaseModel):
    filename: str
    document_id: str
    page_count: int
    chunk_count: int
    sha256: str


class FileIngestionError(BaseModel):
    filename: str
    reason: str

class FileIngestionSkipped(BaseModel):
    filename: str
    reason: str


class IngestionResponse(BaseModel):
    ingested: list[FileIngestionResult]
    skipped: list[FileIngestionSkipped]
    failed: list[FileIngestionError]
    total_chunks_written: int


class ClearIngestionResponse(BaseModel):
    deleted_upload_entries: int
    cleared_documents: int
    cleared_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    relevance_score: float
    keyword_score: float | None = None
    semantic_score: float | None = None


class QueryDiagnostics(BaseModel):
    intent: str
    search_required: bool
    rewrite_applied: bool
    reason: str | None = None
    rewrite_notes: list[str] = Field(default_factory=list)
    topic_query: str | None = None
    retrieval_queries: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    original_query: str
    processed_query: str
    top_k: int
    status: Literal[
        "search_not_required",
        "ready_for_retrieval",
        "retrieval_complete",
        "insufficient_evidence",
    ]
    diagnostics: QueryDiagnostics
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_chunks_searched: int = 0

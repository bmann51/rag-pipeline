from datetime import datetime
from typing import Any

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

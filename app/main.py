from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.ingestion.pdf_ingestor import build_chunks, extract_pdf_pages
from app.schemas import (
    ClearIngestionResponse,
    ChunkRecord,
    DocumentRecord,
    FileIngestionError,
    FileIngestionResult,
    FileIngestionSkipped,
    IngestionResponse,
)
from app.storage.document_store import DocumentStore

settings = get_settings()
store = DocumentStore(upload_dir=settings.upload_dir, record_dir=settings.record_dir)

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.on_event("startup")
def startup() -> None:
    store.ensure_dirs()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    return ClearIngestionResponse(
        deleted_upload_entries=deleted_upload_entries,
        cleared_documents=cleared_documents,
        cleared_chunks=cleared_chunks,
    )

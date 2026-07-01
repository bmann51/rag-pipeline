import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.schemas import ChunkRecord, DocumentRecord


class DocumentStore:
    def __init__(self, upload_dir: Path, record_dir: Path) -> None:
        self.upload_dir = upload_dir
        self.record_dir = record_dir
        self.documents_file = self.record_dir / "documents.jsonl"
        self.chunks_file = self.record_dir / "chunks.jsonl"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.documents_file.touch(exist_ok=True)
        self.chunks_file.touch(exist_ok=True)

    @staticmethod
    def content_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def find_document_by_sha256(self, sha256: str) -> DocumentRecord | None:
        if not self.documents_file.exists():
            return None

        with self.documents_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = line.strip()
                if not payload:
                    continue
                data = json.loads(payload)
                if data.get("sha256") == sha256:
                    return DocumentRecord.model_validate(data)

        return None

    def save_upload(self, original_filename: str, file_bytes: bytes) -> tuple[str, Path]:
        document_id = str(uuid4())
        safe_name = Path(original_filename).name
        stored_name = f"{document_id}_{safe_name}"
        target_path = self.upload_dir / stored_name
        target_path.write_bytes(file_bytes)
        return document_id, target_path

    def append_document(self, record: DocumentRecord) -> None:
        payload = record.model_dump(mode="json")
        with self.documents_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def append_chunks(self, chunks: list[ChunkRecord]) -> None:
        with self.chunks_file.open("a", encoding="utf-8") as fh:
            for chunk in chunks:
                payload = chunk.model_dump(mode="json")
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    @staticmethod
    def _count_jsonl_records(path: Path) -> int:
        if not path.exists():
            return 0

        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def clear_ingested_data(self) -> tuple[int, int, int]:
        self.ensure_dirs()

        deleted_upload_entries = 0
        for item in self.upload_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted_upload_entries += 1
            elif item.is_file():
                item.unlink()
                deleted_upload_entries += 1

        cleared_documents = self._count_jsonl_records(self.documents_file)
        cleared_chunks = self._count_jsonl_records(self.chunks_file)

        self.documents_file.write_text("", encoding="utf-8")
        self.chunks_file.write_text("", encoding="utf-8")

        return deleted_upload_entries, cleared_documents, cleared_chunks

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(timezone.utc)

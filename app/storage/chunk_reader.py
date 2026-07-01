import json
from pathlib import Path

from app.schemas import ChunkRecord


class ChunkReader:
    def __init__(self, record_dir: Path) -> None:
        self.record_dir = record_dir
        self.chunks_file = self.record_dir / "chunks.jsonl"

    def load_chunks(self) -> list[ChunkRecord]:
        if not self.chunks_file.exists():
            return []

        chunks: list[ChunkRecord] = []
        with self.chunks_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                    chunks.append(ChunkRecord.model_validate(data))
                except Exception:
                    continue

        return chunks

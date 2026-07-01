import json
from pathlib import Path


class EmbeddingStore:
    def __init__(self, record_dir: Path) -> None:
        self.record_dir = record_dir
        self.embeddings_file = self.record_dir / "chunk_embeddings.jsonl"

    def ensure_file(self) -> None:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_file.touch(exist_ok=True)

    def load_embeddings(self) -> dict[str, list[float]]:
        if not self.embeddings_file.exists():
            return {}

        embeddings: dict[str, list[float]] = {}
        with self.embeddings_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                    chunk_id = data.get("chunk_id")
                    vector = data.get("embedding")
                    if not chunk_id or not isinstance(vector, list):
                        continue
                    embeddings[chunk_id] = [float(item) for item in vector]
                except Exception:
                    continue

        return embeddings

    def append_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        if not embeddings:
            return

        self.ensure_file()
        with self.embeddings_file.open("a", encoding="utf-8") as fh:
            for chunk_id, vector in embeddings.items():
                payload = {
                    "chunk_id": chunk_id,
                    "embedding": vector,
                }
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
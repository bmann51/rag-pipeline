from __future__ import annotations

import math
from dataclasses import dataclass

from app.retrieval.embeddings import EmbeddingClient
from app.schemas import ChunkRecord


@dataclass
class SemanticHit:
    chunk: ChunkRecord
    score: float


class SemanticSearcher:
    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(
        self,
        query: str,
        chunks: list[ChunkRecord],
        *,
        top_k: int,
        cached_embeddings: dict[str, list[float]] | None = None,
        max_new_chunk_embeddings: int | None = None,
        missing_priority_chunk_ids: list[str] | None = None,
    ) -> tuple[list[SemanticHit], dict[str, list[float]]]:
        if not query.strip() or not chunks:
            return [], {}

        query_embedding = self.embedding_client.embed_text(query)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        embeddings = dict(cached_embeddings or {})

        missing_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in embeddings]
        if missing_ids and missing_priority_chunk_ids:
            prioritized = [chunk_id for chunk_id in missing_priority_chunk_ids if chunk_id in set(missing_ids)]
            remaining = [chunk_id for chunk_id in missing_ids if chunk_id not in set(prioritized)]
            missing_ids = prioritized + remaining

        if max_new_chunk_embeddings is not None:
            missing_ids = missing_ids[:max_new_chunk_embeddings]

        new_embeddings: dict[str, list[float]] = {}
        if missing_ids:
            missing_chunks = [chunk_by_id[chunk_id] for chunk_id in missing_ids]
            vectors = self.embedding_client.embed_texts([chunk.text for chunk in missing_chunks])
            for chunk, vector in zip(missing_chunks, vectors):
                new_embeddings[chunk.chunk_id] = vector
                embeddings[chunk.chunk_id] = vector

        hits: list[SemanticHit] = []
        for chunk in chunks:
            vector = embeddings.get(chunk.chunk_id)
            if vector is None:
                continue
            score = self._cosine_similarity(query_embedding, vector)
            hits.append(SemanticHit(chunk=chunk, score=score))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k], new_embeddings

    async def search_async(
        self,
        query: str,
        chunks: list[ChunkRecord],
        *,
        top_k: int,
        cached_embeddings: dict[str, list[float]] | None = None,
        max_new_chunk_embeddings: int | None = None,
        missing_priority_chunk_ids: list[str] | None = None,
    ) -> tuple[list[SemanticHit], dict[str, list[float]]]:
        if not query.strip() or not chunks:
            return [], {}

        query_embedding = await self.embedding_client.embed_text_async(query)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        embeddings = dict(cached_embeddings or {})

        missing_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in embeddings]
        if missing_ids and missing_priority_chunk_ids:
            prioritized = [chunk_id for chunk_id in missing_priority_chunk_ids if chunk_id in set(missing_ids)]
            remaining = [chunk_id for chunk_id in missing_ids if chunk_id not in set(prioritized)]
            missing_ids = prioritized + remaining

        if max_new_chunk_embeddings is not None:
            missing_ids = missing_ids[:max_new_chunk_embeddings]

        new_embeddings: dict[str, list[float]] = {}
        if missing_ids:
            missing_chunks = [chunk_by_id[chunk_id] for chunk_id in missing_ids]
            vectors = await self.embedding_client.embed_texts_async([chunk.text for chunk in missing_chunks])
            for chunk, vector in zip(missing_chunks, vectors):
                new_embeddings[chunk.chunk_id] = vector
                embeddings[chunk.chunk_id] = vector

        hits: list[SemanticHit] = []
        for chunk in chunks:
            vector = embeddings.get(chunk.chunk_id)
            if vector is None:
                continue
            score = self._cosine_similarity(query_embedding, vector)
            hits.append(SemanticHit(chunk=chunk, score=score))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k], new_embeddings

from __future__ import annotations

import random
import time

from mistralai import Mistral


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        batch_size: int = 16,
        min_request_interval_seconds: float = 0.0,
        max_retries_on_rate_limit: int = 3,
        retry_base_delay_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self.max_retries_on_rate_limit = max(0, max_retries_on_rate_limit)
        self.retry_base_delay_seconds = max(0.1, retry_base_delay_seconds)
        self._client: Mistral | None = None
        self._last_request_time: float | None = None

    def _get_client(self) -> Mistral:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required for semantic search.")
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._request_embeddings(client, batch)
            vectors.extend(item.embedding for item in response.data)

        if len(vectors) != len(texts):
            raise ValueError("Embedding response size mismatch.")

        return vectors

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0]

    def _request_embeddings(self, client: Mistral, batch: list[str]):
        attempt = 0
        while True:
            self._wait_for_rate_window()
            try:
                response = client.embeddings.create(model=self.model, inputs=batch)
                self._last_request_time = time.monotonic()
                return response
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt >= self.max_retries_on_rate_limit:
                    raise
                delay = (self.retry_base_delay_seconds * (2**attempt)) + random.uniform(0.0, 0.25)
                time.sleep(delay)
                attempt += 1

    def _wait_for_rate_window(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return

        now = time.monotonic()
        if self._last_request_time is None:
            return

        elapsed = now - self._last_request_time
        remaining = self.min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True

        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "ratelimit" in message

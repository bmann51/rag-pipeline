from __future__ import annotations

import re
from dataclasses import dataclass

from mistralai import Mistral

from app.schemas import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9\-]+)\]")
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
WORD_PATTERN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)


@dataclass
class AnswerGenerationResult:
    answer: str | None
    cited_chunk_ids: list[str]
    error: str | None = None


class AnswerGenerator:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        max_chunks: int,
        max_chars_per_chunk: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self._client: Mistral | None = None

    def _get_client(self) -> Mistral:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required for answer generation.")
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def generate_answer(self, *, query: str, retrieved_chunks: list[RetrievedChunk]) -> AnswerGenerationResult:
        if not retrieved_chunks:
            return AnswerGenerationResult(answer=None, cited_chunk_ids=[], error="No chunks available.")

        limited_chunks = retrieved_chunks[: self.max_chunks]
        valid_chunk_ids = {chunk.chunk_id for chunk in limited_chunks}
        context = self._build_context(limited_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-grounded assistant. Answer only using the provided context. "
                    "If evidence is insufficient, say: 'I don't have enough evidence in the provided documents to answer that.' "
                    "Cite factual statements inline using chunk IDs in square brackets, e.g. [chunk-id]."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer with citations.",
            },
        ]

        try:
            content = self._chat_complete(messages)
        except Exception as exc:
            return AnswerGenerationResult(answer=None, cited_chunk_ids=[], error=str(exc))

        if not content:
            return AnswerGenerationResult(answer=None, cited_chunk_ids=[], error="Empty generation output.")

        cited_chunk_ids = self._extract_citations(content, valid_chunk_ids)
        if not cited_chunk_ids:
            cited_chunk_ids = self._infer_citations_from_answer(content, limited_chunks)

        if not cited_chunk_ids:
            return AnswerGenerationResult(
                answer=None,
                cited_chunk_ids=[],
                error="Generated answer did not contain valid citations.",
            )

        return AnswerGenerationResult(answer=content.strip(), cited_chunk_ids=cited_chunk_ids, error=None)

    def _extract_citations(self, content: str, valid_chunk_ids: set[str]) -> list[str]:
        seen: set[str] = set()
        cited_chunk_ids: list[str] = []

        for match in CITATION_PATTERN.findall(content):
            if match in valid_chunk_ids and match not in seen:
                seen.add(match)
                cited_chunk_ids.append(match)

        # Accept UUID citations even when not wrapped in square brackets.
        for match in UUID_PATTERN.findall(content):
            if match in valid_chunk_ids and match not in seen:
                seen.add(match)
                cited_chunk_ids.append(match)

        return cited_chunk_ids

    def _infer_citations_from_answer(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[str]:
        answer_terms = set(term.lower() for term in WORD_PATTERN.findall(answer))
        if len(answer_terms) < 6:
            return []

        best_chunk_id: str | None = None
        best_overlap = 0.0
        for chunk in chunks:
            chunk_terms = set(term.lower() for term in WORD_PATTERN.findall(chunk.text))
            if not chunk_terms:
                continue
            overlap = len(answer_terms.intersection(chunk_terms)) / len(answer_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk_id = chunk.chunk_id

        # Conservative fallback: attach one citation only when overlap is strong enough.
        if best_chunk_id and best_overlap >= 0.25:
            return [best_chunk_id]
        return []

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            excerpt = chunk.text.strip()
            if len(excerpt) > self.max_chars_per_chunk:
                excerpt = excerpt[: self.max_chars_per_chunk].rstrip() + " ..."
            blocks.append(
                f"[{chunk.chunk_id}] source={chunk.source_file} pages={chunk.page_start}-{chunk.page_end}\n{excerpt}"
            )
        return "\n\n".join(blocks)

    def _chat_complete(self, messages: list[dict[str, str]]) -> str:
        client = self._get_client()
        chat_api = getattr(client, "chat", None)
        if chat_api is None:
            raise ValueError("Mistral chat API is unavailable in this client version.")

        response = None
        if hasattr(chat_api, "complete"):
            response = chat_api.complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif hasattr(chat_api, "completions") and hasattr(chat_api.completions, "create"):
            response = chat_api.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError("Mistral chat completion method not found.")

        choices = getattr(response, "choices", None)
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
            return "\n".join(parts).strip()
        return str(content)
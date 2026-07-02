from __future__ import annotations

import re
from dataclasses import dataclass, field

from mistralai import Mistral

from app.schemas import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9\-]+)\]")
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
WORD_PATTERN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_HEDGE_PHRASES = (
    "i don't have enough evidence",
    "i do not have enough evidence",
    "insufficient evidence",
    "cannot answer",
    "i do not have",
)

_TOPIC_CLASSIFY_PROMPT = (
    "Classify the user query with exactly one word: \"legal\", \"medical\", or \"none\".\n"
    "legal — seeks legal advice, mentions lawsuits, liability, attorneys, contracts, regulations.\n"
    "medical — seeks medical advice, diagnosis, treatment, medications, symptoms, dosage.\n"
    "none — anything else."
)

_CITE_RULE = (
    "Cite every factual claim inline using the chunk ID in square brackets, e.g. [chunk-id]. "
    "If the provided context is insufficient, say exactly: "
    "'I don't have enough evidence in the provided documents to answer that.'"
)

SYSTEM_PROMPTS: dict[str, str] = {
    "factual": (
        f"You are a retrieval-grounded assistant. Answer only using the provided context. {_CITE_RULE}"
    ),
    "list": (
        "You are a retrieval-grounded assistant. Answer only using the provided context. "
        "Structure your entire answer as a numbered list. "
        f"Each list item must cite its source chunk ID in square brackets. {_CITE_RULE}"
    ),
    "compare": (
        "You are a retrieval-grounded assistant. Answer only using the provided context. "
        "Compare the subjects directly using a structured format — clearly labelled paragraphs "
        "or a side-by-side breakdown per subject. "
        f"{_CITE_RULE}"
    ),
    "summarize": (
        "You are a retrieval-grounded assistant. Answer only using the provided context. "
        "Provide a structured summary using short paragraphs or a brief outline with headers. "
        f"{_CITE_RULE}"
    ),
}


@dataclass
class AnswerGenerationResult:
    answer: str | None
    cited_chunk_ids: list[str]
    error: str | None = None
    unsupported_sentences: list[str] = field(default_factory=list)
    hallucination_warning: bool = False


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

    async def classify_sensitive_topic_async(self, query: str) -> str | None:
        """Returns 'legal_topic', 'medical_topic', or None. Never raises."""
        if not self.api_key:
            return None
        try:
            client = self._get_client()
            response = await client.chat.complete_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": _TOPIC_CLASSIFY_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            content = self._extract_content(response).strip().lower()
            word = content.split()[0] if content else ""
            if word == "legal":
                return "legal_topic"
            if word == "medical":
                return "medical_topic"
        except Exception:
            pass
        return None

    async def generate_answer_async(
        self,
        *,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        answer_intent: str = "factual",
    ) -> AnswerGenerationResult:
        if not retrieved_chunks:
            return AnswerGenerationResult(answer=None, cited_chunk_ids=[], error="No chunks available.")

        limited_chunks = retrieved_chunks[: self.max_chunks]
        valid_chunk_ids = {chunk.chunk_id for chunk in limited_chunks}
        context = self._build_context(limited_chunks)
        system_prompt = SYSTEM_PROMPTS.get(answer_intent, SYSTEM_PROMPTS["factual"])
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer with citations.",
            },
        ]

        try:
            content = await self._chat_complete_async(messages)
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

        unsupported, warning = self._check_sentence_support(content.strip(), limited_chunks)
        return AnswerGenerationResult(
            answer=content.strip(),
            cited_chunk_ids=cited_chunk_ids,
            error=None,
            unsupported_sentences=unsupported,
            hallucination_warning=warning,
        )

    def generate_answer(
        self,
        *,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        answer_intent: str = "factual",
    ) -> AnswerGenerationResult:
        if not retrieved_chunks:
            return AnswerGenerationResult(answer=None, cited_chunk_ids=[], error="No chunks available.")

        limited_chunks = retrieved_chunks[: self.max_chunks]
        valid_chunk_ids = {chunk.chunk_id for chunk in limited_chunks}
        context = self._build_context(limited_chunks)
        system_prompt = SYSTEM_PROMPTS.get(answer_intent, SYSTEM_PROMPTS["factual"])
        messages = [
            {"role": "system", "content": system_prompt},
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

        unsupported, warning = self._check_sentence_support(content.strip(), limited_chunks)
        return AnswerGenerationResult(
            answer=content.strip(),
            cited_chunk_ids=cited_chunk_ids,
            error=None,
            unsupported_sentences=unsupported,
            hallucination_warning=warning,
        )

    @staticmethod
    def _check_sentence_support(
        answer: str,
        chunks: list[RetrievedChunk],
        *,
        min_meaningful_words: int = 5,
        overlap_threshold: float = 0.15,
    ) -> tuple[list[str], bool]:
        chunk_term_sets = [
            set(t.lower() for t in WORD_PATTERN.findall(chunk.text))
            for chunk in chunks
        ]
        unsupported: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(answer.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            if any(hedge in lowered for hedge in _HEDGE_PHRASES):
                continue
            stripped_citations = CITATION_PATTERN.sub("", sentence).strip()
            if not stripped_citations:
                continue
            terms = set(t.lower() for t in WORD_PATTERN.findall(stripped_citations))
            if len(terms) < min_meaningful_words:
                continue
            best_overlap = max(
                (len(terms & chunk_terms) / len(terms) for chunk_terms in chunk_term_sets if chunk_terms),
                default=0.0,
            )
            if best_overlap < overlap_threshold:
                unsupported.append(sentence)

        return unsupported, bool(unsupported)

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

        return self._extract_content(response)

    async def _chat_complete_async(self, messages: list[dict[str, str]]) -> str:
        client = self._get_client()
        chat_api = getattr(client, "chat", None)
        if chat_api is None:
            raise ValueError("Mistral chat API is unavailable in this client version.")

        if not hasattr(chat_api, "complete_async"):
            raise ValueError("Mistral async chat completion not available in this SDK version.")

        response = await chat_api.complete_async(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._extract_content(response)

    def _extract_content(self, response) -> str:
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
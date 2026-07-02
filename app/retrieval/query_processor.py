from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import ChunkRecord

CHITCHAT_QUERIES = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what's up",
    "whats up",
    "yo",
}

CONVERSATIONAL_PREFIXES = (
    "hello",
    "hi",
    "hey",
    "hiya",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what's up",
    "whats up",
    "yo",
    "thanks",
    "thank you",
    "nice",
    "cool",
    "sounds",
)

CONVERSATIONAL_FILLER_TOKENS = {
    "awesome",
    "cool",
    "great",
    "interesting",
    "me",
    "nice",
    "ok",
    "okay",
    "please",
    "sounds",
    "thanks",
    "thank",
    "you",
}

SEARCH_INTENT_PREFIXES = (
    "what",
    "how",
    "why",
    "where",
    "when",
    "which",
    "who",
    "whose",
    "can",
    "could",
    "would",
    "do",
    "does",
    "did",
    "is",
    "are",
    "tell me about",
    "explain",
    "describe",
    "summarize",
    "find",
    "show me",
    "search for",
    "look up",
)

RETRIEVAL_VERB_PREFIXES = (
    "tell me",
    "show me",
    "find",
    "search",
    "look up",
    "lookup",
    "summarize",
    "explain",
    "describe",
    "compare",
    "mention",
    "list",
)

DOCUMENT_CUE_TERMS = {
    "document",
    "documents",
    "doc",
    "docs",
    "file",
    "files",
    "pdf",
    "pdfs",
    "page",
    "pages",
    "chunk",
    "chunks",
    "corpus",
    "kb",
    "knowledgebase",
    "knowledge",
}

DECLARATIVE_TOKENS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "is",
    "was",
    "were",
    "has",
    "have",
    "had",
}

CONVERSATIONAL_LEAD_TOKENS = {
    "i",
    "i'm",
    "im",
    "we",
    "we're",
    "were",
    "you",
    "he",
    "she",
    "they",
    "it",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
}

TOPIC_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}

_PII_SSN = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
_PII_CARD = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")

_LEGAL_TOPIC = re.compile(
    r"\b(legal\s+advice|am\s+i\s+liable|can\s+i\s+sue|lawsuit|litigation|"
    r"attorney|lawyer|is\s+it\s+illegal|is\s+it\s+legal|defamation|"
    r"malpractice|breach\s+of\s+contract|legal\s+action|arbitration)\b",
    re.IGNORECASE,
)
_MEDICAL_TOPIC = re.compile(
    r"\b(medical\s+advice|diagnos[ei]s?|prescription|dosage|overdose|"
    r"should\s+i\s+take|am\s+i\s+sick|do\s+i\s+have|symptoms?\s+of|"
    r"cure\s+for|treatment\s+for|drug\s+interaction|side\s+effects?\s+of|"
    r"what\s+medication|is\s+this\s+cancer|is\s+this\s+serious)\b",
    re.IGNORECASE,
)

_ANSWER_LIST = re.compile(
    r"^(?:list|enumerate|name\s+all|what\s+are\s+all\s+(?:the\s+)?|"
    r"give\s+me\s+(?:a\s+)?list|show\s+all|list\s+all|what\s+are\s+the\s+(?:main\s+)?)\b",
    re.IGNORECASE,
)
_ANSWER_COMPARE = re.compile(
    r"^(?:compare|difference\s+between|contrast|how\s+does\s+.+\s+differ|"
    r"\bvs\b|\bversus\b)",
    re.IGNORECASE,
)
_ANSWER_SUMMARIZE = re.compile(
    r"^(?:summarize|summary\s+of|give\s+(?:me\s+)?(?:a\s+)?summary|"
    r"overview\s+of|provide\s+(?:a\s+)?summary|what\s+is\s+the\s+summary)\b",
    re.IGNORECASE,
)

ACRONYM_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")
LONG_PHRASE_TO_ACRONYM = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\-\s]{2,80}?)\s*\(([A-Z][A-Z0-9]{1,9})\)",
)
ACRONYM_TO_LONG_PHRASE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,9})\s*\(([A-Za-z][A-Za-z0-9\-\s]{2,80}?)\)",
)
NON_WORD_SPACE = re.compile(r"[^\w\s\-]")
MULTI_SPACE = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
ABOUT_DOCUMENT_PATTERN = re.compile(
    r"^(?:what\s+does\s+)?(?:the\s+)?(?:document|documents|doc|docs|file|files|pdf|pdfs)\s+"
    r"(?:say|says|mention|mentions|discuss|discusses|cover|covers)\s+about\s+(.+)$",
    re.IGNORECASE,
)
IN_DOCUMENT_PATTERN = re.compile(
    r"^(?:is|are|was|were|do|does|did)\s+(.+?)\s+"
    r"(?:mentioned|discussed|covered|described|explained)\s+in\s+(?:the\s+)?"
    r"(?:document|documents|doc|docs|file|files|pdf|pdfs)$",
    re.IGNORECASE,
)
MENTIONS_PATTERN = re.compile(r"^(?:show me\s+)?mentions\s+of\s+(.+)$", re.IGNORECASE)
VERB_TOPIC_PATTERN = re.compile(
    r"^(?:tell me about|show me|find|look up|search for|summarize|explain|describe)\s+(.+)$",
    re.IGNORECASE,
)
COMPARE_PATTERN = re.compile(r"^(?:compare|difference between)\s+(.+)$", re.IGNORECASE)


@dataclass
class QueryProcessingResult:
    original_query: str
    normalized_query: str
    processed_query: str
    topic_query: str | None
    retrieval_queries: list[str]
    rewrite_applied: bool
    intent: str
    search_required: bool
    reason: str | None = None
    rewrite_notes: list[str] = field(default_factory=list)
    policy_flag: str | None = None
    answer_intent: str = "factual"


class QueryProcessor:
    def __init__(self) -> None:
        self.filler_prefixes = (
            "what can you tell me about",
            "can you tell me about",
            "could you tell me about",
            "would you tell me about",
            "help me understand",
            "can you",
            "could you",
            "would you",
            "please",
            "i want to know",
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return MULTI_SPACE.sub(" ", text).strip()

    @staticmethod
    def _clean_query_text(text: str) -> str:
        cleaned = text.strip().rstrip("?.! ")
        cleaned = NON_WORD_SPACE.sub(" ", cleaned)
        return QueryProcessor._normalize_whitespace(cleaned)

    def _strip_filler_prefix(self, text: str) -> tuple[str, bool]:
        lowered = text.lower()
        for prefix in self.filler_prefixes:
            if lowered.startswith(prefix + " "):
                stripped = text[len(prefix) :].strip()
                return stripped, True
        return text, False

    @staticmethod
    def _has_document_cue(text: str) -> bool:
        tokens = set(WORD_PATTERN.findall(text.lower()))
        return any(token in DOCUMENT_CUE_TERMS for token in tokens)

    @staticmethod
    def _has_search_intent_prefix(text: str) -> bool:
        lowered = text.lower().strip()
        if lowered.endswith("?"):
            return True
        return any(lowered.startswith(prefix) for prefix in SEARCH_INTENT_PREFIXES)

    @staticmethod
    def _has_retrieval_verb(text: str) -> bool:
        lowered = text.lower().strip()
        return any(lowered.startswith(prefix) for prefix in RETRIEVAL_VERB_PREFIXES)

    @staticmethod
    def _has_conversational_prefix(text: str) -> bool:
        lowered = text.lower().strip()
        return any(lowered.startswith(prefix) for prefix in CONVERSATIONAL_PREFIXES)

    @staticmethod
    def _tokenize_for_intent(text: str) -> list[str]:
        return [token.lower() for token in WORD_PATTERN.findall(text)]

    def _is_compact_topic_query(self, text: str) -> bool:
        tokens = self._tokenize_for_intent(text)
        if not tokens or len(tokens) > 6:
            return False

        if self._has_conversational_prefix(text):
            return False

        if tokens[0] in CONVERSATIONAL_LEAD_TOKENS:
            return False

        if any(token in DECLARATIVE_TOKENS for token in tokens):
            return False

        if all(token in CONVERSATIONAL_FILLER_TOKENS for token in tokens):
            return False

        content_tokens = [token for token in tokens if token not in TOPIC_QUERY_STOPWORDS]
        return 1 <= len(content_tokens) <= 5

    @staticmethod
    def _strip_trailing_document_cue(text: str) -> str:
        lowered = text.lower().strip()
        suffixes = (
            " in the document",
            " in the documents",
            " in the doc",
            " in the docs",
            " in the file",
            " in the files",
            " in the pdf",
            " in the pdfs",
        )
        for suffix in suffixes:
            if lowered.endswith(suffix):
                return text[: -len(suffix)].strip()
        return text

    def _extract_topic_query(
        self,
        *,
        processed_query: str,
        intent: str,
    ) -> tuple[str | None, list[str]]:
        if intent == "no_retrieval_intent":
            return None, []

        if intent == "topic_query":
            return processed_query, []

        lowered = processed_query.lower().strip()
        rewrite_notes: list[str] = []
        candidate: str | None = None

        for pattern in (ABOUT_DOCUMENT_PATTERN, IN_DOCUMENT_PATTERN, MENTIONS_PATTERN, VERB_TOPIC_PATTERN):
            match = pattern.match(lowered)
            if match:
                candidate = match.group(1).strip()
                break

        if candidate is None and COMPARE_PATTERN.match(lowered):
            return None, []

        if candidate is None:
            return None, []

        candidate = self._strip_trailing_document_cue(candidate)
        candidate = self._clean_query_text(candidate)
        if not candidate or candidate == processed_query:
            return None, []

        rewrite_notes.append("Extracted compact topic query for lexical retrieval.")
        return candidate, rewrite_notes

    @staticmethod
    def _build_retrieval_queries(processed_query: str, topic_query: str | None) -> list[str]:
        queries: list[str] = []
        for candidate in (processed_query, topic_query):
            if not candidate:
                continue
            if candidate not in queries:
                queries.append(candidate)
        return queries

    def _classify_intent(
        self,
        *,
        normalized_query: str,
        processed_query: str,
    ) -> tuple[str, bool, str]:
        processed_lower = processed_query.lower().strip()

        if processed_lower in CHITCHAT_QUERIES:
            return (
                "no_retrieval_intent",
                False,
                "Small-talk query detected; knowledge-base search is not required.",
            )

        if self._has_conversational_prefix(normalized_query):
            return (
                "no_retrieval_intent",
                False,
                "Conversational query detected; knowledge-base search is not required.",
            )

        if (
            self._has_search_intent_prefix(processed_query)
            or self._has_retrieval_verb(processed_query)
            or self._has_document_cue(processed_query)
            or self._has_document_cue(normalized_query)
        ):
            return (
                "retrieval_request",
                True,
                "Query passed intent gate and is ready for retrieval.",
            )

        if self._is_compact_topic_query(processed_query):
            return (
                "topic_query",
                True,
                "Compact topic lookup detected; retrieval is required.",
            )

        return (
            "no_retrieval_intent",
            False,
            "No clear retrieval intent detected; knowledge-base search is not required.",
        )

    @staticmethod
    def _is_acronym(token: str) -> bool:
        return bool(ACRONYM_PATTERN.fullmatch(token))

    def _build_corpus_acronym_map(self, chunks: list[ChunkRecord]) -> dict[str, str]:
        matches: dict[str, dict[str, int]] = {}

        def add(acronym: str, phrase: str) -> None:
            phrase_clean = self._normalize_whitespace(phrase)
            if len(phrase_clean) < 3:
                return
            if acronym not in matches:
                matches[acronym] = {}
            matches[acronym][phrase_clean] = matches[acronym].get(phrase_clean, 0) + 1

        for chunk in chunks:
            text = chunk.text
            for phrase, acronym in LONG_PHRASE_TO_ACRONYM.findall(text):
                add(acronym, phrase)
            for acronym, phrase in ACRONYM_TO_LONG_PHRASE.findall(text):
                add(acronym, phrase)

        acronym_map: dict[str, str] = {}
        for acronym, counts in matches.items():
            expansion = max(counts.items(), key=lambda item: item[1])[0]
            acronym_map[acronym] = expansion

        return acronym_map

    def _expand_acronyms(self, query: str, acronym_map: dict[str, str]) -> tuple[str, list[str]]:
        tokens = query.split()
        expansions: list[str] = []
        lowered_query = query.lower()

        for token in tokens:
            normalized = token.upper()
            if not self._is_acronym(normalized):
                continue
            expansion = acronym_map.get(normalized)
            if not expansion:
                continue
            if expansion.lower() in lowered_query:
                continue
            expansions.append(expansion)

        if not expansions:
            return query, []

        unique_expansions = []
        seen = set()
        for expansion in expansions:
            key = expansion.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_expansions.append(expansion)

        expanded_query = query + " " + " ".join(unique_expansions)
        expanded_query = self._normalize_whitespace(expanded_query)
        return expanded_query, unique_expansions

    @staticmethod
    def _check_policy_flag(text: str) -> str | None:
        if _PII_SSN.search(text) or _PII_CARD.search(text):
            return "pii_detected"
        if _MEDICAL_TOPIC.search(text):
            return "medical_topic"
        if _LEGAL_TOPIC.search(text):
            return "legal_topic"
        return None

    @staticmethod
    def _classify_answer_intent(text: str) -> str:
        if _ANSWER_LIST.match(text):
            return "list"
        if _ANSWER_COMPARE.match(text):
            return "compare"
        if _ANSWER_SUMMARIZE.match(text):
            return "summarize"
        return "factual"

    def process_query(
        self,
        query: str,
        *,
        chunks: list[ChunkRecord],
        rewrite_enabled: bool,
        intent_gate_enabled: bool,
    ) -> QueryProcessingResult:
        normalized = self._normalize_whitespace(query)
        if not normalized:
            return QueryProcessingResult(
                original_query=query,
                normalized_query="",
                processed_query="",
                topic_query=None,
                retrieval_queries=[],
                rewrite_applied=False,
                intent="empty",
                search_required=False,
                reason="Query is empty after normalization.",
                rewrite_notes=[],
            )

        cleaned = self._clean_query_text(normalized)
        processed = cleaned
        rewrite_notes: list[str] = []

        if rewrite_enabled:
            stripped, stripped_prefix = self._strip_filler_prefix(processed)
            if stripped_prefix and stripped:
                processed = stripped
                rewrite_notes.append("Removed conversational filler prefix.")

            acronym_map = self._build_corpus_acronym_map(chunks)
            processed, expansions = self._expand_acronyms(processed, acronym_map)
            if expansions:
                rewrite_notes.append(
                    "Expanded acronyms from corpus evidence: " + ", ".join(expansions)
                )

            processed = self._normalize_whitespace(processed)

        rewrite_applied = processed != normalized
        policy_flag = self._check_policy_flag(normalized)

        intent, search_required, reason = self._classify_intent(
            normalized_query=normalized,
            processed_query=processed,
        )
        topic_query, topic_notes = self._extract_topic_query(
            processed_query=processed,
            intent=intent,
        )
        retrieval_queries = self._build_retrieval_queries(processed, topic_query)
        rewrite_notes.extend(topic_notes)
        answer_intent = self._classify_answer_intent(processed) if search_required else "factual"

        if intent_gate_enabled and not search_required:
            return QueryProcessingResult(
                original_query=query,
                normalized_query=normalized,
                processed_query=processed,
                topic_query=topic_query,
                retrieval_queries=retrieval_queries,
                rewrite_applied=rewrite_applied,
                intent=intent,
                search_required=False,
                reason=reason,
                rewrite_notes=rewrite_notes,
                policy_flag=policy_flag,
                answer_intent=answer_intent,
            )

        return QueryProcessingResult(
            original_query=query,
            normalized_query=normalized,
            processed_query=processed,
            topic_query=topic_query,
            retrieval_queries=retrieval_queries,
            rewrite_applied=rewrite_applied,
            intent=intent,
            search_required=True,
            reason=reason,
            rewrite_notes=rewrite_notes,
            policy_flag=policy_flag,
            answer_intent=answer_intent,
        )

import math
import re
from dataclasses import dataclass

from app.schemas import ChunkRecord

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass
class KeywordHit:
    chunk: ChunkRecord
    score: float


class KeywordSearcher:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if token and token.lower() not in STOPWORDS
        ]

    def search(self, query: str, chunks: list[ChunkRecord], *, top_k: int) -> list[KeywordHit]:
        query_terms = self.tokenize(query)
        if not query_terms or not chunks:
            return []

        tokenized_chunks = [self.tokenize(chunk.text) for chunk in chunks]
        doc_count = len(tokenized_chunks)
        doc_lengths = [len(tokens) for tokens in tokenized_chunks]
        avg_doc_len = sum(doc_lengths) / doc_count if doc_count else 0.0

        doc_freq: dict[str, int] = {}
        for tokens in tokenized_chunks:
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        hits: list[KeywordHit] = []
        for chunk, tokens, doc_len in zip(chunks, tokenized_chunks, doc_lengths):
            if not tokens:
                continue

            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue

                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                norm = self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len if avg_doc_len else 1.0))
                score += idf * (freq * (self.k1 + 1.0)) / (freq + norm)

            if score > 0:
                hits.append(KeywordHit(chunk=chunk, score=score))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

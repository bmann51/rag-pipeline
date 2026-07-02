"""
BM25 cache benchmark.

Compares the old "rebuild index on every search call" behaviour against the
new "build once, serve from cache" behaviour at several corpus sizes.

Run from the repo root:
    python scripts/bench_bm25.py
"""
import random
import string
import time
from dataclasses import dataclass, field

from app.retrieval.keyword_search import KeywordSearcher
from app.schemas import ChunkRecord

WORDS = [
    "machine", "learning", "neural", "network", "retrieval", "document",
    "embedding", "vector", "search", "semantic", "keyword", "score",
    "ranking", "chunk", "pipeline", "inference", "attention", "transformer",
    "language", "model", "fine", "tuning", "generation", "query", "index",
    "corpus", "token", "context", "prompt", "answer", "dataset", "training",
    "evaluation", "benchmark", "performance", "latency", "throughput", "batch",
]

QUERIES = [
    "machine learning neural network",
    "semantic search retrieval ranking",
    "transformer language model generation",
    "vector embedding similarity score",
    "document pipeline chunking strategy",
]


def _random_chunk(chunk_id: str) -> ChunkRecord:
    n_words = random.randint(60, 120)
    text = " ".join(random.choices(WORDS, k=n_words))
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_file="bench.pdf",
        page_start=1,
        page_end=1,
        text=text,
        char_count=len(text),
        metadata={},
        created_at="2025-01-01T00:00:00Z",
    )


def _simulate_old(chunks: list[ChunkRecord], queries: list[str], *, top_k: int) -> float:
    """Re-create the old behaviour: rebuild the full index on every search call."""
    searcher = KeywordSearcher()
    start = time.perf_counter()
    for q in queries:
        # Force a rebuild every call by invalidating before each search.
        searcher.invalidate_cache()
        searcher.search(q, chunks, top_k=top_k)
    return time.perf_counter() - start


def _simulate_new(chunks: list[ChunkRecord], queries: list[str], *, top_k: int) -> float:
    """New behaviour: index built once on the first search, reused for the rest."""
    searcher = KeywordSearcher()
    start = time.perf_counter()
    for q in queries:
        searcher.search(q, chunks, top_k=top_k)
    return time.perf_counter() - start


def run(corpus_size: int, n_queries: int = 20, top_k: int = 10, warmup: int = 2) -> None:
    random.seed(42)
    chunks = [_random_chunk(f"chunk-{i}") for i in range(corpus_size)]
    queries = [random.choice(QUERIES) for _ in range(n_queries)]

    # Warmup (JIT, import caches, etc.)
    for _ in range(warmup):
        _simulate_old(chunks, queries[:2], top_k=top_k)
        _simulate_new(chunks, queries[:2], top_k=top_k)

    old_s = _simulate_old(chunks, queries, top_k=top_k)
    new_s = _simulate_new(chunks, queries, top_k=top_k)

    old_ms = old_s * 1000
    new_ms = new_s * 1000
    speedup = old_s / new_s if new_s > 0 else float("inf")

    print(
        f"  corpus={corpus_size:>6,}  queries={n_queries}  "
        f"old={old_ms:7.1f} ms  new={new_ms:7.1f} ms  "
        f"speedup={speedup:.1f}x"
    )


if __name__ == "__main__":
    print("BM25 cache benchmark\n")
    print(f"  {'corpus':>6}  {'queries'}  {'old (rebuild every call)':>23}  {'new (cached)':>12}  speedup")
    print("  " + "-" * 75)
    for size in [200, 500, 1_000, 2_000, 5_000]:
        run(size)

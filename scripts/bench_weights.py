"""
Keyword/semantic weight sweep benchmark.

Sweeps keyword_weight (α) from 0.0 → 1.0 (semantic_weight = 1-α) and
measures hit@5 against 10 hand-crafted queries spanning the ingested corpus.

API usage: one embed_texts() call for the 10 queries — no generation calls.
Chunk embeddings are read from the on-disk cache.

Run:
    python scripts/bench_weights.py
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.retrieval.keyword_search import KeywordSearcher
from app.retrieval.embeddings import EmbeddingClient
from app.storage.chunk_reader import ChunkReader
from app.storage.embedding_store import EmbeddingStore

# ── test cases ────────────────────────────────────────────────────────────────
# Each case: (query, type, expected_keywords)
# type: "keyword" = exact-term query; "semantic" = conceptual/paraphrase query
# expected_keywords: strings that should appear in the text of a correct chunk

CASES = [
    (
        "Who is Fëanor and what did he create?",
        "keyword",
        ["silmaril", "noldor", "jewel", "feanor"],
    ),
    (
        "What is the Ainulindalë?",
        "keyword",
        ["music", "ilúvatar", "ainur", "creation"],
    ),
    (
        "What motivated the elves to sail west?",
        "semantic",
        ["valinor", "undying", "longing", "sea", "west"],
    ),
    (
        "What underlying themes does the Silmarillion explore?",
        "semantic",
        ["pride", "doom", "fate", "mortality", "immortal"],
    ),
    (
        "What is backpropagation?",
        "keyword",
        ["gradient", "chain rule", "derivative", "backward"],
    ),
    (
        "Explain gradient descent",
        "keyword",
        ["learning rate", "update", "loss", "step", "minimize"],
    ),
    (
        "How do we stop a model from memorizing the training data?",
        "semantic",
        ["overfitting", "regularization", "dropout", "generalization"],
    ),
    (
        "Why do deeper networks perform better on complex tasks?",
        "semantic",
        ["representation", "layer", "feature", "depth", "hidden"],
    ),
    (
        "What is a dataframe?",
        "keyword",
        ["row", "column", "pandas", "table", "dataframe"],
    ),
    (
        "What should you do first when you get a new dataset?",
        "semantic",
        ["explore", "clean", "visualize", "understand", "inspect"],
    ),
]

TOP_K_SEARCH = 60   # candidates from each retriever before fusion
HIT_K        = 5    # hit@K metric


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    na    = math.sqrt(sum(x * x for x in a))
    nb    = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def hit_at_k(
    chunks_by_id: dict,
    ranked_ids: list[str],
    expected_kws: list[str],
    k: int,
) -> bool:
    for chunk_id in ranked_ids[:k]:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        text = chunk.text.lower()
        if any(kw.lower() in text for kw in expected_kws):
            return True
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()

    # 1. Load chunks
    print("Loading chunks …")
    chunks = ChunkReader(record_dir=settings.record_dir).load_chunks()
    chunks_by_id = {c.chunk_id: c for c in chunks}
    print(f"  {len(chunks)} chunks")

    # 2. Load cached embeddings
    print("Loading cached embeddings …")
    t0 = time.perf_counter()
    cached = EmbeddingStore(record_dir=settings.record_dir).load_embeddings()
    print(f"  {len(cached)} vectors  ({(time.perf_counter()-t0)*1000:.0f} ms)")

    # 3. Embed queries + any missing chunk embeddings
    client = EmbeddingClient(
        api_key=settings.mistral_api_key,
        model=settings.mistral_embedding_model,
        batch_size=settings.embedding_batch_size,
        min_request_interval_seconds=settings.embedding_min_request_interval_seconds,
        max_retries_on_rate_limit=settings.embedding_max_retries_on_rate_limit,
        retry_base_delay_seconds=settings.embedding_retry_base_delay_seconds,
    )

    missing = [c for c in chunks if c.chunk_id not in cached]
    if missing:
        print(f"Generating embeddings for {len(missing)} chunks (not yet cached) …")
        # Truncate to ~6000 chars (~1500 tokens) to stay under the 8192-token API limit.
        MAX_CHARS = 6000
        vectors = client.embed_texts([c.text[:MAX_CHARS] for c in missing])
        for chunk, vec in zip(missing, vectors):
            cached[chunk.chunk_id] = vec
        print(f"  Done")

    print(f"Embedding {len(CASES)} queries via Mistral API …")
    query_texts  = [c[0] for c in CASES]
    query_vecs   = client.embed_texts(query_texts)
    print("  Done\n")

    # 4. BM25 scores for every query (pure CPU, no API)
    keyword_searcher = KeywordSearcher()
    bm25: list[dict[str, float]] = []
    for query in query_texts:
        hits = keyword_searcher.search(query, chunks, top_k=TOP_K_SEARCH)
        bm25.append({h.chunk.chunk_id: h.score for h in hits})

    # 5. Cosine similarity for every query (pure CPU, uses cached vectors)
    semantic: list[dict[str, float]] = []
    valid_ids = set(cached) & set(chunks_by_id)
    for qvec in query_vecs:
        scores = {cid: cosine(qvec, cached[cid]) for cid in valid_ids}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_SEARCH]
        semantic.append(dict(top))

    # 6. Sweep
    alphas = [round(a * 0.1, 1) for a in range(11)]

    n  = len(CASES)
    kw_cases  = [i for i, c in enumerate(CASES) if c[1] == "keyword"]
    sem_cases = [i for i, c in enumerate(CASES) if c[1] == "semantic"]

    print(f"Weight sweep  —  {n} queries, hit@{HIT_K}")
    print(f"  (α = keyword_weight,  semantic_weight = 1−α)\n")
    header = f"  {'α':>4}  {'1-α':>4}  {'all':>5}  {'keyword':>8}  {'semantic':>9}  {'overlap':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    results: dict[float, int] = {}

    for alpha in alphas:
        sem_w = round(1.0 - alpha, 1)
        hits_total = 0
        hits_kw    = 0
        hits_sem   = 0
        overlap_counts: list[int] = []

        for i, (query, qtype, expected_kws) in enumerate(CASES):
            norm_kw  = normalize(bm25[i])
            norm_sem = normalize(semantic[i])

            both = set(norm_kw) & set(norm_sem)
            overlap_counts.append(len(both))

            all_ids = set(norm_kw) | set(norm_sem)
            fused: dict[str, float] = {}
            for cid in all_ids:
                in_kw  = cid in norm_kw
                in_sem = cid in norm_sem
                if in_kw and in_sem:
                    fused[cid] = alpha * norm_kw[cid] + sem_w * norm_sem[cid]
                elif in_kw:
                    fused[cid] = norm_kw[cid]
                else:
                    fused[cid] = norm_sem[cid]

            ranked = sorted(fused, key=lambda k: fused[k], reverse=True)
            hit = hit_at_k(chunks_by_id, ranked, expected_kws, HIT_K)

            if hit:
                hits_total += 1
                if qtype == "keyword":
                    hits_kw += 1
                else:
                    hits_sem += 1

        results[alpha] = hits_total
        avg_overlap = sum(overlap_counts) / len(overlap_counts)
        marker = "  ← current" if alpha == 0.4 else ""
        print(
            f"  {alpha:>4.1f}  {sem_w:>4.1f}  "
            f"{hits_total:>2}/{n}  "
            f"{hits_kw:>3}/{len(kw_cases)}        "
            f"{hits_sem:>3}/{len(sem_cases)}       "
            f"{avg_overlap:>6.1f}"
            f"{marker}"
        )

    # 7. Summary
    best_alpha  = max(results, key=lambda a: results[a])
    best_score  = results[best_alpha]
    curr_score  = results[0.4]
    delta       = abs(best_alpha - 0.4)

    print()
    print(f"Current  α=0.4  →  {curr_score}/{n} hits")
    print(f"Best     α={best_alpha:.1f}  →  {best_score}/{n} hits", end="")
    if best_alpha == 0.4:
        print("  (current setting is optimal)")
    elif delta <= 0.1:
        print(f"  (within 1 step — marginal gain from tuning)")
    else:
        print(f"  (Δ={delta:.1f} — consider setting keyword_weight={best_alpha} in config)")


if __name__ == "__main__":
    main()

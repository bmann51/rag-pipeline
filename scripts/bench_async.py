"""
Async concurrency benchmark.

Shows that making /query async def lets the event loop handle many requests
concurrently instead of serialising them on a thread-pool worker.

We patch the Mistral SDK with a fake client that sleeps MOCK_DELAY_S to
simulate a realistic network round-trip, so the benchmark is deterministic
and doesn't consume API quota.

Run from the repo root:
    python scripts/bench_async.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import time
import unittest.mock as mock
from dataclasses import dataclass
from typing import Any

import httpx
from httpx import AsyncClient

MOCK_DELAY_S = 0.15   # 150 ms per Mistral call — realistic single-hop latency
N_CONCURRENT = 8      # concurrent requests to fire at the endpoint

# ---------------------------------------------------------------------------
# Fake Mistral responses
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1024


@dataclass
class _FakeEmbeddingData:
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingData]


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeChatResponse:
    choices: list[_FakeChoice]


async def _fake_embed_async(*args: Any, **kwargs: Any) -> _FakeEmbeddingResponse:
    await asyncio.sleep(MOCK_DELAY_S)
    n = len(kwargs.get("inputs", args[0] if args else []))
    return _FakeEmbeddingResponse(data=[_FakeEmbeddingData(embedding=FAKE_EMBEDDING) for _ in range(n)])


async def _fake_chat_async(*args: Any, **kwargs: Any) -> _FakeChatResponse:
    await asyncio.sleep(MOCK_DELAY_S)
    chunk_id = "fake-chunk-id"
    return _FakeChatResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=f"Answer [{chunk_id}]"))]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_mistral():
    """Return a context manager that replaces Mistral network calls with fakes."""
    return mock.patch.multiple(
        "app.retrieval.embeddings.EmbeddingClient",
        embed_text_async=_fake_embed_async,
        embed_texts_async=_fake_embed_async,
    )


def _patch_answer_generator():
    return mock.patch(
        "app.retrieval.answer_generator.AnswerGenerator._chat_complete_async",
        new=_fake_chat_async,
    )


async def _fire(client: AsyncClient, query: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    resp = await client.post("/query", json={"query": query})
    return resp.status_code, time.perf_counter() - t0


async def run_concurrent(client: AsyncClient, query: str, n: int) -> list[float]:
    tasks = [_fire(client, query) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    return [elapsed for _, elapsed in results]


async def run_sequential(client: AsyncClient, query: str, n: int) -> list[float]:
    times: list[float] = []
    for _ in range(n):
        _, elapsed = await _fire(client, query)
        times.append(elapsed)
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # Import app here so patches are applied before any client is instantiated.
    from app.main import app  # noqa: PLC0415

    # A query that passes the intent gate and triggers embedding + generation.
    query = "what topics are covered in the documents"

    print(f"Async concurrency benchmark")
    print(f"  mock Mistral delay : {MOCK_DELAY_S * 1000:.0f} ms per call")
    print(f"  concurrent requests: {N_CONCURRENT}")
    print()

    with _patch_mistral(), _patch_answer_generator():
        async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # Warmup
            await run_sequential(client, query, n=2)

            # Sequential baseline
            t0 = time.perf_counter()
            seq_times = await run_sequential(client, query, n=N_CONCURRENT)
            seq_wall = time.perf_counter() - t0

            # Concurrent
            t0 = time.perf_counter()
            con_times = await run_concurrent(client, query, n=N_CONCURRENT)
            con_wall = time.perf_counter() - t0

    def _stats(label: str, times: list[float], wall: float) -> None:
        avg = sum(times) / len(times) * 1000
        p95 = sorted(times)[int(len(times) * 0.95)] * 1000
        print(
            f"  {label:<12}  wall={wall * 1000:6.0f} ms  "
            f"avg/req={avg:5.0f} ms  p95={p95:5.0f} ms"
        )

    print(f"  {'mode':<12}  {'wall time':>9}  {'avg / req':>9}  {'p95':>8}")
    print("  " + "-" * 55)
    _stats("sequential", seq_times, seq_wall)
    _stats("concurrent", con_times, con_wall)

    speedup = seq_wall / con_wall if con_wall > 0 else float("inf")
    print(f"\n  Concurrency speedup: {speedup:.1f}x  (ideal = {N_CONCURRENT}x)")
    print()
    print("  Note: speedup < ideal because the Mistral embedding + chat calls")
    print("  each add latency that stacks even in async when awaited in sequence")
    print("  within a single request. The gain is that the *server* is not")
    print("  blocking a thread while waiting — it can serve other requests.")


if __name__ == "__main__":
    asyncio.run(main())

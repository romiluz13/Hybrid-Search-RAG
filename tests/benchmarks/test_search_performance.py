"""
Search performance benchmarks.

Measures query latency and throughput for different search modes.

Requires pytest-benchmark: pip install pytest-benchmark

Note: These tests are synchronous because pytest-benchmark's benchmark()
callable does not support async functions. asyncio.run() is used to bridge
the async RAG calls into the sync benchmark harness. This is safe because
these are sync test functions (no running event loop to conflict with).
"""

import asyncio

import pytest

# Skip entire module if pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark
def test_hybrid_search_latency(benchmark, rag, benchmark_queries):
    """Benchmark hybrid search latency."""
    query = benchmark_queries[0]

    async def search():
        return await rag.query(query=query, mode="hybrid", top_k=10)

    result = benchmark(lambda: asyncio.run(search()))
    assert result, "Search should return a non-empty response"


@pytest.mark.benchmark
def test_naive_search_latency(benchmark, rag, benchmark_queries):
    """Benchmark the direct retrieval baseline exposed by the wrapper API."""
    query = benchmark_queries[0]

    async def search():
        return await rag.query(query=query, mode="naive", top_k=10)

    result = benchmark(lambda: asyncio.run(search()))
    assert result, "Search should return a non-empty response"


@pytest.mark.benchmark
def test_batch_queries(benchmark, rag, benchmark_queries):
    """Benchmark batch query throughput."""

    async def batch_search():
        tasks = [rag.query(query=q, mode="hybrid", top_k=10) for q in benchmark_queries]
        return await asyncio.gather(*tasks)

    results = benchmark(lambda: asyncio.run(batch_search()))
    assert len(results) == len(benchmark_queries)

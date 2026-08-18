"""Fast MongoDB-backed integration smoke tests for the public wrapper.

These tests intentionally stay smaller than the real seeded live gate. On the
blessed local preview stack (`atlas-local:preview` on port 27018), native hybrid
ranking quality is asserted in ``tests/e2e_real_test.py`` with real providers and
realistic seeded data. This file focuses on deterministic wrapper/API behavior.
"""

import asyncio

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_and_query_returns_context(rag, test_documents):
    """Mix-mode retrieval returns structured context on the fast local gate."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"])

    for _ in range(20):
        result = await rag.query_data(
            query="What is MongoDB Atlas?",
            mode="mix",
            top_k=5,
        )
        if result["metadata"].get("failure_reason") != "no_results":
            break
        await asyncio.sleep(0.5)
    context = result["context"]

    assert result["metadata"].get("failure_reason") != "no_results"
    assert isinstance(context, str), "Should return retrieved context"
    assert len(context) > 0, "Context should not be empty"
    assert isinstance(result["metadata"], dict), "Should expose retrieval metadata"
    assert result["metadata"]["query_mode"] == "mix"
    assert result["metadata"]["fallback_used"] is False
    assert result["metadata"]["processing_info"]["final_chunks_count"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_with_sources_returns_context(rag, test_documents):
    """Source-aware querying exposes citations even on the fast local gate."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"])

    # Query with sources
    result = await rag.query_with_sources(
        query="What is vector search?",
        mode="mix",
        top_k=3,
    )

    assert isinstance(result["answer"], str), "Should return an answer"
    assert len(result["answer"]) > 0, "Answer should not be empty"
    assert isinstance(result["context"], str), "Should return source context"
    assert isinstance(result["references"], list), "Should expose structured references"
    assert isinstance(result["metadata"], dict), "Should expose retrieval metadata"
    assert (
        "search_types" in result["metadata"]
        or result["metadata"].get("failure_reason") == "no_results"
        or "query_mode" in result["metadata"]
    ), "Should expose retrieval diagnostics or an explicit mode/no-results reason"
    if result["context"] and "search_types" in result["metadata"]:
        assert len(result["references"]) > 0, "Contextful answers should retain sources"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_modes(rag, test_documents):
    """Supported public modes return deterministic text output."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"])

    query = "MongoDB search features"

    for mode in ["naive", "mix", "hybrid"]:
        response = await rag.query(query=query, mode=mode, top_k=3)
        assert isinstance(response, str), f"{mode} mode should return text output"
        assert len(response) > 0, f"{mode} mode should return deterministic output"

    data = await rag.query_data(query=query, mode="mix", top_k=3)
    assert isinstance(data["metadata"], dict)
    assert (
        "fallback_used" in data["metadata"]
        or data["metadata"].get("failure_reason") == "no_results"
        or "query_mode" in data["metadata"]
    )

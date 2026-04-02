"""
Integration test for full RAG pipeline.

Tests end-to-end functionality with real MongoDB connection.
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_and_query_returns_context(rag, test_documents):
    """Test document ingestion and retrieval-only query flow."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"])

    # Query without LLM generation
    context = await rag.query(
        query="What is MongoDB Atlas?",
        mode="mix",
        top_k=5,
        only_context=True,
    )

    # Verify
    assert isinstance(context, str), "Should return retrieved context"
    assert len(context) > 0, "Context should not be empty"
    assert "mongodb" in context.lower() or "atlas" in context.lower(), (
        "Context should be relevant to the ingested documents"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_with_sources_returns_context(rag, test_documents):
    """Test source-aware query flow in retrieval-only mode."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"])

    # Query with sources
    result = await rag.query_with_sources(
        query="What is vector search?",
        mode="mix",
        top_k=3,
    )

    # Verify
    assert isinstance(result["answer"], str), "Should return an answer"
    assert len(result["answer"]) > 0, "Answer should not be empty"
    assert isinstance(result["context"], str), "Should return source context"
    assert (
        "vector" in result["answer"].lower() or "search" in result["answer"].lower()
    ), "Answer should be relevant to the query"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_modes(rag, test_documents):
    """Test the supported public search modes in retrieval-only mode."""
    # Ingest documents
    for doc in test_documents:
        await rag.insert(doc["content"])

    query = "MongoDB search features"

    # Test each mode
    for mode in ["naive", "hybrid", "mix"]:
        context = await rag.query(query=query, mode=mode, top_k=3, only_context=True)
        assert isinstance(context, str), f"{mode} mode should return context text"
        assert len(context) > 0, f"{mode} mode should return non-empty output"
        assert "mongodb" in context.lower() or "search" in context.lower(), (
            f"{mode} mode should remain relevant"
        )

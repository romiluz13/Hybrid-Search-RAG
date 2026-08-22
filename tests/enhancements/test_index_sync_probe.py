"""Tests for index sync functional probe.

Inspired by the Anthropic CMA cookbook's ``wait_for_index_sync`` — verifies
that the probe correctly detects when Atlas Search indexes have ingested
seeded documents, not just when they report ``queryable=True``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class _FakeCursor:
    """Minimal async cursor that returns preset results."""

    def __init__(self, results: list):
        self._results = results

    async def to_list(self, length=None):
        return self._results[:length] if length else self._results


def _make_mock_storage(
    vector_results_sequence: list[list],
    text_results_sequence: list[list],
    embedding_dim: int = 1024,
):
    """Create a mock storage object with a ``probe_index_sync`` method.

    The mock's ``aggregate`` returns successive results from the provided
    sequences, allowing tests to simulate indexes becoming synced over time.
    """
    storage = MagicMock()
    storage._data = AsyncMock()
    storage.workspace = "test"
    storage._index_name = "vector_knn_index_test_chunks"
    storage.embedding_func = MagicMock()
    storage.embedding_func.embedding_dim = embedding_dim

    # _text_index_name is a regular method, not async
    storage._text_index_name = MagicMock(
        return_value="text_search_index_test_chunks"
    )

    call_state = {"vector_idx": 0, "text_idx": 0}

    async def fake_aggregate(pipeline, **kwargs):
        # Determine if this is a vector or text probe
        pipeline_str = str(pipeline)
        if "$vectorSearch" in pipeline_str:
            idx = call_state["vector_idx"]
            call_state["vector_idx"] += 1
            results = (
                vector_results_sequence[idx]
                if idx < len(vector_results_sequence)
                else vector_results_sequence[-1]
            )
            return _FakeCursor(results)
        elif "$search" in pipeline_str:
            idx = call_state["text_idx"]
            call_state["text_idx"] += 1
            results = (
                text_results_sequence[idx]
                if idx < len(text_results_sequence)
                else text_results_sequence[-1]
            )
            return _FakeCursor(results)
        return _FakeCursor([])

    storage._data.aggregate = fake_aggregate

    # Attach the real probe_index_sync method from the actual class
    from hybridrag.engine.kg.mongo_impl import MongoVectorDBStorage

    storage.probe_index_sync = MongoVectorDBStorage.probe_index_sync.__get__(
        storage, type(storage)
    )

    return storage


# ---------------------------------------------------------------------------
# Successful sync detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_detects_immediate_sync() -> None:
    """Both indexes return results on first probe → immediate success."""
    storage = _make_mock_storage(
        vector_results_sequence=[[{"_id": "doc1"}]],
        text_results_sequence=[[{"_id": "doc1"}]],
    )

    result = await storage.probe_index_sync(timeout_seconds=5, poll_interval_seconds=0.1)

    assert result == {"vector_synced": True, "text_synced": True}


@pytest.mark.asyncio
async def test_probe_detects_delayed_sync() -> None:
    """Indexes return empty first, then results → probe waits and succeeds."""
    storage = _make_mock_storage(
        vector_results_sequence=[[], [{"_id": "doc1"}]],
        text_results_sequence=[[], [{"_id": "doc1"}]],
    )

    result = await storage.probe_index_sync(timeout_seconds=10, poll_interval_seconds=0.05)

    assert result == {"vector_synced": True, "text_synced": True}


@pytest.mark.asyncio
async def test_probe_detects_partial_sync() -> None:
    """Vector syncs but text never does → partial result."""
    storage = _make_mock_storage(
        vector_results_sequence=[[{"_id": "doc1"}]],
        text_results_sequence=[[]],  # never returns results
    )

    result = await storage.probe_index_sync(timeout_seconds=1, poll_interval_seconds=0.05)

    assert result == {"vector_synced": True, "text_synced": False}


# ---------------------------------------------------------------------------
# Timeout behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_times_out_when_never_synced() -> None:
    """Neither index returns results → both report False after timeout."""
    storage = _make_mock_storage(
        vector_results_sequence=[[]],
        text_results_sequence=[[]],
    )

    result = await storage.probe_index_sync(timeout_seconds=0.5, poll_interval_seconds=0.1)

    assert result == {"vector_synced": False, "text_synced": False}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_swallows_aggregate_errors_and_retries() -> None:
    """Aggregate failures should not crash the probe; it should retry."""
    storage = _make_mock_storage(
        vector_results_sequence=[[{"_id": "doc1"}]],
        text_results_sequence=[[{"_id": "doc1"}]],
    )

    call_count = {"n": 0}
    original_aggregate = storage._data.aggregate

    async def flaky_aggregate(pipeline, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError("index not ready")
        return await original_aggregate(pipeline, **kwargs)

    storage._data.aggregate = flaky_aggregate

    result = await storage.probe_index_sync(timeout_seconds=5, poll_interval_seconds=0.05)

    assert result == {"vector_synced": True, "text_synced": True}


@pytest.mark.asyncio
async def test_probe_raises_on_uninitialized_storage() -> None:
    """Calling probe before initialize() should raise ValueError."""
    storage = _make_mock_storage(
        vector_results_sequence=[[]],
        text_results_sequence=[[]],
    )
    storage._data = None

    with pytest.raises(ValueError, match="not initialized"):
        await storage.probe_index_sync(timeout_seconds=1)


# ---------------------------------------------------------------------------
# Default vector generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_uses_zero_vector_when_none_provided() -> None:
    """When sample_vector is None, a zero vector of embedding_dim is used."""
    storage = _make_mock_storage(
        vector_results_sequence=[[{"_id": "doc1"}]],
        text_results_sequence=[[{"_id": "doc1"}]],
        embedding_dim=512,
    )

    captured_pipelines: list = []

    original_aggregate = storage._data.aggregate

    async def capturing_aggregate(pipeline, **kwargs):
        captured_pipelines.append(pipeline)
        return await original_aggregate(pipeline, **kwargs)

    storage._data.aggregate = capturing_aggregate

    await storage.probe_index_sync(timeout_seconds=5, poll_interval_seconds=0.1)

    # Find the vector search pipeline
    vector_pipeline = next(
        p for p in captured_pipelines if "$vectorSearch" in str(p)
    )
    vs_stage = next(s for s in vector_pipeline if "$vectorSearch" in s)
    query_vector = vs_stage["$vectorSearch"]["queryVector"]

    assert len(query_vector) == 512
    assert all(v == 0.0 for v in query_vector)


# ---------------------------------------------------------------------------
# HybridRAG facade integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_verify_index_sync_delegates_to_storages() -> None:
    """HybridRAG.verify_index_sync should call probe on each storage."""
    from hybridrag.core.rag import HybridRAG

    # Create a minimal mock rag instance
    rag = MagicMock(spec=HybridRAG)

    # Mock the storages
    mock_chunks = MagicMock()
    mock_chunks.probe_index_sync = AsyncMock(
        return_value={"vector_synced": True, "text_synced": True}
    )
    mock_entities = MagicMock()
    mock_entities.probe_index_sync = AsyncMock(
        return_value={"vector_synced": True, "text_synced": True}
    )
    mock_relationships = MagicMock()
    # relationships has no probe_index_sync → should be skipped
    del mock_relationships.probe_index_sync

    rag._ensure_initialized.return_value = MagicMock(
        chunks_vdb=mock_chunks,
        entities_vdb=mock_entities,
        relationships_vdb=mock_relationships,
    )

    # Bind the real method
    rag.verify_index_sync = HybridRAG.verify_index_sync.__get__(rag, type(rag))

    result = await rag.verify_index_sync(timeout_seconds=5, poll_interval_seconds=0.1)

    assert "chunks" in result
    assert "entities" in result
    assert "relationships" not in result  # skipped (no probe method)
    assert result["chunks"] == {"vector_synced": True, "text_synced": True}
    assert result["entities"] == {"vector_synced": True, "text_synced": True}

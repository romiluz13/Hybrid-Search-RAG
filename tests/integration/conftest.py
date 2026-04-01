"""
Integration test configuration.

Provides fixtures for integration tests that require MongoDB connection.
"""

import os

import pytest

from hybridrag import create_hybridrag
from hybridrag.config import get_settings


@pytest.fixture
def skip_if_no_mongodb():
    """Skip test if MongoDB connection not available."""
    if not os.getenv("MONGODB_URI"):
        pytest.skip("MONGODB_URI not set - skipping integration test")


@pytest.fixture
async def rag(skip_if_no_mongodb):
    """
    Create HybridRAG instance for integration tests.

    Yields:
        Initialized HybridRAG instance
    """
    settings = get_settings()
    rag_instance = await create_hybridrag(settings=settings)
    yield rag_instance
    # Cleanup test data (best-effort)
    try:
        await rag_instance.clear_collection()
    except Exception:
        pass  # Best-effort cleanup -- don't fail tests on cleanup errors


@pytest.fixture
def test_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "MongoDB Atlas is a fully managed cloud database service.",
            "metadata": {"source": "docs", "category": "platform"},
        },
        {
            "content": "Vector search enables semantic similarity searches using embeddings.",
            "metadata": {"source": "docs", "category": "features"},
        },
        {
            "content": "Atlas Search provides full-text search with fuzzy matching.",
            "metadata": {"source": "docs", "category": "features"},
        },
    ]

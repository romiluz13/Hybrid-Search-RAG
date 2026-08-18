"""
Integration test configuration.

Provides fixtures for integration tests that require MongoDB connection.
"""

import os
import uuid

import pytest
from pydantic import SecretStr

from hybridrag import create_hybridrag
from hybridrag.config import get_settings
from tests.conftest import _create_test_client, _resolve_test_mongodb_uri


def _openai_llm_overrides(base_settings) -> dict:
    """Build blessed-stack LLM overrides for integration tests.

    If OPENAI_API_KEY is not set, integration tests fall back to retrieval-only mode.
    For OpenAI-compatible gateways, set OPENAI_BASE_URL and OPENAI_EXTRA_HEADERS.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return {"enable_llm": False}

    return {
        "enable_llm": True,
        "llm_provider": "openai",
        "openai_api_key": SecretStr(openai_key),
        "openai_model": os.getenv("OPENAI_MODEL", base_settings.openai_model),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", base_settings.openai_base_url),
        "openai_extra_headers": os.getenv(
            "OPENAI_EXTRA_HEADERS", base_settings.openai_extra_headers
        ),
    }


@pytest.fixture
async def rag(require_mongodb_uri, tmp_path):
    """
    Create HybridRAG instance for integration tests.

    Uses real MongoDB (atlas-local) and optionally a real OpenAI-compatible LLM.
    Requires VOYAGE_API_KEY (Voyage is the only supported embedding provider);
    skips when it is not configured so CI stays green on the Mongo-only path.
    The live release gate (tests/e2e_real_test.py) still validates the full
    blessed stack when secrets are present.
    """
    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip(
            "VOYAGE_API_KEY not set; integration tests that ingest real "
            "embeddings are skipped. Set the secret to run the full pipeline."
        )
    base_settings = get_settings()
    # Each test gets a unique database to avoid cross-test contamination
    test_db = f"hybridrag_integ_{uuid.uuid4().hex[:8]}"
    overrides = {
        "mongodb_uri": SecretStr(_resolve_test_mongodb_uri()),
        "mongodb_database": test_db,
        "mongodb_workspace": "test",
        "enable_rerank": False,
        **_openai_llm_overrides(base_settings),
    }
    settings = base_settings.model_copy(update=overrides)

    # Reset ALL engine singletons to avoid event loop contamination between tests
    from hybridrag.engine.kg.mongo_impl import ClientManager
    from hybridrag.engine.kg.shared_storage import finalize_share_data

    await ClientManager.reset()
    finalize_share_data()

    from hybridrag.core.mongodb_client import close_shared_client

    close_shared_client()

    # Force workspace env var to match settings (prevents stale env override)
    os.environ["MONGODB_WORKSPACE"] = settings.mongodb_workspace
    rag_instance = await create_hybridrag(
        settings=settings,
        working_dir=str(tmp_path / "hybridrag_workspace"),
    )
    applied_indexes = await rag_instance.apply_search_index_plans()
    assert {
        plan["index_name"] for plan in applied_indexes if plan["index_kind"] == "vector"
    } == {
        "vector_knn_index_test_chunks",
        "vector_knn_index_test_entities",
        "vector_knn_index_test_relationships",
    }
    await rag_instance.wait_for_search_indexes(
        [plan["index_name"] for plan in applied_indexes],
        timeout_seconds=180,
    )
    yield rag_instance

    # Teardown: reset ALL engine singletons, then drop test database
    await ClientManager.reset()
    finalize_share_data()
    close_shared_client()
    client = _create_test_client(settings)
    await client.drop_database(settings.mongodb_database)
    await client.close()


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

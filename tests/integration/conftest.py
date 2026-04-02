"""
Integration test configuration.

Provides fixtures for integration tests that require MongoDB connection.
"""

import json
import os
import uuid

import pytest
from pydantic import SecretStr

from hybridrag import create_hybridrag
from hybridrag.config import get_settings
from tests.conftest import _create_test_client, _resolve_test_mongodb_uri


def _grove_llm_overrides() -> dict:
    """Build settings overrides for Grove Gateway LLM, if configured.

    Returns empty dict if GROVE_API_KEY is not set, allowing tests to fall
    back to enable_llm=False (retrieval-only mode).
    """
    grove_key = os.getenv("GROVE_API_KEY") or os.getenv("API_KEY")
    if not grove_key:
        return {"enable_llm": False}

    grove_base = os.getenv(
        "GROVE_BASE_URL",
        "https://grove-gateway-prod.azure-api.net/grove-foundry-prod/openai/v1",
    )
    grove_model = os.getenv("GROVE_MODEL", "gpt-5.4")

    return {
        "enable_llm": True,
        "llm_provider": "openai",
        "openai_api_key": SecretStr(grove_key),
        "openai_model": grove_model,
        "openai_base_url": grove_base,
        "openai_extra_headers": json.dumps({"api-key": grove_key}),
    }


@pytest.fixture
async def rag(require_mongodb_uri, tmp_path):
    """
    Create HybridRAG instance for integration tests.

    Uses real MongoDB (atlas-local) and optionally real LLM (Grove Gateway).
    """
    base_settings = get_settings()
    # Each test gets a unique database to avoid cross-test contamination
    test_db = f"hybridrag_integ_{uuid.uuid4().hex[:8]}"
    overrides = {
        "mongodb_uri": SecretStr(_resolve_test_mongodb_uri()),
        "mongodb_database": test_db,
        "mongodb_workspace": "test",
        **_grove_llm_overrides(),
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

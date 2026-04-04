"""Shared pytest fixtures for HybridRAG test suites."""

from __future__ import annotations

import os
import socket
import uuid

import pytest
from pydantic import SecretStr
from pymongo import AsyncMongoClient

from hybridrag.config import Settings, get_settings

DEFAULT_LOCAL_MONGODB_URI = "mongodb://localhost:27018/?directConnection=true"
FALLBACK_LOCAL_MONGODB_URI = "mongodb://localhost:27017/?directConnection=true"


def _local_mongodb_is_available(port: int) -> bool:
    """Return True when a local MongoDB server is reachable on the given port."""
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


def _resolve_test_mongodb_uri() -> str:
    """Resolve the MongoDB URI for test runs.

    Prefer Atlas Local Preview on 27018, then fall back to a generic local MongoDB.
    """
    if os.getenv("HYBRIDRAG_TEST_MONGODB_URI"):
        return os.getenv("HYBRIDRAG_TEST_MONGODB_URI") or DEFAULT_LOCAL_MONGODB_URI

    if _local_mongodb_is_available(27018):
        return DEFAULT_LOCAL_MONGODB_URI

    if _local_mongodb_is_available(27017):
        return FALLBACK_LOCAL_MONGODB_URI

    return os.getenv("MONGODB_URI") or DEFAULT_LOCAL_MONGODB_URI


@pytest.fixture
def require_mongodb_uri() -> None:
    """Fail fast when MongoDB-backed tests run without a connection string."""
    if os.getenv("HYBRIDRAG_TEST_MONGODB_URI") or os.getenv("MONGODB_URI"):
        return

    if _local_mongodb_is_available(27018) or _local_mongodb_is_available(27017):
        return

    try:
        with socket.create_connection(("localhost", 27018), timeout=1):
            return
    except OSError as exc:
        pytest.fail(
            "No MongoDB test URI provided and no local MongoDB found on "
            f"localhost:27018 ({exc})"
        )


@pytest.fixture
def mongodb_test_settings(require_mongodb_uri: None) -> Settings:
    """Create isolated MongoDB settings for a single test."""
    base_settings = get_settings()
    suffix = uuid.uuid4().hex[:8]

    return base_settings.model_copy(
        update={
            "mongodb_uri": SecretStr(_resolve_test_mongodb_uri()),
            "mongodb_database": os.getenv(
                "HYBRIDRAG_TEST_DB", f"hybridrag_test_{suffix}"
            ),
            "mongodb_workspace": os.getenv(
                "HYBRIDRAG_TEST_WORKSPACE", f"pytest_{suffix}"
            ),
        }
    )


def _create_test_client(settings: Settings) -> AsyncMongoClient:
    """Create a test MongoDB client with pool settings from the given Settings."""
    return AsyncMongoClient(
        settings.mongodb_uri.get_secret_value(),
        maxPoolSize=settings.mongodb_max_pool_size,
        minPoolSize=settings.mongodb_min_pool_size,
        maxIdleTimeMS=settings.mongodb_max_idle_time_ms,
        serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
        connectTimeoutMS=settings.mongodb_connect_timeout_ms,
        retryWrites=True,
        retryReads=True,
        appName="hybridrag-pytest",
    )


@pytest.fixture
async def mongodb_test_db(mongodb_test_settings: Settings):
    """Yield a clean MongoDB database and drop it after the test completes."""
    client = _create_test_client(mongodb_test_settings)
    database_name = mongodb_test_settings.mongodb_database

    await client.drop_database(database_name)
    db = client[database_name]

    yield db

    await client.drop_database(database_name)
    await client.close()

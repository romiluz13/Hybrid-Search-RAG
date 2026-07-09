"""
Unified MongoDB Client Factory.

Centralizes MongoDB client creation with proper connection pool settings.
Uses pymongo.AsyncMongoClient (replacing deprecated Motor AsyncIOMotorClient).

[Rule: consistency-read-concern-levels] - Configurable read/write concerns
[Rule: fundamental-commit-write-concern] - Explicit write concern
[Rule: mongodb-connection] - Create client once and reuse across application
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from pymongo import AsyncMongoClient
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern


def _tls_kwargs(uri: str, *, tls_flag: bool = False) -> dict[str, Any]:
    """Build TLS kwargs for MongoClient, fixing macOS + python.org CA issues.

    On macOS with python.org Python, the system CA trust store is not available
    to the ssl module, so Atlas (mongodb+srv://) connections fail with
    ``SSL: CERTIFICATE_VERIFY_FAILED``. We pass ``tlsCAFile`` from certifi for
    TLS connections. Override with the ``MONGODB_TLS_CA_FILE`` env var (e.g. for
    corporate custom CAs). On Linux/Windows the system store usually works, but
    certifi is a safe superset so we apply it for TLS connections everywhere.
    """
    del tls_flag  # kept for API stability; URI inspection decides TLS
    # Connection-string options are case-insensitive per the URI spec.
    uri_lower = uri.lower()
    uses_tls = (
        uri_lower.startswith("mongodb+srv://")
        or "tls=true" in uri_lower
        or "ssl=true" in uri_lower
    )
    if not uses_tls:
        return {}
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi is a declared core dep
        return {}
    ca = os.environ.get("MONGODB_TLS_CA_FILE") or certifi.where()
    return {"tlsCAFile": ca}


if TYPE_CHECKING:
    from pymongo.asynchronous.database import AsyncDatabase

    from ..config.settings import Settings

logger = logging.getLogger("hybridrag.mongodb_client")

# ── Singleton shared client ────────────────────────────────────────────────
# [Rule: mongodb-connection] Create client once only and reuse across application.
# Don't manually close connections unless shutting down.

_shared_client: AsyncMongoClient | None = None


def get_shared_client(settings: Settings) -> AsyncMongoClient:
    """Get or create the shared singleton MongoDB async client.

    Creates the client on first call, returns the same instance on subsequent
    calls. The client is configured with pool settings, retry, and appName
    per mongodb-connection skill best practices.

    Args:
        settings: HybridRAG Settings instance with MongoDB configuration.

    Returns:
        Shared AsyncMongoClient instance.
    """
    global _shared_client
    if _shared_client is None:
        uri = settings.mongodb_uri.get_secret_value()
        _shared_client = AsyncMongoClient(
            uri,
            maxPoolSize=settings.mongodb_max_pool_size,
            minPoolSize=settings.mongodb_min_pool_size,
            maxIdleTimeMS=settings.mongodb_max_idle_time_ms,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongodb_connect_timeout_ms,
            retryWrites=True,
            retryReads=True,
            appName="hybridrag",
            **_tls_kwargs(uri, tls_flag=settings.mongodb_tls),
        )
        logger.info(
            "[CLIENT] Shared AsyncMongoClient created "
            f"(pool={settings.mongodb_min_pool_size}-{settings.mongodb_max_pool_size}, "
            f"serverSelectionTimeout={settings.mongodb_server_selection_timeout_ms}ms, "
            f"connectTimeout={settings.mongodb_connect_timeout_ms}ms, "
            f"appName=hybridrag)"
        )
    return _shared_client


def close_shared_client() -> None:
    """Close and reset the shared singleton client.

    Call this only during application shutdown. Do not call from
    individual methods or request handlers.
    """
    global _shared_client
    if _shared_client is not None:
        _shared_client.close()
        logger.info("[CLIENT] Shared AsyncMongoClient closed")
        _shared_client = None


def _reset_shared_client() -> None:
    """Reset the shared client without closing (for tests only).

    This is used in unit tests to reset state between tests without
    actually closing a real connection.
    """
    global _shared_client
    _shared_client = None


# ── Legacy factory functions (kept for backward compatibility) ─────────────


def create_motor_client(
    uri: str,
    *,
    max_pool_size: int = 100,
    min_pool_size: int = 0,
    max_idle_time_ms: int = 60000,
    **kwargs: Any,
) -> AsyncMongoClient:
    """Create an async MongoDB client with proper pool settings.

    Note: Prefer get_shared_client() for production code.
    This factory is kept for backward compatibility and testing.

    Args:
        uri: MongoDB connection URI
        max_pool_size: Maximum connection pool size
        min_pool_size: Minimum connection pool size
        max_idle_time_ms: Max idle time for connections in ms
        **kwargs: Additional kwargs passed to AsyncMongoClient

    Returns:
        Configured AsyncMongoClient
    """
    return AsyncMongoClient(
        uri,
        maxPoolSize=max_pool_size,
        minPoolSize=min_pool_size,
        maxIdleTimeMS=max_idle_time_ms,
        **_tls_kwargs(uri),
        **kwargs,
    )


def create_motor_client_from_settings(settings: Settings) -> AsyncMongoClient:
    """Create an async MongoDB client from HybridRAG Settings.

    Note: Prefer get_shared_client() for production code.
    This factory is kept for backward compatibility.

    Args:
        settings: HybridRAG Settings instance

    Returns:
        Configured AsyncMongoClient
    """
    return create_motor_client(
        uri=settings.mongodb_uri.get_secret_value(),
        max_pool_size=settings.mongodb_max_pool_size,
        min_pool_size=settings.mongodb_min_pool_size,
        max_idle_time_ms=settings.mongodb_max_idle_time_ms,
    )


def get_database(
    client: AsyncMongoClient,
    database_name: str,
    *,
    read_concern: str = "local",
    write_concern: str = "1",
) -> AsyncDatabase:
    """Get a database handle with proper read/write concerns.

    Args:
        client: MongoDB async client
        database_name: Name of the database
        read_concern: Read concern level ('local', 'majority', 'snapshot')
        write_concern: Write concern level ('0', '1', 'majority')

    Returns:
        Database handle with configured concerns
    """
    # [M6] Validate write_concern parameter explicitly
    valid_concerns = {"0", "1", "majority"}
    if write_concern not in valid_concerns:
        raise ValueError(
            f"Invalid write_concern: {write_concern!r}. Must be one of {valid_concerns}"
        )

    db = client[database_name]

    # Apply read concern
    rc = ReadConcern(level=read_concern)
    db = db.with_options(read_concern=rc)

    # Apply write concern
    # [M7] Add wtimeout=10000 when w="majority" to prevent unbounded waits
    if write_concern == "majority":
        wc = WriteConcern(w="majority", wtimeout=10000)
    elif write_concern == "0":
        wc = WriteConcern(w=0)
    else:
        wc = WriteConcern(w=1)
    db = db.with_options(write_concern=wc)

    return db


def get_database_from_settings(
    client: AsyncMongoClient,
    settings: Settings,
) -> AsyncDatabase:
    """Get a database handle from HybridRAG Settings.

    Args:
        client: MongoDB async client
        settings: HybridRAG Settings instance

    Returns:
        Database handle with configured concerns from settings
    """
    return get_database(
        client,
        settings.mongodb_database,
        read_concern=settings.mongodb_read_concern,
        write_concern=settings.mongodb_write_concern,
    )

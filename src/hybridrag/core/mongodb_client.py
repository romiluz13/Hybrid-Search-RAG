"""
Unified MongoDB Client Factory.

Centralizes MongoDB client creation with proper connection pool settings.
Replaces scattered ad-hoc motor.AsyncIOMotorClient instantiations.

[Rule: consistency-read-concern-levels] - Configurable read/write concerns
[Rule: fundamental-commit-write-concern] - Explicit write concern
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

    from ..config.settings import Settings

logger = logging.getLogger("hybridrag.mongodb_client")


def create_motor_client(
    uri: str,
    *,
    max_pool_size: int = 100,
    min_pool_size: int = 0,
    max_idle_time_ms: int = 60000,
    **kwargs: Any,
) -> AsyncIOMotorClient:
    """Create a Motor async client with proper pool settings.

    Args:
        uri: MongoDB connection URI
        max_pool_size: Maximum connection pool size
        min_pool_size: Minimum connection pool size
        max_idle_time_ms: Max idle time for connections in ms
        **kwargs: Additional kwargs passed to AsyncIOMotorClient

    Returns:
        Configured AsyncIOMotorClient
    """
    return AsyncIOMotorClient(
        uri,
        maxPoolSize=max_pool_size,
        minPoolSize=min_pool_size,
        maxIdleTimeMS=max_idle_time_ms,
        **kwargs,
    )


def create_motor_client_from_settings(settings: Settings) -> AsyncIOMotorClient:
    """Create a Motor async client from HybridRAG Settings.

    Uses pool, concern, and timeout settings from the Settings object.

    Args:
        settings: HybridRAG Settings instance

    Returns:
        Configured AsyncIOMotorClient
    """
    return create_motor_client(
        uri=settings.mongodb_uri.get_secret_value(),
        max_pool_size=settings.mongodb_max_pool_size,
        min_pool_size=settings.mongodb_min_pool_size,
        max_idle_time_ms=settings.mongodb_max_idle_time_ms,
    )


def get_database(
    client: AsyncIOMotorClient,
    database_name: str,
    *,
    read_concern: str = "local",
    write_concern: str = "1",
) -> AsyncIOMotorDatabase:
    """Get a database handle with proper read/write concerns.

    Args:
        client: MongoDB async client
        database_name: Name of the database
        read_concern: Read concern level ('local', 'majority', 'snapshot')
        write_concern: Write concern level ('0', '1', 'majority')

    Returns:
        Database handle with configured concerns
    """
    db = client[database_name]

    # Apply read concern
    rc = ReadConcern(level=read_concern)
    db = db.with_options(read_concern=rc)

    # Apply write concern
    if write_concern == "majority":
        wc = WriteConcern(w="majority")
    elif write_concern == "0":
        wc = WriteConcern(w=0)
    else:
        wc = WriteConcern(w=1)
    db = db.with_options(write_concern=wc)

    return db


def get_database_from_settings(
    client: AsyncIOMotorClient,
    settings: Settings,
) -> AsyncIOMotorDatabase:
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

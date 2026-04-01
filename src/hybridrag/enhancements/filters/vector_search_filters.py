"""
Vector Search Filter Builder for MongoDB 8.0+.

CRITICAL: Vector search filters use STANDARD MongoDB operators.
This is DIFFERENT from Atlas Search filters which use Atlas-specific operators.

Reference: ai-agents-meetup/src/lib/search/vector-search.ts

Example $vectorSearch with filters:
{
    "$vectorSearch": {
        "index": "vector_knn_index",
        "path": "embedding",
        "queryVector": [...],
        "numCandidates": 200,
        "filter": {
            "timestamp": {"$gte": start_date, "$lte": end_date},
            "senderName": {"$eq": "John"},
            "category": {"$in": ["tech", "science"]}
        },
        "limit": 10
    }
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class VectorSearchFilterConfig:
    """Configuration for vector search filters.

    All filters use STANDARD MongoDB operators:
    - Date ranges: $gte, $lte
    - Equality: $eq
    - In-list: $in
    - Comparison: $gt, $lt, $ne
    """

    # Date range filters
    start_date: datetime | None = None
    end_date: datetime | None = None
    timestamp_field: str = "timestamp"

    # Equality filters: {field_name: value}
    equality_filters: dict[str, Any] = field(default_factory=dict)

    # In-list filters: {field_name: [values]}
    in_filters: dict[str, list[Any]] = field(default_factory=dict)

    # Comparison filters: {field_name: {"$gt": value}} etc.
    comparison_filters: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Negation filters: {field_name: value} -> {field_name: {"$ne": value}}
    not_equal_filters: dict[str, Any] = field(default_factory=dict)


def build_vector_search_filters(config: VectorSearchFilterConfig) -> dict[str, Any]:
    """
    Build filter object for $vectorSearch using STANDARD MongoDB operators.

    IMPORTANT: This is for $vectorSearch prefiltering (MongoDB 8.0+).
    Do NOT use Atlas Search operators (range, equals) here!

    Args:
        config: Filter configuration

    Returns:
        Filter dict using standard MongoDB operators

    Example output:
        {
            "timestamp": {"$gte": datetime(2024,1,1), "$lte": datetime(2024,12,31)},
            "senderName": {"$eq": "John"},
            "category": {"$in": ["tech", "science"]}
        }
    """
    filters: dict[str, Any] = {}

    # [L12] Validate field names to prevent operator injection
    all_field_names = (
        list(config.equality_filters.keys())
        + list(config.in_filters.keys())
        + list(config.comparison_filters.keys())
        + list(config.not_equal_filters.keys())
    )
    for field_name in all_field_names:
        if "$" in field_name or field_name.startswith("."):
            raise ValueError(
                f"Invalid filter field name: '{field_name}'. "
                "Field names must not contain '$' or start with '.'."
            )

    # Date range filters
    if config.start_date or config.end_date:
        date_filter: dict[str, datetime] = {}
        if config.start_date:
            date_filter["$gte"] = config.start_date
        if config.end_date:
            date_filter["$lte"] = config.end_date
        if date_filter:
            filters[config.timestamp_field] = date_filter

    # Equality filters using $eq
    for field_name, value in config.equality_filters.items():
        filters[field_name] = {"$eq": value}

    # In-list filters using $in (merge with existing equality_filters if collision)
    for field_name, values in config.in_filters.items():
        if field_name in filters:
            # Merge: if existing is $eq, include that value in $in list
            existing = filters[field_name]
            if "$eq" in existing:
                values = list(values) + [existing["$eq"]]
        filters[field_name] = {"$in": values}

    # Direct comparison filters (already in MongoDB format)
    for field_name, comparison in config.comparison_filters.items():
        if field_name in filters:
            # Merge with existing filter
            filters[field_name].update(comparison)
        else:
            filters[field_name] = comparison

    # Not-equal filters using $ne
    for field_name, value in config.not_equal_filters.items():
        if field_name in filters:
            filters[field_name]["$ne"] = value
        else:
            filters[field_name] = {"$ne": value}

    return filters


def build_vector_search_stage(
    index_name: str,
    query_vector: list[float],
    limit: int = 10,
    num_candidates: int = 100,
    path: str = "vector",
    filter_config: VectorSearchFilterConfig | None = None,
) -> dict[str, Any]:
    """
    Build complete $vectorSearch aggregation stage.

    Args:
        index_name: Name of the vector search index
        query_vector: Query embedding vector
        limit: Number of results to return
        num_candidates: Number of candidates to consider (should be >= limit * 10)
        path: Path to the vector field in documents
        filter_config: Optional filter configuration

    Returns:
        Complete $vectorSearch stage dict
    """
    stage: dict[str, Any] = {
        "$vectorSearch": {
            "index": index_name,
            "path": path,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }

    # Add filters if provided
    if filter_config:
        filters = build_vector_search_filters(filter_config)
        if filters:
            stage["$vectorSearch"]["filter"] = filters

    return stage

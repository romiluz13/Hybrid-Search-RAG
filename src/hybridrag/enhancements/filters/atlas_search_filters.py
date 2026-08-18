"""
Atlas Search Filter Builder for MongoDB Atlas Search.

CRITICAL: Atlas Search filters use ATLAS-SPECIFIC operators.
This is DIFFERENT from vector search filters which use standard MongoDB operators.

Reference: ai-agents-meetup/src/lib/search/atlas-search-filters.ts

Example $search with filters in compound query:
{
    "$search": {
        "index": "text_search_index",
        "compound": {
            "must": [{
                "text": {
                    "query": "search terms",
                    "path": ["content", "topics"]
                }
            }],
            "filter": [
                {"range": {"path": "timestamp", "gte": start_date, "lte": end_date}},
                {"equals": {"path": "senderName", "value": "John"}}
            ]
        }
    }
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AtlasSearchFilterConfig:
    """Configuration for Atlas Search filters.

    All filters use ATLAS SEARCH operators:
    - Date ranges: range (with gte, lte, gt, lt)
    - Equality: equals (with path, value)
    - In-list: compound.should with multiple equals
    - Exists: exists (with path)
    """

    # Date range filters
    start_date: datetime | None = None
    end_date: datetime | None = None
    timestamp_field: str = "timestamp"

    # Equality filters: {field_name: value}
    equality_filters: dict[str, Any] = field(default_factory=dict)

    # In-list filters: {field_name: [values]}
    in_filters: dict[str, list[Any]] = field(default_factory=dict)

    # Numeric range filters: {field_name: {"gte": min, "lte": max}}
    numeric_range_filters: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Exists filters: [field_names] - fields that must exist
    exists_filters: list[str] = field(default_factory=list)


def build_atlas_search_filters(config: AtlasSearchFilterConfig) -> list[dict[str, Any]]:
    """
    Build filter array for $search compound query using ATLAS SEARCH operators.

    IMPORTANT: This is for Atlas Search $search compound.filter clause.
    Do NOT use standard MongoDB operators ($gte, $eq) here!

    Args:
        config: Filter configuration

    Returns:
        List of filter clauses using Atlas Search operators

    Example output:
        [
            {"range": {"path": "timestamp", "gte": datetime(2024,1,1), "lte": datetime(2024,12,31)}},
            {"equals": {"path": "senderName", "value": "John"}},
            {"compound": {"should": [
                {"equals": {"path": "category", "value": "tech"}},
                {"equals": {"path": "category", "value": "science"}}
            ], "minimumShouldMatch": 1}}
        ]
    """
    filters: list[dict[str, Any]] = []

    # Date range filter using 'range' operator
    if config.start_date or config.end_date:
        date_range_filter: dict[str, Any] = {"path": config.timestamp_field}
        if config.start_date:
            date_range_filter["gte"] = config.start_date
        if config.end_date:
            date_range_filter["lte"] = config.end_date
        filters.append({"range": date_range_filter})

    # Equality filters using 'equals' operator
    for field_name, value in config.equality_filters.items():
        filters.append({"equals": {"path": field_name, "value": value}})

    # In-list filters using compound.should with multiple equals
    for field_name, values in config.in_filters.items():
        if not values:
            raise ValueError(
                f"in_filter values for field '{field_name}' must not be empty"
            )
        if len(values) == 1:
            # Single value - use simple equals
            filters.append({"equals": {"path": field_name, "value": values[0]}})
        elif len(values) > 1:
            # Multiple values - use compound.should
            should_clauses = [
                {"equals": {"path": field_name, "value": v}} for v in values
            ]
            filters.append(
                {"compound": {"should": should_clauses, "minimumShouldMatch": 1}}
            )

    # Numeric range filters using 'range' operator
    for field_name, range_spec in config.numeric_range_filters.items():
        range_filter: dict[str, Any] = {"path": field_name}
        if "gte" in range_spec:
            range_filter["gte"] = range_spec["gte"]
        if "lte" in range_spec:
            range_filter["lte"] = range_spec["lte"]
        if "gt" in range_spec:
            range_filter["gt"] = range_spec["gt"]
        if "lt" in range_spec:
            range_filter["lt"] = range_spec["lt"]
        if len(range_filter) == 1:
            raise ValueError(
                f"range filter for field '{field_name}' requires at least one bound"
            )
        filters.append({"range": range_filter})

    # Exists filters
    for field_name in config.exists_filters:
        filters.append({"exists": {"path": field_name}})

    return filters


def build_compound_search_stage(
    index_name: str,
    query_text: str,
    search_paths: list[str],
    filter_config: AtlasSearchFilterConfig | None = None,
    fuzzy_max_edits: int = 2,
    fuzzy_prefix_length: int = 3,
    path_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Build complete $search compound aggregation stage.

    Args:
        index_name: Name of the Atlas Search index
        query_text: Text query to search for
        search_paths: Paths to search (e.g., ["content", "topics", "senderName"])
        filter_config: Optional filter configuration
        fuzzy_max_edits: Max character edits for fuzzy matching
        fuzzy_prefix_length: Characters that must match exactly at start
        path_weights: Optional weights for paths (e.g., {"content": 10, "topics": 5})

    Returns:
        Complete $search stage dict with compound query
    """
    # Build the text search clause
    text_clause: dict[str, Any] = {
        "text": {
            "query": query_text,
            "path": search_paths,
            "fuzzy": {
                "maxEdits": fuzzy_max_edits,
                "prefixLength": fuzzy_prefix_length,
            },
        }
    }

    # Add path weights if provided
    if path_weights:
        # Use correct Atlas Search score boost format: simple numeric value
        text_clause["text"]["score"] = {"boost": {"value": 2.0}}

    # Build compound query
    compound: dict[str, Any] = {"must": [text_clause]}

    # Add filters if provided
    if filter_config:
        filters = build_atlas_search_filters(filter_config)
        if filters:
            compound["filter"] = filters

    stage: dict[str, Any] = {"$search": {"index": index_name, "compound": compound}}

    return stage

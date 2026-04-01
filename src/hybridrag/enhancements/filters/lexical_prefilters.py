"""
Lexical Prefilter Builder for MongoDB 8.2+ $search.vectorSearch.

NEW in MongoDB 8.2 (Nov 2025): The $search.vectorSearch operator supports
Atlas Search operators as prefilters, enabling fuzzy, phrase, wildcard,
geoWithin, and queryString filtering BEFORE vector search.

This is DIFFERENT from:
1. $vectorSearch filters - Only support MQL operators ($eq, $gte, $in)
2. Atlas $search filters - Used with text search, not vector search

Reference: https://www.mongodb.com/docs/atlas/atlas-search/operators-collectors/vectorSearch/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class TextFilter(TypedDict, total=False):
    """Atlas Search text filter for lexical prefiltering."""

    path: str | list[str]
    query: str
    fuzzy: dict[str, Any]  # Optional: {"maxEdits": 2, "prefixLength": 3}
    score: dict[str, Any]  # Optional score modification


class FuzzyFilter(TypedDict, total=False):
    """Atlas Search fuzzy text filter."""

    path: str | list[str]
    query: str
    maxEdits: int  # 1 or 2, default 2
    prefixLength: int  # default 0
    maxExpansions: int  # default 50


class PhraseFilter(TypedDict, total=False):
    """Atlas Search phrase filter for exact phrase matching."""

    path: str | list[str]
    query: str
    slop: int  # Allowable distance between words, default 0


class WildcardFilter(TypedDict, total=False):
    """Atlas Search wildcard filter."""

    path: str | list[str]
    query: str  # Supports * and ? wildcards
    allowAnalyzedField: bool  # default False


class GeoFilter(TypedDict, total=False):
    """Atlas Search geo filter."""

    path: str
    relation: str  # "contains", "disjoint", "intersects", "within"
    geometry: dict[str, Any]  # GeoJSON geometry


class QueryStringFilter(TypedDict, total=False):
    """Atlas Search queryString filter for Lucene-style queries."""

    defaultPath: str
    query: str  # Lucene query syntax


@dataclass
class LexicalPrefilterConfig:
    """Configuration for lexical prefilters in $search.vectorSearch.

    These filters use Atlas Search operators and are applied BEFORE
    vector search, narrowing the candidate set for better performance
    and precision.

    NEW in MongoDB 8.2 (November 2025).
    """

    # Text filters (exact or analyzed text matching)
    text_filters: list[TextFilter] = field(default_factory=list)

    # Fuzzy text filters (typo-tolerant matching)
    fuzzy_filters: list[FuzzyFilter] = field(default_factory=list)

    # Phrase filters (exact phrase matching with optional slop)
    phrase_filters: list[PhraseFilter] = field(default_factory=list)

    # Wildcard filters (pattern matching with * and ?)
    wildcard_filters: list[WildcardFilter] = field(default_factory=list)

    # Range filters: {field_path: {"gte": min, "lte": max}}
    range_filters: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Geo filters (geospatial filtering)
    geo_filters: list[GeoFilter] = field(default_factory=list)

    # QueryString filter (Lucene-style query)
    query_string_filter: QueryStringFilter | None = None

    # Equality filters (simple value matching)
    equality_filters: dict[str, Any] = field(default_factory=dict)


def build_lexical_prefilters(config: LexicalPrefilterConfig) -> dict[str, Any]:
    """
    Build filter object for $search.vectorSearch using Atlas Search operators.

    IMPORTANT: This is for MongoDB 8.2+ $search.vectorSearch lexical prefilters.
    These use Atlas Search operators (text, fuzzy, phrase, wildcard, geo).

    Args:
        config: Lexical prefilter configuration

    Returns:
        Filter dict for $search.vectorSearch filter parameter

    Example output for $search.vectorSearch:
        {
            "compound": {
                "filter": [
                    {"text": {"path": "category", "query": "technology"}},
                    {"range": {"path": "timestamp", "gte": "2024-01-01"}}
                ]
            }
        }
    """
    filter_clauses: list[dict[str, Any]] = []

    # Text filters
    # Per PEP 589: TypedDict total=False makes keys optional, validate before access
    for text_filter in config.text_filters:
        path = text_filter.get("path")
        query = text_filter.get("query")
        if not path or not query:
            continue  # Skip invalid filter (missing required fields)
        clause: dict[str, Any] = {
            "text": {
                "path": path,
                "query": query,
            }
        }
        if "fuzzy" in text_filter:
            clause["text"]["fuzzy"] = text_filter["fuzzy"]
        if "score" in text_filter:
            clause["text"]["score"] = text_filter["score"]
        filter_clauses.append(clause)

    # Fuzzy filters
    for fuzzy_filter in config.fuzzy_filters:
        path = fuzzy_filter.get("path")
        query = fuzzy_filter.get("query")
        if not path or not query:
            continue  # Skip invalid filter (missing required fields)
        clause = {
            "text": {
                "path": path,
                "query": query,
                "fuzzy": {
                    "maxEdits": fuzzy_filter.get("maxEdits", 2),
                    "prefixLength": fuzzy_filter.get("prefixLength", 0),
                    "maxExpansions": fuzzy_filter.get("maxExpansions", 50),
                },
            }
        }
        filter_clauses.append(clause)

    # Phrase filters
    for phrase_filter in config.phrase_filters:
        path = phrase_filter.get("path")
        query = phrase_filter.get("query")
        if not path or not query:
            continue  # Skip invalid filter (missing required fields)
        clause = {
            "phrase": {
                "path": path,
                "query": query,
            }
        }
        if "slop" in phrase_filter:
            clause["phrase"]["slop"] = phrase_filter["slop"]
        filter_clauses.append(clause)

    # Wildcard filters
    for wildcard_filter in config.wildcard_filters:
        path = wildcard_filter.get("path")
        query = wildcard_filter.get("query")
        if not path or not query:
            continue  # Skip invalid filter (missing required fields)
        clause = {
            "wildcard": {
                "path": path,
                "query": query,
            }
        }
        if wildcard_filter.get("allowAnalyzedField"):
            clause["wildcard"]["allowAnalyzedField"] = True
        filter_clauses.append(clause)

    # Range filters
    for path, range_spec in config.range_filters.items():
        range_clause: dict[str, Any] = {"path": path}
        for op in ["gte", "gt", "lte", "lt"]:
            if op in range_spec:
                range_clause[op] = range_spec[op]
        filter_clauses.append({"range": range_clause})

    # Geo filters (L10: use relation field to select correct operator)
    _GEO_RELATION_OPERATORS = {
        "within": "geoWithin",
        "intersects": "geoIntersects",
        "contains": "geoWithin",  # Atlas Search maps "contains" to geoWithin
        "disjoint": "geoWithin",  # No native disjoint; closest operator
    }
    for geo_filter in config.geo_filters:
        path = geo_filter.get("path")
        geometry = geo_filter.get("geometry")
        if not path or not geometry:
            continue  # Skip invalid filter (missing required fields)
        relation = geo_filter.get("relation", "within")
        geo_operator = _GEO_RELATION_OPERATORS.get(relation, "geoWithin")
        filter_clauses.append(
            {
                geo_operator: {
                    "path": path,
                    "geometry": geometry,
                }
            }
        )

    # QueryString filter
    if config.query_string_filter:
        default_path = config.query_string_filter.get("defaultPath")
        query = config.query_string_filter.get("query")
        if default_path and query:  # Only add if both required fields present
            filter_clauses.append(
                {
                    "queryString": {
                        "defaultPath": default_path,
                        "query": query,
                    }
                }
            )

    # Equality filters (using equals operator)
    for path, value in config.equality_filters.items():
        filter_clauses.append(
            {
                "equals": {
                    "path": path,
                    "value": value,
                }
            }
        )

    # Return empty dict if no filters
    if not filter_clauses:
        return {}

    # Wrap in compound filter structure
    return {"compound": {"filter": filter_clauses}}


def build_search_vector_search_stage(
    index_name: str,
    query_vector: list[float],
    vector_path: str = "vector",
    limit: int = 10,
    num_candidates: int | None = None,
    exact: bool = False,
    filter_config: LexicalPrefilterConfig | None = None,
    score_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build complete $search stage with vectorSearch operator.

    NEW in MongoDB 8.2: Uses $search.vectorSearch instead of $vectorSearch.
    This enables Atlas Search operators as prefilters.

    Args:
        index_name: Name of the Atlas Search index with vector field
        query_vector: Query embedding vector
        vector_path: Path to the vector field in documents
        limit: Number of results to return
        num_candidates: ANN candidates (default: limit * 20)
        exact: Use exact search instead of ANN (slower but precise)
        filter_config: Lexical prefilter configuration
        score_options: Optional score modification options

    Returns:
        Complete $search stage dict with vectorSearch operator

    Example:
        {
            "$search": {
                "index": "vector_index",
                "vectorSearch": {
                    "queryVector": [...],
                    "path": "vector",
                    "numCandidates": 200,
                    "limit": 10,
                    "filter": {...}  # Atlas Search operators!
                }
            }
        }
    """
    # Build vectorSearch operator
    vector_search: dict[str, Any] = {
        "queryVector": query_vector,
        "path": vector_path,
        "limit": limit,
    }

    # numCandidates and exact are mutually exclusive (MongoDB docs):
    # numCandidates is required when exact is false or omitted.
    # When exact=True, numCandidates MUST NOT be present.
    if exact:
        vector_search["exact"] = True
    else:
        # Calculate numCandidates if not provided
        if num_candidates is None:
            num_candidates = limit * 20  # MongoDB best practice
        vector_search["numCandidates"] = num_candidates

    # Add lexical prefilters if provided
    if filter_config:
        filters = build_lexical_prefilters(filter_config)
        if filters:
            vector_search["filter"] = filters

    # Add score options if provided
    if score_options:
        vector_search["score"] = score_options

    # Build complete $search stage
    return {
        "$search": {
            "index": index_name,
            "vectorSearch": vector_search,
        }
    }

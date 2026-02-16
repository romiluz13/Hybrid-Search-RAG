"""
Graph Search via MongoDB $graphLookup.

Implements bidirectional graph traversal for knowledge graph enhanced search.
Uses MongoDB's $graphLookup aggregation stage for efficient multi-hop queries.

Reference Architecture:
    Query: "What projects does John work on?"

    Graph Traversal:
        John --WORKS_AT--> Acme Corp
        John --LEADS--> Project Alpha
        Project Alpha --USES--> MongoDB
        MongoDB --PART_OF--> Tech Stack

    $graphLookup finds:
        - Direct relationships (depth=0): Acme Corp, Project Alpha
        - Indirect relationships (depth=1): MongoDB, Tech Stack

Features:
    - Bidirectional traversal (entity as source OR target)
    - Configurable depth and node limits
    - Weight-based prioritization
    - Integration with hybrid search
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection
    from pymongo.asynchronous.database import AsyncDatabase

import os

logger = logging.getLogger("hybridrag.graph_search")

# [Rule: ops-transaction-runtime-limit] Default aggregation timeout in ms
_GRAPH_AGGREGATE_TIMEOUT_MS = int(os.getenv("MONGO_AGGREGATE_TIMEOUT_MS", "30000"))


@dataclass
class GraphTraversalConfig:
    """Configuration for graph traversal operations."""

    # Collection settings
    edges_collection: str = "kg_edges"
    chunks_collection: str = "text_chunks"
    documents_collection: str = "documents"

    # Field mappings for edge documents
    source_field: str = "source_node_id"
    target_field: str = "target_node_id"
    relationship_type_field: str = "relationship_type"
    weight_field: str = "weight"

    # Traversal settings
    max_depth: int = 2  # Maximum hops from starting entity
    max_nodes: int = 50  # Maximum related entities to return
    min_weight: float = 0.0  # Minimum edge weight to traverse

    # Workspace prefix (for multi-tenant collections)
    workspace: str = ""


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""

    source: str
    target: str
    relationship_type: str
    weight: float
    depth: int = 0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphTraversalResult:
    """Result of a graph traversal operation."""

    starting_entity: str
    related_entities: list[str]
    edges: list[GraphEdge]
    max_depth_reached: int
    total_edges_traversed: int


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for consistent matching.

    Converts to lowercase and removes extra whitespace.

    Args:
        name: Raw entity name

    Returns:
        Normalized entity name
    """
    return re.sub(r"\s+", " ", name.lower().strip())


def escape_regex(text: str) -> str:
    """Escape special regex characters in a string."""
    return re.escape(text)


def build_graph_lookup_pipeline(
    entity_name: str,
    config: GraphTraversalConfig,
) -> list[dict[str, Any]]:
    """
    Build bidirectional $graphLookup aggregation pipeline.

    Creates a pipeline that:
    1. Matches edges where the entity is source or target (case-insensitive)
    2. Traverses outbound edges (source -> target)
    3. Collects related entities with depth info

    Args:
        entity_name: Starting entity for traversal
        config: Graph traversal configuration

    Returns:
        MongoDB aggregation pipeline stages
    """
    # Use case-insensitive regex for matching (handles varied casing in data)
    escaped_entity = escape_regex(entity_name.strip())
    entity_regex = re.compile(f"^{escaped_entity}$", re.IGNORECASE)

    # Get collection name with workspace prefix
    edges_collection = (
        f"{config.workspace}_{config.edges_collection}"
        if config.workspace
        else config.edges_collection
    )

    pipeline: list[dict[str, Any]] = [
        # Stage 1: Find edges where this entity is source or target (case-insensitive)
        {
            "$match": {
                "$or": [
                    {config.source_field: {"$regex": entity_regex}},
                    {config.target_field: {"$regex": entity_regex}},
                ],
            },
        },
        # Stage 2: Outbound traversal from matched edges
        {
            "$graphLookup": {
                "from": edges_collection,
                "startWith": f"${config.target_field}",
                "connectFromField": config.target_field,
                "connectToField": config.source_field,
                "maxDepth": config.max_depth
                - 1,  # -1 because we already matched first hop
                "depthField": "depth",
                "as": "connected_edges",
            },
        },
        # Stage 3: Unwind connected edges (preserve original if empty)
        {"$unwind": {"path": "$connected_edges", "preserveNullAndEmptyArrays": True}},
        # Stage 4: Sort by depth and weight
        {
            "$sort": {
                "connected_edges.depth": 1,
                f"connected_edges.{config.weight_field}": -1,
            }
        },
        # Stage 5: Limit results
        {"$limit": config.max_nodes * 2},  # Over-fetch for deduplication
    ]

    return pipeline


async def graph_traversal(
    db: AsyncDatabase,
    entity_name: str,
    config: GraphTraversalConfig | None = None,
) -> GraphTraversalResult:
    """
    Traverse the knowledge graph to find related entities.

    Uses MongoDB's $graphLookup for efficient multi-hop traversal.

    Args:
        db: MongoDB database instance
        entity_name: Starting entity for traversal
        config: Graph traversal configuration

    Returns:
        GraphTraversalResult with related entities and edges

    Example:
        >>> result = await graph_traversal(db, "MongoDB")
        >>> print(result.related_entities)
        ['Atlas', 'Vector Search', 'Aggregation Pipeline']
    """
    if config is None:
        config = GraphTraversalConfig()

    normalized_entity = normalize_entity_name(entity_name)

    # Get collection name with workspace prefix
    edges_collection_name = (
        f"{config.workspace}_{config.edges_collection}"
        if config.workspace
        else config.edges_collection
    )

    collection: AsyncCollection = db[edges_collection_name]

    # Build the pipeline
    pipeline = build_graph_lookup_pipeline(entity_name, config)

    try:
        cursor = collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_GRAPH_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        # Collect unique entities and edges
        entity_set: set[str] = set()
        edges: list[GraphEdge] = []
        seen_edges: set[str] = set()
        max_depth = 0

        for doc in results:
            # Add direct connections
            source = doc.get(config.source_field, "")
            target = doc.get(config.target_field, "")

            if source and source != normalized_entity:
                entity_set.add(source)
            if target and target != normalized_entity:
                entity_set.add(target)

            # Add first-hop edge
            rel_type = doc.get(config.relationship_type_field, "RELATED")
            edge_key = f"{source}-{target}-{rel_type}"

            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    GraphEdge(
                        source=source,
                        target=target,
                        relationship_type=rel_type,
                        weight=doc.get(config.weight_field, 1.0),
                        depth=0,
                    )
                )

            # Add edges from graphLookup results
            connected = doc.get("connected_edges")
            if connected:
                conn_source = connected.get(config.source_field, "")
                conn_target = connected.get(config.target_field, "")
                conn_rel_type = connected.get(config.relationship_type_field, "RELATED")
                conn_depth = connected.get("depth", 0) + 1

                max_depth = max(max_depth, conn_depth)

                edge_key = f"{conn_source}-{conn_target}-{conn_rel_type}"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    entity_set.add(conn_source)
                    entity_set.add(conn_target)

                    edges.append(
                        GraphEdge(
                            source=conn_source,
                            target=conn_target,
                            relationship_type=conn_rel_type,
                            weight=connected.get(config.weight_field, 1.0),
                            depth=conn_depth,
                        )
                    )

        # Remove the starting entity from results
        entity_set.discard(normalized_entity)

        # Sort edges by depth ascending, weight descending
        edges.sort(key=lambda e: (e.depth, -e.weight))

        # Limit to max_nodes
        related_entities = list(entity_set)[: config.max_nodes]

        logger.info(
            f"[GRAPH_SEARCH] Traversed from '{entity_name}': "
            f"found {len(related_entities)} related entities, {len(edges)} edges, "
            f"max_depth={max_depth}"
        )

        return GraphTraversalResult(
            starting_entity=entity_name,
            related_entities=related_entities,
            edges=edges[: config.max_nodes * 2],
            max_depth_reached=max_depth,
            total_edges_traversed=len(edges),
        )

    except Exception as e:
        logger.error(f"[GRAPH_SEARCH] Graph traversal failed for '{entity_name}': {e}")
        return GraphTraversalResult(
            starting_entity=entity_name,
            related_entities=[],
            edges=[],
            max_depth_reached=0,
            total_edges_traversed=0,
        )


async def get_chunks_for_entities(
    db: AsyncDatabase,
    entity_names: list[str],
    limit: int = 20,
    config: GraphTraversalConfig | None = None,
    date_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Get text chunks mentioning any of the given entities.

    Useful for retrieving content related to graph-discovered entities.

    Args:
        db: MongoDB database instance
        entity_names: List of entity names to search for
        limit: Maximum chunks to return
        config: Graph traversal configuration
        date_filter: Optional date range filter

    Returns:
        List of chunk documents mentioning the entities

    Example:
        >>> chunks = await get_chunks_for_entities(
        ...     db, ["MongoDB", "Atlas", "Vector Search"], limit=10
        ... )
    """
    if not entity_names:
        return []

    if config is None:
        config = GraphTraversalConfig()

    # Get collection name with workspace prefix
    chunks_collection_name = (
        f"{config.workspace}_{config.chunks_collection}"
        if config.workspace
        else config.chunks_collection
    )

    collection: AsyncCollection = db[chunks_collection_name]

    # Normalize entity names for matching
    normalized_names = [normalize_entity_name(name) for name in entity_names]

    # Build match stage with case-insensitive regex
    match_stage: dict[str, Any] = {
        "entities.name": {
            "$in": [
                re.compile(f"^{escape_regex(name)}$", re.IGNORECASE)
                for name in normalized_names
            ],
        },
    }

    # Add date filter if provided
    if date_filter:
        match_stage["timestamp"] = date_filter

    pipeline: list[dict[str, Any]] = [
        {"$match": match_stage},
        # Calculate entity match score
        {
            "$addFields": {
                "matched_entities": {
                    "$size": {
                        "$filter": {
                            "input": {"$ifNull": ["$entities", []]},
                            "as": "entity",
                            "cond": {
                                "$in": [
                                    {"$toLower": "$$entity.name"},
                                    normalized_names,
                                ],
                            },
                        },
                    },
                },
            },
        },
        {"$sort": {"matched_entities": -1, "timestamp": -1}},
        {"$limit": limit},
        {"$project": {"vector": 0, "embedding": 0}},  # Exclude large vectors
    ]

    try:
        cursor = collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_GRAPH_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        logger.info(
            f"[GRAPH_SEARCH] Found {len(results)} chunks for "
            f"{len(entity_names)} entities"
        )

        return results

    except Exception as e:
        logger.error(f"[GRAPH_SEARCH] Entity chunk lookup failed: {e}")
        return []


async def expand_entities_via_graph(
    db: AsyncDatabase,
    query_entities: list[str],
    config: GraphTraversalConfig | None = None,
) -> tuple[list[str], list[GraphEdge]]:
    """
    Expand a list of entities using graph traversal.

    Traverses from each query entity and collects all related entities.

    Args:
        db: MongoDB database instance
        query_entities: Entities extracted from the query
        config: Graph traversal configuration

    Returns:
        Tuple of (expanded_entities, all_edges)

    Example:
        >>> entities, edges = await expand_entities_via_graph(
        ...     db, ["MongoDB", "machine learning"]
        ... )
        >>> print(f"Expanded to {len(entities)} entities via {len(edges)} edges")
    """
    if config is None:
        config = GraphTraversalConfig()

    all_entities: set[str] = set()
    all_edges: list[GraphEdge] = []
    seen_edges: set[str] = set()

    for entity in query_entities:
        result = await graph_traversal(db, entity, config)

        # Collect unique entities
        all_entities.update(result.related_entities)

        # Collect unique edges
        for edge in result.edges:
            edge_key = f"{edge.source}-{edge.target}-{edge.relationship_type}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                all_edges.append(edge)

    logger.info(
        f"[GRAPH_SEARCH] Expanded {len(query_entities)} entities to "
        f"{len(all_entities)} related entities via {len(all_edges)} edges"
    )

    return list(all_entities), all_edges

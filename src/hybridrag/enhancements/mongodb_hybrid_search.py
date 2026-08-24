"""
MongoDB Native Hybrid Search using $rankFusion.

This module implements proper rank fusion for combining vector similarity
search with full-text keyword search using MongoDB Atlas's native $rankFusion
aggregation stage.

Why it matters:
    Simple interleaving of results from different sources using round-robin
    provides no actual fusion. This is inadequate for production RAG systems
    that need proper relevance scoring.

    MongoDB's $rankFusion uses Reciprocal Rank Fusion (RRF):
    score = Σ (1 / (60 + rank_i)) for each input pipeline

    This provides mathematically sound fusion of multiple retrieval signals.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pymongo.errors
from pydantic import BaseModel, Field

from hybridrag.engine.exceptions import (
    RetrievalCapabilityError,
    RetrievalExecutionError,
    RetrievalValidationError,
    is_retrieval_capability_error,
)

if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection
    from pymongo.asynchronous.database import AsyncDatabase

    from hybridrag.enhancements.filters import (
        AtlasSearchFilterConfig,
        LexicalPrefilterConfig,
        VectorSearchFilterConfig,
    )

logger = logging.getLogger("hybridrag.mongodb_hybrid")
logger.setLevel(logging.INFO)

# Default RRF constant (MongoDB default is 60, but configurable)
DEFAULT_RRF_CONSTANT = 60

# numCandidates multiplier (per MongoDB best practices: 10-20x limit)
# Reference: coleam00 recommendations, ai-agents-meetup patterns
NUM_CANDIDATES_MULTIPLIER = 20

# [Rule: ops-transaction-runtime-limit] Default aggregation timeout in ms
_HYBRID_AGGREGATE_TIMEOUT_MS = int(os.getenv("MONGO_AGGREGATE_TIMEOUT_MS", "30000"))


def calculate_num_candidates(
    top_k: int, multiplier: int = NUM_CANDIDATES_MULTIPLIER
) -> int:
    """
    Calculate numCandidates dynamically based on requested limit.

    Per MongoDB best practices (coleam00, official docs):
    numCandidates should be 10-20x the limit for good recall.

    Args:
        top_k: Number of results requested
        multiplier: Multiplier for top_k (default: 20)

    Returns:
        numCandidates value for vector search
    """
    return min(top_k * multiplier, 10_000)


def extract_pipeline_score(
    score_details: dict[str, Any] | None, pipeline_name: str
) -> float:
    """
    Extract per-pipeline score from scoreDetails.

    Reference: JohnGUnderwood/atlas-hybrid-search, ai-agents-meetup

    Args:
        score_details: The scoreDetails object from $rankFusion
        pipeline_name: Name of the pipeline ("vector" or "text")

    Returns:
        The score value for that pipeline, or 0.0 if not found
    """
    if not score_details or "details" not in score_details:
        return 0.0

    details = score_details.get("details", [])
    for detail in details:
        if detail.get("inputPipelineName") == pipeline_name:
            return detail.get("value", 0.0)

    return 0.0


class SearchResult(BaseModel):
    """
    Type-safe model for search results.

    Provides consistent structure for all search types (semantic, text, hybrid).
    Uses Pydantic for validation and serialization.
    """

    chunk_id: str = Field(..., description="MongoDB ObjectId of chunk as string")
    document_id: str = Field(
        default="", description="Parent document ObjectId as string"
    )
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(
        ..., description="Relevance score (0-1 for vector, RRF score for hybrid)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata (source, page, etc.)"
    )
    document_title: str = Field(default="", description="Title from document lookup")
    document_source: str = Field(
        default="", description="Source path from document lookup"
    )
    search_type: str = Field(
        default="unknown", description="Type of search that produced this result"
    )
    # Per-pipeline scores from $rankFusion scoreDetails
    source_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-pipeline scores: {vector: float, text: float}",
    )
    # Raw scoreDetails from $rankFusion (for debugging)
    score_details: dict[str, Any] | None = Field(
        default=None, description="Raw scoreDetails from $rankFusion"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()


@dataclass
class MongoDBHybridSearchConfig:
    """Configuration for MongoDB hybrid search."""

    # Vector search settings
    vector_index_name: str = "vector_knn_index"
    vector_path: str = "vector"
    # Optional explicit numCandidates override; None = dynamic calculation (top_k * NUM_CANDIDATES_MULTIPLIER)
    vector_num_candidates: int | None = None

    # Full-text search settings
    text_index_name: str = "text_search_index"
    text_search_path: str | list[str] = "content"  # Can be single field or list

    # Multi-field search paths with weights
    # Keys are field paths, values are boost weights
    # Example: {"content": 10, "topics": 5, "senderName": 1}
    text_search_path_weights: dict[str, float] | None = None

    # Fuzzy search settings (for text search)
    fuzzy_max_edits: int = 2  # Max character edits for fuzzy matching
    fuzzy_prefix_length: int = 3  # Chars that must match exactly at start

    # Hybrid search settings
    vector_weight: float = 0.6  # Weight for vector search in score fusion
    text_weight: float = 0.4  # Weight for text search in score fusion
    use_rank_fusion: bool = False  # Use $rankFusion (RRF) instead of $scoreFusion

    # Per-branch over-fetch for $rankFusion / $scoreFusion input pipelines.
    # Inspired by the Anthropic CMA cookbook: each branch over-fetches so the
    # fusion stage has more candidates to rank, improving result quality for
    # small top_k values.  The branch limit is computed as:
    #   max(top_k * branch_overfetch_factor, branch_overfetch_floor)
    # Previous default was top_k * 2 (factor=2, floor=0).  The new defaults
    # (factor=4, floor=20) match the cookbook's proven heuristic.
    branch_overfetch_factor: int = 4
    branch_overfetch_floor: int = 20

    # Document lookup settings
    documents_collection: str = "documents"  # Collection for document metadata
    enable_document_lookup: bool = True  # Join with documents for metadata

    # Filtering
    cosine_threshold: float = 0.3

    # Lexical Prefilters (MongoDB 8.2+)
    use_lexical_prefilters: bool = (
        True  # Enable $search.vectorSearch with Atlas Search filters (recommended)
    )
    lexical_prefilter_index: str = (
        "default"  # Atlas Search index name for lexical prefilters
    )

    def __post_init__(self) -> None:
        if self.vector_num_candidates is not None and not (
            1 <= self.vector_num_candidates <= 10_000
        ):
            raise RetrievalValidationError(
                "vector_num_candidates must produce numCandidates between 1 and 10000"
            )
        if self.branch_overfetch_factor < 1:
            raise RetrievalValidationError("branch_overfetch_factor must be >= 1")
        if self.branch_overfetch_floor < 0:
            raise RetrievalValidationError("branch_overfetch_floor must be >= 0")

    def get_search_paths(self) -> list[str]:
        """Get list of search paths."""
        if self.text_search_path_weights:
            return list(self.text_search_path_weights.keys())
        if isinstance(self.text_search_path, list):
            return self.text_search_path
        return [self.text_search_path]

    def branch_limit(self, top_k: int) -> int:
        """Compute the per-branch over-fetch limit for fusion input pipelines.

        Uses ``max(top_k * branch_overfetch_factor, branch_overfetch_floor)``
        so small ``top_k`` values still feed the fusion stage enough candidates
        to produce high-quality ranked results.

        Args:
            top_k: The final number of results requested by the caller.

        Returns:
            The ``$limit`` value to use inside each fusion input branch.
        """
        return max(top_k * self.branch_overfetch_factor, self.branch_overfetch_floor)


async def create_text_search_index_if_not_exists(
    collection: AsyncCollection,
    index_name: str = "text_search_index",
    search_fields: list[str] | None = None,
) -> bool:
    """
    Create a MongoDB Atlas Search index for full-text search.

    Args:
        collection: MongoDB collection
        index_name: Name for the search index
        search_fields: Fields to index for text search (default: ["content"])

    Returns:
        bool: True if index was created, False if it already exists
    """
    if search_fields is None:
        search_fields = ["content"]

    try:
        # Check if index already exists
        indexes_cursor = await collection.list_search_indexes()
        indexes = await indexes_cursor.to_list(length=None)

        for index in indexes:
            if index.get("name") == index_name:
                logger.info(f"Text search index '{index_name}' already exists")
                return False

        # Create the search index definition
        # Using "lucene.standard" analyzer for general text search
        field_mappings = {}
        for field_name in search_fields:
            field_mappings[field_name] = {
                "type": "string",
                "analyzer": "lucene.standard",
            }

        from pymongo.operations import SearchIndexModel

        search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": field_mappings,
                }
            },
            name=index_name,
            type="search",
        )

        await collection.create_search_index(search_index_model)
        logger.info(f"Text search index '{index_name}' created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating text search index '{index_name}': {e}")
        raise


async def hybrid_search_with_rank_fusion(
    collection: AsyncCollection,
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    vector_filter_config: VectorSearchFilterConfig | None = None,
    atlas_filter_config: AtlasSearchFilterConfig | None = None,
    lexical_filter_config: LexicalPrefilterConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search using MongoDB's $rankFusion.

    This combines:
    1. Vector similarity search ($vectorSearch) with optional prefiltering
    2. Full-text keyword search ($search) with optional Atlas Search filters

    CRITICAL: Vector and Atlas use DIFFERENT filter syntaxes!
    - Vector: Standard MongoDB operators ($gte, $lte, $eq)
    - Atlas: Atlas Search operators (range, equals)

    Using Reciprocal Rank Fusion (RRF) formula:
    score = sum(1 / (60 + rank_i))

    Args:
        collection: MongoDB collection with both vector and text indexes
        query_text: The search query text
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Hybrid search configuration
        vector_filter_config: Filters for vector search (standard MongoDB operators)
        atlas_filter_config: Filters for Atlas Search (Atlas-specific operators)

    Returns:
        List of documents with fused relevance scores
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    if any(
        filter_config is not None
        for filter_config in (
            vector_filter_config,
            atlas_filter_config,
            lexical_filter_config,
        )
    ):
        raise RetrievalValidationError(
            "legacy fusion cannot prove independent filter models equivalent; "
            "use HybridRAG with the unified FilterConfig"
        )

    logger.info(
        f"[HYBRID_SEARCH] Starting $rankFusion search: "
        f"query='{query_text[:50]}...', top_k={top_k}, "
        f"vector_filtered={vector_filter_config is not None}, "
        f"text_filtered={atlas_filter_config is not None}"
    )

    # Calculate numCandidates from the branch limit (not top_k) so the
    # 10-20x ratio is maintained relative to the actual $vectorSearch limit.
    # MongoDB docs: "You can't specify a number less than the number of
    # documents to return (limit)." and "We recommend that you specify a
    # number at least 20 times higher than the number of documents to
    # return (limit)."
    # Ref: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/
    branch_lim = config.branch_limit(top_k)
    num_candidates = (
        config.vector_num_candidates
        if config.vector_num_candidates is not None
        else calculate_num_candidates(branch_lim)
    )

    # Build vector search stage
    # Choose between $search.vectorSearch (MongoDB 8.2+) or $vectorSearch (legacy)
    if config.use_lexical_prefilters and lexical_filter_config:
        # Use NEW $search.vectorSearch with lexical prefilters
        from hybridrag.enhancements.filters import build_lexical_prefilters

        lexical_filters = build_lexical_prefilters(lexical_filter_config)

        vector_pipeline_stage: dict[str, Any] = {
            "$search": {
                "index": config.lexical_prefilter_index or config.vector_index_name,
                "vectorSearch": {
                    "queryVector": query_vector,
                    "path": config.vector_path,
                    "numCandidates": num_candidates,
                    "limit": branch_lim,
                },
            }
        }

        if lexical_filters:
            vector_pipeline_stage["$search"]["vectorSearch"]["filter"] = lexical_filters

        vector_pipeline = [vector_pipeline_stage]
        logger.info(
            "[HYBRID_SEARCH] Using $search.vectorSearch with lexical prefilters"
        )
    else:
        # Use legacy $vectorSearch with MQL filters
        vector_search_stage: dict[str, Any] = {
            "index": config.vector_index_name,
            "path": config.vector_path,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": branch_lim,
        }

        # Add vector prefilters if provided
        if vector_filter_config:
            from hybridrag.enhancements.filters import build_vector_search_filters

            vector_filters = build_vector_search_filters(vector_filter_config)
            if vector_filters:
                vector_search_stage["filter"] = vector_filters

        vector_pipeline = [{"$vectorSearch": vector_search_stage}]

    # M16: Add explicit $limit to vector pipeline for $rankFusion
    # Per mongodb-search-and-ai skill: "$search does not auto-limit results --
    # always add $limit inside the input pipeline"
    vector_pipeline.append({"$limit": branch_lim})

    # Build text search stage with compound query
    text_clause: dict[str, Any] = {
        "text": {
            "query": query_text,
            "path": config.text_search_path,
            "fuzzy": {
                "maxEdits": config.fuzzy_max_edits,
                "prefixLength": config.fuzzy_prefix_length,
            },
        }
    }

    compound_query: dict[str, Any] = {"must": [text_clause]}

    # Add Atlas Search filters if provided
    if atlas_filter_config:
        from hybridrag.enhancements.filters import build_atlas_search_filters

        atlas_filters = build_atlas_search_filters(atlas_filter_config)
        if atlas_filters:
            compound_query["filter"] = atlas_filters

    # Build the hybrid search pipeline using $rankFusion
    # Reference: JohnGUnderwood/atlas-hybrid-search, ai-agents-meetup patterns
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": vector_pipeline,  # Use the built vector pipeline
                        "text": [
                            {
                                "$search": {
                                    "index": config.text_index_name,
                                    "compound": compound_query,
                                }
                            },
                            {"$limit": branch_lim},
                        ],
                    }
                },
                # Explicit weights (configurable) instead of default RRF
                # Reference: MongoDB $rankFusion docs (combination.weights)
                "combination": {
                    "weights": {
                        "vector": config.vector_weight,
                        "text": config.text_weight,
                    }
                },
                # CRITICAL: Always enable scoreDetails for per-pipeline debugging
                # Reference: mongodb/docs, JohnGUnderwood/atlas-hybrid-search
                "scoreDetails": True,
            }
        },
        # Extract the fused score and scoreDetails for per-pipeline analysis.
        # Per official $meta docs, the score metadata field for $rankFusion (and
        # $scoreFusion) is "score" (NOT "rankFusionScore" — that keyword is not
        # documented and errors on many builds, silently degrading to manual RRF).
        {
            "$addFields": {
                "hybrid_score": {"$meta": "score"},
                "score_details": {"$meta": "scoreDetails"},
            }
        },
        {"$limit": top_k},
        {"$project": {"vector": 0}},
    ]

    try:
        cursor = await collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_HYBRID_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        logger.info(f"[HYBRID_SEARCH] $rankFusion returned {len(results)} results")

        # Format results with per-pipeline scores
        # Reference: ai-agents-meetup/src/lib/search/index.ts
        formatted_results = []
        for doc in results:
            score_details = doc.get("score_details")
            formatted_results.append(
                {
                    **doc,
                    "id": doc.get("_id"),
                    "score": doc.get("hybrid_score"),
                    "search_type": (
                        "hybrid_rrf_filtered"
                        if (vector_filter_config or atlas_filter_config)
                        else "hybrid_rrf"
                    ),
                    # Per-pipeline scores for debugging/analysis
                    "source_scores": {
                        "vector": extract_pipeline_score(score_details, "vector"),
                        "text": extract_pipeline_score(score_details, "text"),
                    },
                    "score_details": score_details,
                }
            )

        if formatted_results:
            top_result = formatted_results[0]
            top_score = top_result.get("score", 0)
            source_scores = top_result.get("source_scores", {})
            logger.info(
                f"[HYBRID_SEARCH] Top result score: {top_score:.4f} "
                f"(vector: {source_scores.get('vector', 0):.4f}, "
                f"text: {source_scores.get('text', 0):.4f})"
            )

        return formatted_results

    except pymongo.errors.OperationFailure as e:
        if is_retrieval_capability_error(e):
            raise RetrievalCapabilityError("rank fusion is unavailable") from e
        raise RetrievalExecutionError("rank fusion failed") from e
    except pymongo.errors.PyMongoError as e:
        raise RetrievalExecutionError("rank fusion failed") from e


async def hybrid_search_with_score_fusion(
    collection: AsyncCollection,
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search using MongoDB's $scoreFusion with custom weights.

    This allows explicit weighting of vector vs text search scores:
    final_score = (vector_weight * vector_score) + (text_weight * text_score)

    Args:
        collection: MongoDB collection
        query_text: The search query text
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Hybrid search configuration

    Returns:
        List of documents with weighted fusion scores
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    # Calculate numCandidates from the branch limit (not top_k) so the
    # 10-20x ratio is maintained relative to the actual $vectorSearch limit.
    # Ref: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/
    branch_lim = config.branch_limit(top_k)
    num_candidates = (
        config.vector_num_candidates
        if config.vector_num_candidates is not None
        else calculate_num_candidates(branch_lim)
    )

    logger.info(
        f"[HYBRID_SEARCH] Starting $scoreFusion search: "
        f"weights=[vector:{config.vector_weight}, text:{config.text_weight}]"
    )

    # Build the score fusion pipeline
    # Reference: https://www.mongodb.com/docs/manual/reference/operator/aggregation/scoreFusion/
    pipeline = [
        {
            "$scoreFusion": {
                "input": {
                    "pipelines": {
                        # Pipeline 1: Vector search
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": config.vector_index_name,
                                    "path": config.vector_path,
                                    "queryVector": query_vector,
                                    "numCandidates": num_candidates,
                                    "limit": branch_lim,
                                }
                            }
                        ],
                        # Pipeline 2: Full-text search
                        "text": [
                            {
                                "$search": {
                                    "index": config.text_index_name,
                                    "text": {
                                        "query": query_text,
                                        "path": config.text_search_path,
                                    },
                                }
                            },
                            {"$limit": branch_lim},
                        ],
                    },
                    # Sigmoid normalization for score scaling (must be inside input)
                    "normalization": "sigmoid",
                },
                "combination": {
                    "method": "expression",
                    "expression": {
                        "$sum": [
                            {
                                "$multiply": [
                                    "$$vector",
                                    config.vector_weight,
                                ]
                            },
                            {
                                "$multiply": [
                                    "$$text",
                                    config.text_weight,
                                ]
                            },
                        ]
                    },
                },
                "scoreDetails": True,
            }
        },
        {"$addFields": {"fusion_score": {"$meta": "score"}}},
        {"$limit": top_k},
        {"$project": {"vector": 0}},
    ]

    try:
        cursor = await collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_HYBRID_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        logger.info(f"[HYBRID_SEARCH] $scoreFusion returned {len(results)} results")

        formatted_results = []
        for doc in results:
            formatted_results.append(
                {
                    **doc,
                    "id": doc.get("_id"),
                    "score": doc.get("fusion_score"),
                    "search_type": "hybrid_score_fusion",
                }
            )

        return formatted_results

    except pymongo.errors.OperationFailure as e:
        if is_retrieval_capability_error(e):
            raise RetrievalCapabilityError("score fusion is unavailable") from e
        raise RetrievalExecutionError("score fusion failed") from e
    except pymongo.errors.PyMongoError as e:
        raise RetrievalExecutionError("score fusion failed") from e


def build_weighted_text_search_clause(
    query_text: str,
    path_weights: dict[str, float],
    fuzzy_max_edits: int = 2,
    fuzzy_prefix_length: int = 3,
) -> list[dict[str, Any]]:
    """
    Build weighted text search clauses for multi-field search.

    Creates separate text clauses for each field with score boosting.

    Args:
        query_text: Search query text
        path_weights: Dict of {field_path: boost_weight}
        fuzzy_max_edits: Max edits for fuzzy matching
        fuzzy_prefix_length: Required prefix length for fuzzy

    Returns:
        List of text search clauses with score boosting

    Example:
        path_weights = {"content": 10, "topics": 5, "senderName": 1}
        Returns clauses where matches in "content" score 10x higher
    """
    clauses = []

    for path, weight in path_weights.items():
        clause = {
            "text": {
                "query": query_text,
                "path": path,
                "fuzzy": {
                    "maxEdits": fuzzy_max_edits,
                    "prefixLength": fuzzy_prefix_length,
                },
                "score": {"boost": {"value": weight}},
            }
        }
        clauses.append(clause)

    return clauses


async def multi_field_text_search(
    collection: AsyncCollection,
    query_text: str,
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: AsyncDatabase | None = None,
    filter_config: AtlasSearchFilterConfig | None = None,
) -> list[SearchResult]:
    """
    Perform multi-field weighted text search.

    Searches across multiple fields with different weights for each field.
    Higher weights mean matches in that field score higher.

    Args:
        collection: MongoDB collection
        query_text: Search query
        top_k: Number of results
        config: Search config with path_weights
        db: Database for lookups
        filter_config: Optional Atlas Search filters

    Returns:
        List of SearchResult ordered by weighted relevance
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    # Use path weights if provided, otherwise fall back to simple search
    if not config.text_search_path_weights:
        return await text_only_search(
            collection, query_text, top_k, config, db, filter_config
        )

    # Build weighted search clauses
    weighted_clauses = build_weighted_text_search_clause(
        query_text,
        config.text_search_path_weights,
        config.fuzzy_max_edits,
        config.fuzzy_prefix_length,
    )

    # Build compound query with "should" for weighted fields
    compound_query: dict[str, Any] = {
        "should": weighted_clauses,
        "minimumShouldMatch": 1,  # At least one field must match
    }

    # Add filters if provided
    if filter_config:
        from hybridrag.enhancements.filters import build_atlas_search_filters

        filters = build_atlas_search_filters(filter_config)
        if filters:
            compound_query["filter"] = filters

    pipeline: list[dict[str, Any]] = [
        {
            "$search": {
                "index": config.text_index_name,
                "compound": compound_query,
            }
        },
        {"$limit": top_k * 2},
    ]

    # Add lookup and projection
    if config.enable_document_lookup and db is not None:
        pipeline.extend(
            [
                {
                    "$lookup": {
                        "from": config.documents_collection,
                        "localField": "document_id",
                        "foreignField": "_id",
                        "as": "document_info",
                    }
                },
                {
                    "$unwind": {
                        "path": "$document_info",
                        "preserveNullAndEmptyArrays": True,
                    }
                },
            ]
        )

    pipeline.append(
        {
            "$project": {
                "chunk_id": "$_id",
                "document_id": 1,
                "content": 1,
                "similarity": {"$meta": "searchScore"},
                "metadata": 1,
                "document_title": {"$ifNull": ["$document_info.title", ""]},
                "document_source": {"$ifNull": ["$document_info.source", ""]},
            }
        }
    )

    try:
        cursor = await collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_HYBRID_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        search_results = [
            SearchResult(
                chunk_id=str(doc.get("chunk_id", "")),
                document_id=str(doc.get("document_id", "")),
                content=doc.get("content", ""),
                similarity=doc.get("similarity", 0.0),
                metadata=doc.get("metadata", {}),
                document_title=doc.get("document_title", ""),
                document_source=doc.get("document_source", ""),
                search_type="text_multi_field_weighted",
            )
            for doc in results
        ]

        logger.info(
            f"[MULTI_FIELD_SEARCH] Completed: query='{query_text[:50]}...', "
            f"fields={list(config.text_search_path_weights.keys())}, "
            f"results={len(search_results)}"
        )

        return search_results

    except pymongo.errors.PyMongoError as e:
        raise RetrievalExecutionError("text search failed") from e


async def text_only_search(
    collection: AsyncCollection,
    query_text: str,
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: AsyncDatabase | None = None,
    filter_config: AtlasSearchFilterConfig | None = None,
    search_paths: list[str] | None = None,
) -> list[SearchResult]:
    """
    Full-text search using MongoDB Atlas Search with compound queries.

    Uses $search operator with compound query for:
    - Multi-field weighted search
    - Fuzzy matching for typo tolerance
    - Prefiltering with Atlas Search operators

    Works on all Atlas tiers including M0 (free tier).

    Args:
        collection: MongoDB collection with text search index
        query_text: The search query text
        top_k: Number of results to return
        config: Search configuration
        db: Database instance for $lookup (optional)
        filter_config: Atlas Search filter configuration (optional)
        search_paths: Fields to search (default: ["content"])

    Returns:
        List of SearchResult objects ordered by text relevance
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    if search_paths is None:
        search_paths = (
            [config.text_search_path]
            if isinstance(config.text_search_path, str)
            else config.text_search_path
        )

    # Build the text clause with fuzzy matching
    text_clause: dict[str, Any] = {
        "text": {
            "query": query_text,
            "path": search_paths,
            "fuzzy": {
                "maxEdits": config.fuzzy_max_edits,
                "prefixLength": config.fuzzy_prefix_length,
            },
        }
    }

    # Build compound query
    compound_query: dict[str, Any] = {"must": [text_clause]}

    # Add filters if provided
    if filter_config:
        from hybridrag.enhancements.filters import build_atlas_search_filters

        filters = build_atlas_search_filters(filter_config)
        if filters:
            compound_query["filter"] = filters
            logger.debug(f"[TEXT_SEARCH] Applied {len(filters)} Atlas Search filters")

    # Build pipeline with compound query
    pipeline: list[dict[str, Any]] = [
        {
            "$search": {
                "index": config.text_index_name,
                "compound": compound_query,
            }
        },
        {"$limit": top_k * 2},  # Over-fetch for better results
    ]

    # Add $lookup for document metadata if enabled and db provided
    if config.enable_document_lookup and db is not None:
        pipeline.extend(
            [
                {
                    "$lookup": {
                        "from": config.documents_collection,
                        "localField": "document_id",
                        "foreignField": "_id",
                        "as": "document_info",
                    }
                },
                {
                    "$unwind": {
                        "path": "$document_info",
                        "preserveNullAndEmptyArrays": True,
                    }
                },
            ]
        )

    # Project final fields (inclusion-only: unlisted fields like 'vector' are
    # automatically excluded; mixing inclusion with exclusion causes server error)
    pipeline.append(
        {
            "$project": {
                "chunk_id": "$_id",
                "document_id": 1,
                "content": 1,
                "similarity": {"$meta": "searchScore"},
                "metadata": 1,
                "document_title": {"$ifNull": ["$document_info.title", ""]},
                "document_source": {"$ifNull": ["$document_info.source", ""]},
            }
        }
    )

    try:
        cursor = await collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_HYBRID_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                chunk_id=str(doc.get("chunk_id", doc.get("_id", ""))),
                document_id=str(doc.get("document_id", "")),
                content=doc.get("content", ""),
                similarity=doc.get("similarity", 0.0),
                metadata=doc.get("metadata", {}),
                document_title=doc.get("document_title", ""),
                document_source=doc.get("document_source", ""),
                search_type="text_compound" if filter_config else "text_only",
            )
            for doc in results
        ]

        logger.info(
            f"[TEXT_SEARCH] Completed: query='{query_text[:50]}...', "
            f"results={len(search_results)}, filtered={filter_config is not None}"
        )

        return search_results

    except pymongo.errors.PyMongoError as e:
        raise RetrievalExecutionError("text search failed") from e


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = DEFAULT_RRF_CONSTANT,
) -> list[SearchResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF is a simple yet effective algorithm for combining results from different
    search methods. It works by scoring each document based on its rank position
    in each result list.

    Args:
        result_lists: List of ranked SearchResult lists from different searches
        k: RRF constant (default: 60, standard in literature)

    Returns:
        Unified list of SearchResult sorted by combined RRF score

    Algorithm:
        For each document d appearing in result lists:
            RRF_score(d) = Σ(1 / (k + rank_i(d)))
        Where rank_i(d) is the position of document d in result list i.

    References:
        - Cormack et al. (2009): "Reciprocal Rank Fusion outperforms the best system"
        - Standard k=60 performs well across various datasets
    """
    # Build score dictionary by chunk_id
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    # Process each search result list
    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id

            # Calculate RRF contribution: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            # Accumulate score (automatic deduplication)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = result

    # Sort by combined RRF score (descending)
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final result list with updated similarity scores
    merged_results = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_map[chunk_id]
        # Create new SearchResult with updated similarity (RRF score)
        merged_result = SearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            content=result.content,
            similarity=rrf_score,  # Combined RRF score
            metadata=result.metadata,
            document_title=result.document_title,
            document_source=result.document_source,
            search_type="hybrid_rrf_manual",
        )
        merged_results.append(merged_result)

    logger.info(
        f"[RRF] Merged {len(result_lists)} result lists into "
        f"{len(merged_results)} unique results"
    )

    return merged_results


async def manual_hybrid_search_with_rrf(
    collection: AsyncCollection,
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: AsyncDatabase | None = None,
) -> list[SearchResult]:
    """
    Explicit manual RRF implementation.

    It runs semantic (vector) search and text search concurrently, then merges
    results using Reciprocal Rank Fusion (RRF).

    Args:
        collection: MongoDB collection with both vector and text indexes
        query_text: The search query text
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Hybrid search configuration
        db: Database instance for $lookup (optional)

    Returns:
        List of SearchResult with fused RRF scores

    Algorithm:
        1. Run semantic search (vector similarity)
        2. Run text search (keyword/fuzzy matching)
        3. Merge results using Reciprocal Rank Fusion
        4. Return top N results by combined score
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    logger.info(
        f"[MANUAL_HYBRID] Starting manual RRF search: "
        f"query='{query_text[:50]}...', top_k={top_k}"
    )

    # Over-fetch for better RRF fusion results
    # Uses the same branch_limit heuristic as $rankFusion / $scoreFusion
    fetch_count = config.branch_limit(top_k)

    # Run both searches concurrently for performance
    vector_results, text_results = await asyncio.gather(
        vector_only_search(collection, query_vector, fetch_count, config, db),
        text_only_search(collection, query_text, fetch_count, config, db),
    )

    # If both failed, return empty list
    if not vector_results and not text_results:
        logger.error("[MANUAL_HYBRID] Both searches failed, returning empty results")
        return []

    # If only one succeeded, return those results directly
    if not vector_results:
        logger.info(
            "[MANUAL_HYBRID] Only text search succeeded, returning text results"
        )
        return text_results[:top_k]

    if not text_results:
        logger.info(
            "[MANUAL_HYBRID] Only vector search succeeded, returning vector results"
        )
        return vector_results[:top_k]

    # Merge results using Reciprocal Rank Fusion
    merged = reciprocal_rank_fusion(
        [vector_results, text_results],
        k=DEFAULT_RRF_CONSTANT,
    )

    # Return top N results
    final_results = merged[:top_k]

    logger.info(
        f"[MANUAL_HYBRID] Completed: "
        f"semantic={len(vector_results)}, text={len(text_results)}, "
        f"merged={len(merged)}, returned={len(final_results)}"
    )

    return final_results


async def vector_only_search(
    collection: AsyncCollection,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: AsyncDatabase | None = None,
    filter_config: VectorSearchFilterConfig | None = None,
) -> list[SearchResult]:
    """
    Perform semantic vector search using MongoDB Atlas Vector Search.

    Supports prefiltering with standard MongoDB operators.

    Args:
        collection: MongoDB collection with vector search index
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Search configuration
        db: Database instance for $lookup (optional)
        filter_config: Vector search filter configuration (optional)

    Returns:
        List of SearchResult objects ordered by vector similarity
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    # Calculate dynamic numCandidates (MongoDB best practice: 10-20x limit)
    num_candidates = (
        config.vector_num_candidates
        if config.vector_num_candidates is not None
        else calculate_num_candidates(top_k)
    )

    # Build $vectorSearch stage
    vector_search_stage: dict[str, Any] = {
        "index": config.vector_index_name,
        "path": config.vector_path,
        "queryVector": query_vector,
        "numCandidates": num_candidates,
        "limit": top_k,
    }

    # Add prefilters if provided.
    if filter_config:
        from hybridrag.enhancements.filters import build_vector_search_filters

        filters = build_vector_search_filters(filter_config)
        if filters:
            vector_search_stage["filter"] = filters
            logger.debug(f"[VECTOR_SEARCH] Applied prefilters: {list(filters.keys())}")

    # Build pipeline
    pipeline: list[dict[str, Any]] = [
        {"$vectorSearch": vector_search_stage},
    ]

    # Add $lookup for document metadata if enabled and db provided
    if config.enable_document_lookup and db is not None:
        pipeline.extend(
            [
                {
                    "$lookup": {
                        "from": config.documents_collection,
                        "localField": "document_id",
                        "foreignField": "_id",
                        "as": "document_info",
                    }
                },
                {
                    "$unwind": {
                        "path": "$document_info",
                        "preserveNullAndEmptyArrays": True,
                    }
                },
            ]
        )

    # Project final fields (inclusion-only: unlisted fields like 'vector' are
    # automatically excluded; mixing inclusion with exclusion causes server error)
    pipeline.append(
        {
            "$project": {
                "chunk_id": "$_id",
                "document_id": 1,
                "content": 1,
                "similarity": {"$meta": "vectorSearchScore"},
                "metadata": 1,
                "document_title": {"$ifNull": ["$document_info.title", ""]},
                "document_source": {"$ifNull": ["$document_info.source", ""]},
            }
        }
    )

    # Filter by cosine threshold
    pipeline.append({"$match": {"similarity": {"$gte": config.cosine_threshold}}})

    try:
        cursor = await collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_HYBRID_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                chunk_id=str(doc.get("chunk_id", doc.get("_id", ""))),
                document_id=str(doc.get("document_id", "")),
                content=doc.get("content", ""),
                similarity=doc.get("similarity", 0.0),
                metadata=doc.get("metadata", {}),
                document_title=doc.get("document_title", ""),
                document_source=doc.get("document_source", ""),
                search_type="vector_prefiltered" if filter_config else "vector_only",
            )
            for doc in results
        ]

        logger.info(
            f"[VECTOR_SEARCH] Completed: results={len(search_results)}, "
            f"threshold={config.cosine_threshold}, filtered={filter_config is not None}"
        )

        return search_results

    except pymongo.errors.PyMongoError as e:
        raise RetrievalExecutionError("vector search failed") from e


async def vector_search_with_lexical_prefilters(
    collection: AsyncCollection,
    query_vector: list[float],
    top_k: int = 10,
    config: MongoDBHybridSearchConfig | None = None,
    db: AsyncDatabase | None = None,
    lexical_filter_config: LexicalPrefilterConfig | None = None,
) -> list[SearchResult]:
    """
    Perform vector search using $search.vectorSearch with lexical prefilters.

    This uses $search.vectorSearch instead of $vectorSearch, enabling Atlas
    Search operators (text, fuzzy, phrase, wildcard, geo)
    as prefilters, narrowing the candidate set BEFORE vector similarity.

    Benefits over $vectorSearch:
    - Complex text filtering (fuzzy, phrase, wildcard)
    - Geospatial filtering
    - QueryString (Lucene syntax) filtering
    - Better performance for narrow result sets

    Args:
        collection: MongoDB collection with Atlas Search index
        query_vector: The query embedding vector
        top_k: Number of results to return
        config: Search configuration
        db: Database instance for $lookup (optional)
        lexical_filter_config: Lexical prefilter configuration

    Returns:
        List of SearchResult objects ordered by vector similarity

    Raises:
        Falls back to vector_only_search if $search.vectorSearch unavailable
    """
    if config is None:
        config = MongoDBHybridSearchConfig()

    from hybridrag.enhancements.filters import (
        build_search_vector_search_stage,
    )

    # Calculate dynamic numCandidates
    num_candidates = (
        config.vector_num_candidates
        if config.vector_num_candidates is not None
        else calculate_num_candidates(top_k)
    )

    # Build $search.vectorSearch stage
    search_stage = build_search_vector_search_stage(
        index_name=config.lexical_prefilter_index or config.vector_index_name,
        query_vector=query_vector,
        vector_path=config.vector_path,
        limit=top_k,
        num_candidates=num_candidates,
        filter_config=lexical_filter_config,
    )

    # Build pipeline
    pipeline: list[dict[str, Any]] = [search_stage]

    # Add score extraction
    # IMPORTANT: $search.vectorSearch uses "searchScore" (NOT "vectorSearchScore")
    # Reference: https://www.mongodb.com/docs/atlas/atlas-search/operators-collectors/vectorSearch/
    pipeline.append(
        {
            "$addFields": {
                "similarity": {"$meta": "searchScore"},
            }
        }
    )

    # Add $lookup for document metadata if enabled
    if config.enable_document_lookup and db is not None:
        pipeline.extend(
            [
                {
                    "$lookup": {
                        "from": config.documents_collection,
                        "localField": "document_id",
                        "foreignField": "_id",
                        "as": "document_info",
                    }
                },
                {
                    "$unwind": {
                        "path": "$document_info",
                        "preserveNullAndEmptyArrays": True,
                    }
                },
            ]
        )

    # Project final fields (inclusion-only: unlisted fields like 'vector' are
    # automatically excluded; mixing inclusion with exclusion causes server error)
    pipeline.append(
        {
            "$project": {
                "chunk_id": "$_id",
                "document_id": 1,
                "content": 1,
                "similarity": 1,
                "metadata": 1,
                "document_title": {"$ifNull": ["$document_info.title", ""]},
                "document_source": {"$ifNull": ["$document_info.source", ""]},
            }
        }
    )

    # Filter by cosine threshold
    pipeline.append({"$match": {"similarity": {"$gte": config.cosine_threshold}}})

    try:
        cursor = await collection.aggregate(
            pipeline, allowDiskUse=True, maxTimeMS=_HYBRID_AGGREGATE_TIMEOUT_MS
        )
        results = await cursor.to_list(length=None)

        search_results = [
            SearchResult(
                chunk_id=str(doc.get("chunk_id", doc.get("_id", ""))),
                document_id=str(doc.get("document_id", "")),
                content=doc.get("content", ""),
                similarity=doc.get("similarity", 0.0),
                metadata=doc.get("metadata", {}),
                document_title=doc.get("document_title", ""),
                document_source=doc.get("document_source", ""),
                search_type=(
                    "vector_lexical_prefiltered"
                    if lexical_filter_config
                    else "vector_search_new"
                ),
            )
            for doc in results
        ]

        logger.info(
            f"[VECTOR_LEXICAL] Completed: results={len(search_results)}, "
            f"threshold={config.cosine_threshold}, "
            f"lexical_filtered={lexical_filter_config is not None}"
        )

        return search_results

    except pymongo.errors.PyMongoError as e:
        raise RetrievalExecutionError("vector search failed") from e


class MongoDBHybridSearcher:
    """
    High-level interface for MongoDB hybrid search operations.

    This class manages the setup and execution of hybrid searches
    across multiple collections (chunks, entities, relationships).
    """

    def __init__(
        self,
        db: AsyncDatabase,
        workspace: str = "",
        config: MongoDBHybridSearchConfig | None = None,
    ):
        self.db = db
        self.workspace = workspace
        self.config = config or MongoDBHybridSearchConfig()
        self._initialized_collections: set[str] = set()

    def _get_collection_name(self, namespace: str) -> str:
        """Get the full collection name with workspace prefix."""
        if self.workspace:
            return f"{self.workspace}_{namespace}"
        return namespace

    async def ensure_text_index(
        self, namespace: str, search_fields: list[str] | None = None
    ) -> None:
        """
        Ensure text search index exists for a collection.

        Args:
            namespace: Collection namespace (e.g., "text_chunks")
            search_fields: Fields to index (default: ["content"])
        """
        collection_name = self._get_collection_name(namespace)

        if collection_name in self._initialized_collections:
            return

        collection = self.db[collection_name]

        # Determine index name based on workspace
        if self.workspace:
            index_name = f"text_search_index_{collection_name}"
        else:
            index_name = f"text_search_index_{namespace}"

        await create_text_search_index_if_not_exists(
            collection,
            index_name=index_name,
            search_fields=search_fields or ["content"],
        )

        self._initialized_collections.add(collection_name)

    async def hybrid_search(
        self,
        namespace: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        use_rank_fusion: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search on a collection.

        Args:
            namespace: Collection namespace
            query_text: Search query text
            query_vector: Query embedding vector
            top_k: Number of results
            use_rank_fusion: Use RRF (True) or weighted score fusion (False)

        Returns:
            List of search results with fusion scores
        """
        collection_name = self._get_collection_name(namespace)
        collection = self.db[collection_name]

        # Update config with workspace-specific index names while
        # preserving all other settings from self.config (vector_path,
        # num_candidates, over-fetch, fuzzy, prefilter, etc.).
        config = replace(
            self.config,
            vector_index_name=(
                f"vector_knn_index_{collection_name}"
                if self.workspace
                else "vector_knn_index"
            ),
            text_index_name=(
                f"text_search_index_{collection_name}"
                if self.workspace
                else f"text_search_index_{namespace}"
            ),
        )

        if use_rank_fusion:
            return await hybrid_search_with_rank_fusion(
                collection, query_text, query_vector, top_k, config
            )
        else:
            return await hybrid_search_with_score_fusion(
                collection, query_text, query_vector, top_k, config
            )


# Factory function for easy integration
def create_hybrid_searcher(
    db: AsyncDatabase,
    workspace: str = "",
    vector_weight: float = 0.6,
    text_weight: float = 0.4,
    cosine_threshold: float = 0.3,
) -> MongoDBHybridSearcher:
    """
    Create a MongoDB hybrid searcher with custom configuration.

    Args:
        db: MongoDB database instance
        workspace: Workspace prefix for collections
        vector_weight: Weight for vector search (0.0 to 1.0)
        text_weight: Weight for text search (0.0 to 1.0)
        cosine_threshold: Minimum cosine similarity threshold

    Returns:
        Configured MongoDBHybridSearcher instance
    """
    config = MongoDBHybridSearchConfig(
        vector_weight=vector_weight,
        text_weight=text_weight,
        cosine_threshold=cosine_threshold,
    )

    return MongoDBHybridSearcher(db, workspace, config)

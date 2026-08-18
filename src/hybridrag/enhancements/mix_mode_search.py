"""
Mix Mode Search - Knowledge Graph Enhanced Hybrid Search.

Combines multiple retrieval strategies:
1. Native $rankFusion (vector + keyword RRF) - Primary hybrid search
2. Graph traversal (entity relationships via $graphLookup)
3. Entity-boosted reranking (Voyage + entity overlap)

This is the most comprehensive search mode, ideal for:
- Complex queries requiring multi-hop reasoning
- Entity-centric searches (people, organizations, concepts)
- Cases where keyword and semantic search complement each other

Reference: JohnGUnderwood/atlas-hybrid-search, coleam00/MongoDB-RAG-Agent
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hybridrag.engine.exceptions import RetrievalError, RetrievalExecutionError

from .graph_search import (
    GraphTraversalConfig,
    expand_entities_via_graph,
    get_chunks_for_entities,
)
from .mongodb_hybrid_search import (
    MongoDBHybridSearchConfig,
    extract_pipeline_score,
    hybrid_search_with_rank_fusion,
)

if TYPE_CHECKING:
    from pymongo.asynchronous.database import AsyncDatabase

    from .filters import LexicalPrefilterConfig

logger = logging.getLogger("hybridrag.mix_mode")


class MixModeSearchResult(BaseModel):
    """Extended search result with mix mode metadata."""

    chunk_id: str = Field(..., description="MongoDB ObjectId of chunk as string")
    document_id: str = Field(
        default="", description="Parent document ObjectId as string"
    )
    content: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Combined relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Search source breakdown
    search_type: str = Field(default="mix_mode", description="Type of search")
    source_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-source scores: {vector, text, entity}",
    )

    # Graph metadata
    graph_entities: list[str] = Field(
        default_factory=list,
        description="Related entities discovered via graph traversal",
    )
    entity_boost: float = Field(
        default=0.0,
        description="Boost applied from entity overlap",
    )

    # Document metadata
    document_title: str = Field(default="", description="Title from document lookup")
    document_source: str = Field(default="", description="Source path from document")


@dataclass
class MixModeConfig:
    """Configuration for mix mode search."""

    # Hybrid search config
    hybrid_config: MongoDBHybridSearchConfig = field(
        default_factory=MongoDBHybridSearchConfig
    )

    # Graph traversal config
    graph_config: GraphTraversalConfig = field(default_factory=GraphTraversalConfig)

    # Mix mode specific settings
    enable_graph_traversal: bool = True
    enable_entity_boosting: bool = True
    enable_reranking: bool = True

    # Entity boosting weight (how much entity overlap affects final score)
    entity_boost_weight: float = 0.2

    # Entity-only result weight (for results found only via graph)
    entity_only_weight: float = 0.5

    # Lexical prefiltering
    use_lexical_prefilters: bool = True
    default_lexical_prefilter: LexicalPrefilterConfig | None = None


async def mix_mode_search(
    db: AsyncDatabase,
    query: str,
    query_vector: list[float],
    top_k: int = 10,
    config: MixModeConfig | None = None,
    query_entities: list[str] | None = None,
    collection_name: str = "text_chunks",
    lexical_filter_config: LexicalPrefilterConfig | None = None,
) -> list[MixModeSearchResult]:
    """
    Knowledge graph enhanced hybrid search.

    Combines:
    1. Native $rankFusion (vector + keyword) - Primary retrieval
    2. Graph traversal for query entities - Expands search scope
    3. Entity-based result boosting - Improves relevance

    Args:
        db: MongoDB database instance
        query: Search query text
        query_vector: Query embedding vector
        top_k: Number of results to return
        config: Mix mode configuration
        query_entities: Entities extracted from the query (optional)
        collection_name: Name of the chunks collection

    Returns:
        List of MixModeSearchResult with combined scores and metadata

    Example:
        >>> results = await mix_mode_search(
        ...     db=db,
        ...     query="How does MongoDB handle vector search?",
        ...     query_vector=embedding,
        ...     query_entities=["MongoDB", "vector search"],
        ...     top_k=10,
        ... )
    """
    if config is None:
        config = MixModeConfig()

    if query_entities is None:
        query_entities = []

    logger.info(
        f"[MIX_MODE] Starting search: query='{query[:50]}...', "
        f"entities={len(query_entities)}, top_k={top_k}"
    )

    # Track all discovered entities from graph traversal
    all_graph_entities: set[str] = set()
    search_tasks: list[asyncio.Task] = []

    # Get the collection
    collection = db[collection_name]

    # ============================================
    # Stage 1: Native $rankFusion (vector + keyword)
    # ============================================
    rank_fusion_results: list[dict[str, Any]] = []

    async def run_hybrid_search() -> None:
        nonlocal rank_fusion_results
        effective_lexical_filter = (
            lexical_filter_config or config.default_lexical_prefilter
            if config.use_lexical_prefilters
            else None
        )

        rank_fusion_results = await hybrid_search_with_rank_fusion(
            collection=collection,
            query_text=query,
            query_vector=query_vector,
            top_k=top_k * 2,
            config=config.hybrid_config,
            lexical_filter_config=effective_lexical_filter,
        )
        logger.info(
            f"[MIX_MODE] $rankFusion returned {len(rank_fusion_results)} results"
        )

    search_tasks.append(asyncio.create_task(run_hybrid_search()))

    # ============================================
    # Stage 2: Graph traversal (parallel with hybrid)
    # ============================================
    entity_results: list[dict[str, Any]] = []

    if config.enable_graph_traversal and query_entities:

        async def run_graph_traversal() -> None:
            nonlocal entity_results
            try:
                # Expand entities via graph
                expanded_entities, edges = await expand_entities_via_graph(
                    db=db,
                    query_entities=query_entities,
                    config=config.graph_config,
                )

                # Track discovered entities
                all_graph_entities.update(expanded_entities)
                all_graph_entities.update(query_entities)

                logger.info(
                    f"[MIX_MODE] Graph expanded to {len(expanded_entities)} entities "
                    f"via {len(edges)} edges"
                )

                # Get chunks mentioning these entities
                if expanded_entities:
                    chunks = await get_chunks_for_entities(
                        db=db,
                        entity_names=expanded_entities,
                        limit=top_k * 2,
                        config=config.graph_config,
                    )

                    for chunk in chunks:
                        entity_results.append(
                            {
                                "chunk_id": str(chunk.get("_id", "")),
                                "document_id": str(chunk.get("document_id", "")),
                                "content": chunk.get("content", ""),
                                "score": 0.5,  # Base score for entity matches
                                "metadata": chunk.get("metadata", {}),
                                "search_type": "entity",
                            }
                        )

                    logger.info(
                        f"[MIX_MODE] Entity search added {len(entity_results)} results"
                    )

            except RetrievalError:
                raise
            except Exception as e:
                raise RetrievalExecutionError("graph traversal failed") from e

        search_tasks.append(asyncio.create_task(run_graph_traversal()))

    # Wait for all searches to complete
    await asyncio.gather(*search_tasks)

    # ============================================
    # Stage 3: Merge results
    # ============================================
    # Deduplicate: prefer $rankFusion scores over entity scores
    merged_map: dict[str, dict[str, Any]] = {}

    # Add $rankFusion results first (higher priority)
    for result in rank_fusion_results:
        chunk_id = str(result.get("chunk_id", result.get("_id", "")))
        if not chunk_id:
            continue

        score_details = result.get("score_details")
        merged_map[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": str(result.get("document_id", "")),
            "content": result.get("content", ""),
            "score": result.get("score", result.get("hybrid_score", 0.0)),
            "metadata": result.get("metadata", {}),
            "search_type": result.get("search_type", "hybrid_rrf"),
            "source_scores": {
                "vector": extract_pipeline_score(score_details, "vector"),
                "text": extract_pipeline_score(score_details, "text"),
                "entity": 0.0,
            },
            "document_title": result.get("document_title", ""),
            "document_source": result.get("document_source", ""),
        }

    # Add entity results only if not already present, or boost existing
    for result in entity_results:
        chunk_id = str(result.get("chunk_id", ""))
        if not chunk_id:
            continue

        if chunk_id not in merged_map:
            # New result from entity search only
            merged_map[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": result.get("document_id", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0) * config.entity_only_weight,
                "metadata": result.get("metadata", {}),
                "search_type": "entity_only",
                "source_scores": {
                    "vector": 0.0,
                    "text": 0.0,
                    "entity": result.get("score", 0.0),
                },
                "document_title": "",
                "document_source": "",
            }
        else:
            # Boost existing result that also has entity match
            existing = merged_map[chunk_id]
            entity_score = result.get("score", 0.0)
            existing["source_scores"]["entity"] = entity_score
            existing["score"] += entity_score * config.entity_boost_weight

    # Sort by combined score
    fused_results = list(merged_map.values())
    fused_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Handle graceful degradation
    if not fused_results:
        logger.error("[MIX_MODE] All searches failed, returning empty results")
        return []

    logger.info(
        f"[MIX_MODE] Merged {len(fused_results)} unique results "
        f"({len(rank_fusion_results)} hybrid, {len(entity_results)} entity)"
    )

    # ============================================
    # Stage 4: Convert to MixModeSearchResult
    # ============================================
    final_results: list[MixModeSearchResult] = []
    graph_entities_list = list(all_graph_entities)

    for result in fused_results[:top_k]:
        entity_score = result.get("source_scores", {}).get("entity", 0.0)

        final_results.append(
            MixModeSearchResult(
                chunk_id=result.get("chunk_id", ""),
                document_id=result.get("document_id", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                search_type=result.get("search_type", "mix_mode"),
                source_scores=result.get("source_scores", {}),
                graph_entities=graph_entities_list,
                entity_boost=entity_score * config.entity_boost_weight,
                document_title=result.get("document_title", ""),
                document_source=result.get("document_source", ""),
            )
        )

    logger.info(
        f"[MIX_MODE] Returning {len(final_results)} results, "
        f"discovered {len(all_graph_entities)} graph entities"
    )

    return final_results


async def create_mix_mode_searcher(
    db: AsyncDatabase,
    workspace: str = "",
    vector_weight: float = 0.6,
    text_weight: float = 0.4,
    entity_boost_weight: float = 0.2,
    enable_graph: bool = True,
) -> MixModeSearcher:
    """
    Factory function to create a configured mix mode searcher.

    Args:
        db: MongoDB database instance
        workspace: Workspace prefix for collections
        vector_weight: Weight for vector search in fusion
        text_weight: Weight for text search in fusion
        entity_boost_weight: Weight for entity overlap boosting
        enable_graph: Whether to enable graph traversal

    Returns:
        Configured MixModeSearcher instance
    """
    hybrid_config = MongoDBHybridSearchConfig(
        vector_weight=vector_weight,
        text_weight=text_weight,
    )

    graph_config = GraphTraversalConfig(workspace=workspace)

    config = MixModeConfig(
        hybrid_config=hybrid_config,
        graph_config=graph_config,
        entity_boost_weight=entity_boost_weight,
        enable_graph_traversal=enable_graph,
    )

    return MixModeSearcher(db=db, config=config)


class MixModeSearcher:
    """
    High-level interface for mix mode search operations.

    Encapsulates the configuration and database connection for easy reuse.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        config: MixModeConfig | None = None,
    ):
        self.db = db
        self.config = config or MixModeConfig()

    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        query_entities: list[str] | None = None,
        collection_name: str = "text_chunks",
        lexical_filter_config: LexicalPrefilterConfig | None = None,
    ) -> list[MixModeSearchResult]:
        """
        Execute mix mode search.

        Args:
            query: Search query text
            query_vector: Query embedding vector
            top_k: Number of results
            query_entities: Entities extracted from query
            collection_name: Name of chunks collection
            lexical_filter_config: Lexical prefilter configuration

        Returns:
            List of MixModeSearchResult
        """
        return await mix_mode_search(
            db=self.db,
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            config=self.config,
            query_entities=query_entities,
            collection_name=collection_name,
            lexical_filter_config=lexical_filter_config,
        )

    async def search_with_graph_only(
        self,
        query_entities: list[str],
        top_k: int = 10,
    ) -> list[MixModeSearchResult]:
        """
        Search using only graph traversal (no hybrid search).

        Useful for entity-centric exploration.

        Args:
            query_entities: Entities to expand via graph
            top_k: Number of results

        Returns:
            List of MixModeSearchResult from graph-discovered chunks
        """
        if not query_entities:
            return []

        # Expand entities via graph
        expanded_entities, edges = await expand_entities_via_graph(
            db=self.db,
            query_entities=query_entities,
            config=self.config.graph_config,
        )

        # Get chunks for expanded entities
        chunks = await get_chunks_for_entities(
            db=self.db,
            entity_names=expanded_entities + query_entities,
            limit=top_k,
            config=self.config.graph_config,
        )

        # Convert to MixModeSearchResult
        results: list[MixModeSearchResult] = []
        all_entities = list(set(expanded_entities + query_entities))

        for chunk in chunks:
            results.append(
                MixModeSearchResult(
                    chunk_id=str(chunk.get("_id", "")),
                    document_id=str(chunk.get("document_id", "")),
                    content=chunk.get("content", ""),
                    score=chunk.get("matched_entities", 1) / max(len(all_entities), 1),
                    metadata=chunk.get("metadata", {}),
                    search_type="graph_only",
                    source_scores={"vector": 0, "text": 0, "entity": 1.0},
                    graph_entities=expanded_entities,
                    entity_boost=0.0,
                )
            )

        return results[:top_k]

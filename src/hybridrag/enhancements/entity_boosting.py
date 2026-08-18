"""
Entity Boosting Enhancement.

Boosts relevance scores for chunks that contain entities relevant to the query,
adding a structural signal to complement semantic similarity.

Why it matters:
    Cross-encoder reranking optimizes for semantic similarity.
    Entity boosting adds structural context - chunks with more relevant
    entities are likely more informative for the query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("hybridrag.entity_boosting")
logger.setLevel(logging.INFO)


@dataclass
class EntityBoostingReranker:
    """
    Wrap a base reranker with entity overlap boosting.

    Two-stage reranking:
    1. Cross-encoder scoring (e.g., Voyage rerank-2.5)
    2. Entity overlap boosting based on relevant entities
    """

    base_rerank_func: Callable[..., list[dict[str, Any]]]
    boost_weight: float = 0.2

    def _find_entities_in_text(self, text: str, entity_names: set[str]) -> set[str]:
        """
        Find which entity names appear in the text (case-insensitive).

        Returns set of entity names found in the text.
        """
        text_lower = text.lower()
        found = set()
        for entity in entity_names:
            if entity.lower() in text_lower:
                found.add(entity)
        return found

    async def rerank_with_boost(
        self,
        query: str,
        chunks: Sequence[dict[str, Any]],
        relevant_entity_ids: set[str],
        top_n: int = 10,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks with entity boosting.

        Entity boosting uses TEXT-BASED matching: checks if entity names
        appear in the chunk text (case-insensitive).

        Args:
            query: Search query
            chunks: List of chunk dicts with 'content'
            relevant_entity_ids: Set of entity names relevant to the query
            top_n: Number of top results to return

        Returns:
            Reranked chunks with 'relevance_score', 'entity_boost', 'final_score'
        """
        logger.info(
            f"[ENTITY_BOOST] Starting boosted rerank: {len(chunks)} chunks, {len(relevant_entity_ids)} relevant entities, top_n={top_n}"
        )
        logger.debug(
            f"[ENTITY_BOOST] Relevant entities: {list(relevant_entity_ids)[:5]}..."
        )
        logger.debug(
            f"[ENTITY_BOOST] Query: '{query[:50]}...', boost_weight={self.boost_weight}"
        )

        if not chunks:
            logger.warning(
                "[ENTITY_BOOST] Empty chunks list provided, returning empty list"
            )
            return []

        # Extract texts for base reranking
        texts = [c.get("content") or c.get("text", "") for c in chunks]
        logger.debug(f"[ENTITY_BOOST] Extracted {len(texts)} texts for base reranking")

        # Stage 1: Base cross-encoder reranking
        # Request more than top_n to allow boosting to reorder
        logger.info(
            f"[ENTITY_BOOST] Stage 1: Calling base reranker for {min(top_n * 2, len(texts))} results"
        )
        reranked = await self.base_rerank_func(
            query, texts, top_n=min(top_n * 2, len(texts)), **kwargs
        )
        logger.info(
            f"[ENTITY_BOOST] Stage 1 complete: got {len(reranked)} reranked results"
        )

        # Map back to chunks with metadata
        result: list[dict[str, Any]] = []
        total_boost_applied = 0
        entities_found_in_any_chunk = set()

        for r in reranked:
            idx = r["index"]
            chunk = dict(chunks[idx])  # Copy
            chunk["relevance_score"] = r["relevance_score"]

            # Stage 2: Calculate entity boost using TEXT-BASED matching
            chunk_text = texts[idx]
            found_entities = self._find_entities_in_text(
                chunk_text, relevant_entity_ids
            )
            entities_found_in_any_chunk.update(found_entities)
            overlap = len(found_entities)

            if relevant_entity_ids:
                # Boost based on fraction of relevant entities found in this chunk
                boost = self.boost_weight * (overlap / len(relevant_entity_ids))
            else:
                boost = 0.0

            if boost > 0:
                total_boost_applied += 1
                logger.debug(
                    f"[ENTITY_BOOST] Chunk {idx}: found {overlap} entities ({found_entities}), boost={boost:.4f}"
                )

            chunk["entity_overlap"] = overlap
            chunk["entities_found"] = list(found_entities)
            chunk["entity_boost"] = boost
            chunk["final_score"] = min(1.0, chunk["relevance_score"] + boost)
            result.append(chunk)

        # Re-sort by final score
        result.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info(
            f"[ENTITY_BOOST] Stage 2 complete: {total_boost_applied}/{len(result)} chunks received boost"
        )
        logger.info(
            f"[ENTITY_BOOST] Entities found across all chunks: {entities_found_in_any_chunk}"
        )
        if result:
            logger.info(
                f"[ENTITY_BOOST] Top result: final_score={result[0]['final_score']:.4f} (relevance={result[0]['relevance_score']:.4f}, boost={result[0]['entity_boost']:.4f})"
            )

        return result[:top_n]

    async def __call__(
        self,
        query: str,
        documents: Sequence[str] | Sequence[dict[str, Any]],
        top_n: int = 10,
        relevant_entity_ids: set[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Callable interface for HybridRAG.

        Can accept either:
        - List of strings (basic reranking)
        - List of dicts with 'content' and 'entity_ids' (boosted reranking)
        """
        logger.info(
            f"[ENTITY_BOOST] __call__ invoked: {len(documents)} documents, top_n={top_n}, has_entity_ids={relevant_entity_ids is not None}"
        )

        # Convert strings to dicts if needed
        if documents and isinstance(documents[0], str):
            chunks = [{"content": d} for d in documents]
            logger.debug("[ENTITY_BOOST] Converted string documents to dict format")
        else:
            chunks = list(documents)  # type: ignore

        # If no entity IDs provided, use base reranking
        if not relevant_entity_ids:
            logger.info(
                "[ENTITY_BOOST] No entity IDs provided, falling back to base reranker"
            )
            texts = [c.get("content") or c.get("text", "") for c in chunks]
            try:
                result = await self.base_rerank_func(
                    query, texts, top_n=top_n, **kwargs
                )
                logger.info(
                    f"[ENTITY_BOOST] Base reranker returned {len(result)} results"
                )
                return result
            except Exception as e:
                logger.error(f"[ENTITY_BOOST] Base reranker error: {e}")
                raise

        return await self.rerank_with_boost(
            query,
            chunks,
            relevant_entity_ids,
            top_n,
            **kwargs,
        )


def create_boosted_rerank_func(
    base_rerank_func: Callable[..., list[dict[str, Any]]],
    boost_weight: float = 0.2,
) -> Callable[..., list[dict[str, Any]]]:
    """
    Create an entity-boosting reranker wrapping a base rerank function.

    Args:
        base_rerank_func: Base reranking function (e.g., Voyage rerank)
        boost_weight: Weight for entity overlap boost (0.0-1.0)

    Returns:
        Async rerank function with entity boosting
    """
    logger.info(
        f"[ENTITY_BOOST] Creating boosted reranker with boost_weight={boost_weight}"
    )
    reranker = EntityBoostingReranker(
        base_rerank_func=base_rerank_func,
        boost_weight=boost_weight,
    )

    async def boosted_rerank_func(
        query: str,
        documents: list[str] | list[dict[str, Any]],
        top_n: int = 10,
        relevant_entity_ids: set[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Rerank function with entity boosting."""
        return await reranker(query, documents, top_n, relevant_entity_ids, **kwargs)

    boosted_rerank_func.cache_identity = {  # type: ignore[attr-defined]
        "base": getattr(base_rerank_func, "cache_identity", None),
        "entity_boost_weight": boost_weight,
    }
    return boosted_rerank_func

"""
Implicit Expansion Enhancement.

Finds semantically related entities that don't have explicit graph edges,
improving multi-hop query performance.

Example:
    Query: "How does MongoDB handle concurrent writes?"

    Explicit Graph:
        MongoDB --USES--> Transactions
        MongoDB --HAS--> Write Concern

    Implicit (via similarity):
        MongoDB ~similar~ Write Locking
        MongoDB ~similar~ Optimistic Concurrency
        Transactions ~similar~ ACID Properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np


@dataclass
class ImplicitExpander:
    """
    Enhance entity retrieval with implicit (semantic) expansion.

    Finds entities similar to explicitly retrieved entities using vector
    similarity, discovering relationships not captured in the knowledge graph.
    """

    embedding_func: Callable[[list[str]], np.ndarray]
    similarity_threshold: float = 0.75
    max_expansions: int = 10
    _entity_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    async def expand_from_entities(
        self,
        explicit_entities: Sequence[dict[str, Any]],
        all_entity_embeddings: dict[str, tuple[str, list[float]]],
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find entities semantically similar to explicitly retrieved entities.

        Args:
            explicit_entities: Entities from graph traversal
            all_entity_embeddings: Dict mapping entity_id to (description, embedding)
            exclude_ids: Entity IDs to exclude from results

        Returns:
            List of implicitly related entities with similarity scores
        """
        import numpy as np

        exclude_ids = exclude_ids or set()
        expanded: list[dict[str, Any]] = []
        seen: set[str] = set(exclude_ids)

        # Add explicit entity IDs to seen
        for entity in explicit_entities:
            entity_id = entity.get("id") or entity.get("_id") or entity.get("entity_id")
            if entity_id:
                seen.add(str(entity_id))

        for entity in explicit_entities:
            entity_id = entity.get("id") or entity.get("_id") or entity.get("entity_id")
            if not entity_id:
                continue

            # Get entity description for similarity search
            description = entity.get("description") or entity.get("name", "")
            if not description:
                continue

            # Embed the entity description
            entity_embedding = await self.embedding_func([description])
            entity_vec = entity_embedding[0]

            # Find similar entities via cosine similarity
            similarities: list[tuple[str, float, dict]] = []

            for other_id, (
                other_desc,
                other_embedding,
            ) in all_entity_embeddings.items():
                if other_id in seen:
                    continue

                # Cosine similarity
                other_vec = np.array(other_embedding)
                similarity = float(
                    np.dot(entity_vec, other_vec)
                    / (np.linalg.norm(entity_vec) * np.linalg.norm(other_vec) + 1e-8)
                )

                if similarity >= self.similarity_threshold:
                    similarities.append(
                        (
                            other_id,
                            similarity,
                            {"id": other_id, "description": other_desc},
                        )
                    )

            # Sort by similarity and take top expansions
            similarities.sort(key=lambda x: x[1], reverse=True)

            for other_id, sim_score, other_entity in similarities[
                : self.max_expansions
            ]:
                if other_id not in seen:
                    expanded.append(
                        {
                            **other_entity,
                            "expansion_source": entity_id,
                            "expansion_type": "implicit",
                            "similarity_score": sim_score,
                        }
                    )
                    seen.add(other_id)

        return expanded

    async def expand_from_query(
        self,
        query: str,
        all_entity_embeddings: dict[str, tuple[str, list[float]]],
        explicit_entities: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Comprehensive implicit expansion from query.

        Args:
            query: Search query
            all_entity_embeddings: Dict mapping entity_id to (description, embedding)
            explicit_entities: Optional explicit entities for secondary expansion

        Returns:
            Dict with 'query_implicit' and 'entity_implicit' lists
        """
        import numpy as np

        result: dict[str, list[dict[str, Any]]] = {
            "query_implicit": [],
            "entity_implicit": [],
        }

        # Embed query
        query_embedding = await self.embedding_func([query])
        query_vec = query_embedding[0]

        # Find entities similar to query
        similarities: list[tuple[str, float, dict]] = []

        for entity_id, (description, embedding) in all_entity_embeddings.items():
            entity_vec = np.array(embedding)
            similarity = float(
                np.dot(query_vec, entity_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(entity_vec) + 1e-8)
            )

            if similarity >= self.similarity_threshold:
                similarities.append(
                    (
                        entity_id,
                        similarity,
                        {"id": entity_id, "description": description},
                    )
                )

        # Sort and take top
        similarities.sort(key=lambda x: x[1], reverse=True)

        seen: set[str] = set()
        for entity_id, sim_score, entity in similarities[: self.max_expansions]:
            result["query_implicit"].append(
                {
                    **entity,
                    "expansion_type": "query_implicit",
                    "similarity_score": sim_score,
                }
            )
            seen.add(entity_id)

        # Expand from explicit entities if provided
        if explicit_entities:
            entity_expanded = await self.expand_from_entities(
                explicit_entities,
                all_entity_embeddings,
                exclude_ids=seen,
            )
            result["entity_implicit"] = entity_expanded

        return result

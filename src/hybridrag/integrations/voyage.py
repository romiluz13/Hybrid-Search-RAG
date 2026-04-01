"""
Voyage AI integration for embeddings and reranking.

Provides:
- VoyageEmbedder: Embedding wrapper with voyage-context-3 support (+13% recall)
- VoyageReranker: Reranking wrapper with rerank-2.5
- Factory functions for HybridRAG integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("hybridrag.voyage")
logger.setLevel(logging.INFO)


@dataclass
class VoyageEmbedder:
    """
    Voyage AI embedding wrapper with support for contextualized embeddings.

    Features:
    - Standard embeddings with voyage-4-large
    - Contextualized embeddings with voyage-context-3 (+13% recall)
    - Async support for better performance
    - Batching for efficiency
    """

    api_key: str
    embedding_model: str = "voyage-4-large"
    context_model: str = "voyage-context-3"
    batch_size: int = 128

    def __post_init__(self) -> None:
        self._sync_client = voyageai.Client(api_key=self.api_key)
        self._async_client = voyageai.AsyncClient(api_key=self.api_key)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _embed_batch(
        self,
        batch: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """Embed a single batch with retry for rate limit errors."""
        result = await self._async_client.embed(
            batch,
            model=self.embedding_model,
            input_type=input_type,
        )
        return result.embeddings

    async def embed_async(
        self,
        texts: Sequence[str],
        input_type: str = "document",
    ) -> np.ndarray:
        """
        Embed texts asynchronously.

        Args:
            texts: List of texts to embed
            input_type: "document" for chunks, "query" for queries

        Returns:
            Numpy array of embeddings (n, 1024)
        """
        logger.info(
            f"[EMBEDDING] Starting embedding of {len(texts)} texts (type={input_type}, model={self.embedding_model})"
        )

        if not texts:
            logger.warning(
                "[EMBEDDING] Empty texts list provided, returning empty array"
            )
            return np.empty((0, 1024), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = list(texts[i : i + self.batch_size])
            logger.info(
                f"[EMBEDDING] Processing batch {batch_num}/{total_batches} ({len(batch)} texts)"
            )

            batch_embeddings = await self._embed_batch(batch, input_type)
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                f"[EMBEDDING] Batch {batch_num} complete, got {len(batch_embeddings)} embeddings"
            )

        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(
            f"[EMBEDDING] Complete: {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 'N/A'}"
        )
        return embeddings_array

    def embed_sync(
        self,
        texts: Sequence[str],
        input_type: str = "document",
    ) -> np.ndarray:
        """
        Embed texts synchronously.

        Args:
            texts: List of texts to embed
            input_type: "document" for chunks, "query" for queries

        Returns:
            Numpy array of embeddings (n, 1024)
        """
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            result = self._sync_client.embed(
                batch,
                model=self.embedding_model,
                input_type=input_type,
            )
            all_embeddings.extend(result.embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    async def embed_query_async(self, query: str) -> np.ndarray:
        """Embed a single query with query-optimized input type."""
        result = await self._async_client.embed(
            [query],
            model=self.embedding_model,
            input_type="query",
        )
        return np.array(result.embeddings[0], dtype=np.float32)

    async def embed_contextualized_async(
        self,
        chunks_by_document: Sequence[Sequence[str]],
    ) -> np.ndarray:
        """
        Embed chunks with document context for +13% recall improvement.

        Uses voyage-context-3 which considers the full document context
        when embedding each chunk.

        NOTE: This method is implemented but not yet wired into the main
        ingestion flow. The RAG engine's embedding happens at the storage layer
        without document grouping context. Wiring this requires refactoring
        to group chunks by document before embedding, which is a future enhancement.

        To use manually:
            embedder = VoyageEmbedder(api_key="...")
            embeddings = await embedder.embed_contextualized_async([
                ["doc1_chunk1", "doc1_chunk2"],  # Chunks from document 1
                ["doc2_chunk1"],                  # Chunks from document 2
            ])

        Args:
            chunks_by_document: List of lists, where each inner list contains
                               chunks from a single document.
                               Example: [[doc1_chunk1, doc1_chunk2], [doc2_chunk1]]

        Returns:
            Flattened numpy array of embeddings for all chunks
        """
        if not chunks_by_document or all(len(doc) == 0 for doc in chunks_by_document):
            return np.empty((0, 1024), dtype=np.float32)

        result = await self._async_client.contextualized_embed(
            list(chunks_by_document),
            model=self.context_model,
            input_type="document",
        )

        # Flatten embeddings from all documents
        all_embeddings: list[list[float]] = []
        for doc_result in result.results:
            all_embeddings.extend(doc_result.embeddings)

        return np.array(all_embeddings, dtype=np.float32)


@dataclass
class VoyageReranker:
    """
    Voyage AI reranker using rerank-2.5.

    Provides cross-encoder reranking for improved relevance scoring.
    """

    api_key: str
    model: str = "rerank-2.5"

    def __post_init__(self) -> None:
        self._sync_client = voyageai.Client(api_key=self.api_key)
        self._async_client = voyageai.AsyncClient(api_key=self.api_key)

    async def rerank_async(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int = 10,
        instructions: str | None = None,
    ) -> list[dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Number of top results to return
            instructions: Optional instructions to guide reranking (e.g.,
                        "Prioritize peer-reviewed sources"). When provided,
                        prepended to query as "{instructions}\nQuery: {query}"

        Returns:
            List of dicts with 'index', 'document', 'relevance_score'
        """
        logger.info(
            f"[RERANK] Starting reranking: query='{query[:50]}...', {len(documents)} documents, top_n={top_n}, model={self.model}"
        )

        if not documents:
            logger.warning(
                "[RERANK] Empty documents list provided, returning empty list"
            )
            return []

        # Format query with instructions if provided
        formatted_query = query
        if instructions:
            # Strip whitespace and validate
            instructions_clean = instructions.strip()
            if instructions_clean:
                formatted_query = f"{instructions_clean}\nQuery: {query}"
                logger.debug(
                    f"[RERANK] Instructions applied: '{instructions_clean[:50]}...'"
                )

        try:
            result = await self._async_client.rerank(
                query=formatted_query,
                documents=list(documents),
                model=self.model,
                top_k=min(top_n, len(documents)),  # Voyage API uses top_k
            )

            reranked = [
                {
                    "index": r.index,
                    "document": r.document,
                    "relevance_score": r.relevance_score,
                }
                for r in result.results
            ]

            logger.info(f"[RERANK] Complete: {len(reranked)} results returned")
            for i, r in enumerate(reranked[:3]):
                logger.debug(
                    f"[RERANK] Top {i + 1}: idx={r['index']}, score={r['relevance_score']:.4f}"
                )

            return reranked
        except Exception as e:
            logger.error(f"[RERANK] Error during reranking: {e}")
            raise

    def rerank_sync(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int = 10,
        instructions: str | None = None,
    ) -> list[dict]:
        """
        Synchronous reranking.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Number of top results to return
            instructions: Optional instructions to guide reranking (e.g.,
                        "Prioritize peer-reviewed sources"). When provided,
                        prepended to query as "{instructions}\nQuery: {query}"

        Returns:
            List of dicts with 'index', 'document', 'relevance_score'
        """
        if not documents:
            return []

        # Format query with instructions if provided
        formatted_query = query
        if instructions:
            # Strip whitespace and validate
            instructions_clean = instructions.strip()
            if instructions_clean:
                formatted_query = f"{instructions_clean}\nQuery: {query}"

        result = self._sync_client.rerank(
            query=formatted_query,
            documents=list(documents),
            model=self.model,
            top_k=min(top_n, len(documents)),  # Voyage API uses top_k
        )

        return [
            {
                "index": r.index,
                "document": r.document,
                "relevance_score": r.relevance_score,
            }
            for r in result.results
        ]


def create_embedding_func(
    api_key: str,
    model: str = "voyage-4-large",
    batch_size: int = 128,
) -> Callable[[list[str]], np.ndarray]:
    """
    Create embedding function for HybridRAG.

    Args:
        api_key: Voyage AI API key
        model: Embedding model name
        batch_size: Batch size for API calls

    Returns:
        Async function that takes list[str] and returns np.ndarray
    """
    embedder = VoyageEmbedder(
        api_key=api_key,
        embedding_model=model,
        batch_size=batch_size,
    )

    async def embed_func(texts: list[str]) -> np.ndarray:
        return await embedder.embed_async(texts, input_type="document")

    return embed_func


def create_rerank_func(
    api_key: str,
    model: str = "rerank-2.5",
    default_instructions: str | None = None,
) -> Callable[..., list[dict]]:
    """
    Create rerank function for HybridRAG.

    Args:
        api_key: Voyage AI API key
        model: Reranking model name
        default_instructions: Optional default instructions applied to all queries

    Returns:
        Async function for reranking documents
    """
    reranker = VoyageReranker(api_key=api_key, model=model)

    async def rerank_func(
        query: str,
        documents: list[str],
        top_n: int = 10,
        instructions: str | None = None,
        **kwargs,
    ) -> list[dict]:
        # Use per-query instructions if provided, otherwise use default
        final_instructions = (
            instructions if instructions is not None else default_instructions
        )
        logger.info(
            f"[RERANK_FUNC] Called with query='{query[:50]}...', {len(documents)} docs, top_n={top_n}, has_instructions={final_instructions is not None}"
        )
        result = await reranker.rerank_async(
            query, documents, top_n, instructions=final_instructions
        )
        logger.info(f"[RERANK_FUNC] Returning {len(result)} results")
        return result

    return rerank_func

"""
HybridRAG - State-of-the-art RAG with MongoDB + Multi-Provider Support.

This is the main RAG engine providing:
- Multiple LLM providers: Anthropic Claude, OpenAI GPT, Google Gemini
- Voyage AI embeddings and reranking (rerank-2.5)
- MongoDB Atlas storage (vector + graph + KV)
- Knowledge graph construction and querying
- Implicit expansion (semantic entity discovery)
- Entity boosting (structural relevance boost)
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..config.settings import Settings, get_settings

# Internal RAG engine imports (bundled in hybridrag.engine)
from ..engine import BaseRAGEngine as _RAGEngine
from ..engine import EmbeddingFunc
from ..engine import QueryParam as _QueryParam
from ..enhancements.entity_boosting import create_boosted_rerank_func
from ..ingestion import (
    ChunkingConfig,
    DocumentIngestionPipeline,
    IngestionConfig,
    IngestionResult,
)
from ..integrations import (
    langfuse_enabled,
    log_ingestion,
    log_rag_query,
)
from ..memory import ConversationMemory


# Re-export QueryParam with our own name
@dataclass
class QueryParam:
    """Query parameters for RAG operations."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    top_k: int = 60
    chunk_top_k: int = 10
    enable_rerank: bool = True
    only_need_context: bool = False
    stream: bool = False
    include_references: bool = False
    response_type: str = "Multiple Paragraphs"

    def _to_internal(self) -> _QueryParam:
        """Convert to internal format."""
        return _QueryParam(
            mode=self.mode,
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
            enable_rerank=self.enable_rerank,
            only_need_context=self.only_need_context,
            stream=self.stream,
            include_references=self.include_references,
            response_type=self.response_type,
        )


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger("hybridrag.core")


def _build_inline_file_paths(
    documents: Sequence[str], ids: Sequence[str] | None = None
) -> list[str]:
    """Create stable synthetic sources for inline text inserts.

    Raw string ingestion should still produce meaningful citations instead of
    collapsing to ``unknown_source`` inside the engine. When the caller does not
    provide file paths, we derive a deterministic inline source per document.
    """
    inline_paths: list[str] = []
    for index, document in enumerate(documents):
        stable_id = ids[index] if ids and index < len(ids) else None
        if not stable_id:
            stable_id = hashlib.md5(document.encode("utf-8")).hexdigest()[:12]
        inline_paths.append(f"inline://{stable_id}")
    return inline_paths


def _extract_references_from_query_data(
    query_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Return references from structured query data, synthesizing them when needed."""
    data = query_data.get("data", {})
    references = list(data.get("references", []))
    if references:
        return references

    synthesized: list[dict[str, str]] = []
    seen: set[str] = set()

    def _remember_reference(reference_id: Any, file_path: Any) -> None:
        ref_id = str(reference_id or "").strip()
        if not ref_id or ref_id in seen:
            return
        seen.add(ref_id)
        synthesized.append(
            {
                "reference_id": ref_id,
                "file_path": str(file_path or "unknown_source"),
            }
        )

    for chunk in data.get("chunks", []):
        _remember_reference(
            chunk.get("reference_id") or chunk.get("chunk_id"),
            chunk.get("file_path"),
        )

    if synthesized:
        return synthesized

    for entity in data.get("entities", []):
        _remember_reference(
            entity.get("reference_id")
            or entity.get("source_id")
            or entity.get("entity_name"),
            entity.get("file_path"),
        )

    for relationship in data.get("relationships", []):
        _remember_reference(
            relationship.get("reference_id")
            or relationship.get("source_id")
            or f"{relationship.get('src_id', '')}->{relationship.get('tgt_id', '')}",
            relationship.get("file_path"),
        )

    return synthesized


def _extract_context_from_query_data(query_data: dict[str, Any]) -> str:
    """Return a best-effort readable context string from structured query output."""
    explicit_context = query_data.get("context")
    if isinstance(explicit_context, str) and explicit_context.strip():
        return explicit_context

    llm_context = query_data.get("llm_response", {}).get("content")
    if isinstance(llm_context, str) and llm_context.strip():
        return llm_context

    chunks = query_data.get("data", {}).get("chunks", [])
    chunk_text = [
        str(chunk.get("content", "")).strip()
        for chunk in chunks
        if str(chunk.get("content", "")).strip()
    ]
    return "\n\n".join(chunk_text)


def _extract_retrieval_diagnostics(query_data: dict[str, Any]) -> dict[str, Any]:
    """Expose retrieval strategy details so fallback behavior is never silent."""
    diagnostics: dict[str, Any] = {}
    data = query_data.get("data", {})
    chunks = data.get("chunks", [])
    search_types = sorted(
        {str(chunk.get("search_type")) for chunk in chunks if chunk.get("search_type")}
    )
    if search_types:
        diagnostics["search_types"] = search_types

    fallback_search_types = [
        search_type
        for search_type in search_types
        if "manual" in search_type
        or "fallback" in search_type
        or search_type in {"vector_only", "text_only"}
    ]
    if fallback_search_types:
        diagnostics["fallback_search_types"] = fallback_search_types
        diagnostics["fallback_used"] = True
    elif search_types:
        diagnostics["fallback_used"] = False
    elif any(data.get(key) for key in ("chunks", "entities", "relationships")):
        # Some retrieval paths do not annotate chunk-level search_type yet, but they
        # still represent the primary blessed-stack path rather than a hidden fallback.
        diagnostics["fallback_used"] = False

    return diagnostics


def _create_embedding_func(
    settings: Settings,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Create Voyage AI embedding function - VOYAGE ONLY, no fallbacks."""
    from ..integrations.voyage import create_embedding_func

    if not settings.voyage_api_key:
        raise ValueError(
            "VOYAGE_API_KEY is REQUIRED - Voyage AI is the only supported embedding provider"
        )

    embed_func = create_embedding_func(
        api_key=settings.voyage_api_key.get_secret_value(),
        model=settings.voyage_embedding_model,
        batch_size=settings.embedding_batch_size,
        base_url=settings.voyage_base_url,
    )
    logger.info(
        f"[INIT] Voyage embedding configured: model={settings.voyage_embedding_model}, dim={settings.embedding_dim}, batch_size={settings.embedding_batch_size}"
    )
    return embed_func, settings.embedding_dim  # Use configured dimension


def _create_llm_func(settings: Settings) -> Callable[..., str]:
    """Create LLM function based on provider setting."""
    if not settings.enable_llm:
        from ..engine.prompt import PROMPTS

        completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        async def _llm_disabled(*args, **kwargs) -> str:
            return completion_delimiter

        logger.info("[INIT] LLM generation disabled by settings.enable_llm=False")
        return _llm_disabled

    provider = settings.llm_provider
    logger.info(f"[INIT] Creating LLM function for provider: {provider}")

    if provider == "anthropic":
        from ..integrations.anthropic import create_llm_func

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required when llm_provider=anthropic")
        logger.info(
            f"[INIT] Anthropic LLM configured: model={settings.anthropic_model}"
        )
        return create_llm_func(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.anthropic_model,
        )

    elif provider == "openai":
        from ..integrations.openai import create_openai_llm_func

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when llm_provider=openai")
        logger.info(f"[INIT] OpenAI LLM configured: model={settings.openai_model}")
        extra_headers = None
        if settings.openai_extra_headers:
            import json

            extra_headers = json.loads(settings.openai_extra_headers)
        return create_openai_llm_func(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            default_headers=extra_headers,
        )

    elif provider == "gemini":
        from ..integrations.gemini import create_gemini_llm_func

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required when llm_provider=gemini")
        logger.info(f"[INIT] Gemini LLM configured: model={settings.gemini_model}")
        return create_gemini_llm_func(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
        )

    elif provider == "grove":
        # Grove is a MongoDB-internal OpenAI-compatible gateway. Reuse the
        # OpenAI client with the Grove base_url + key.
        from ..integrations.openai import create_openai_llm_func

        grove_key_setting = settings.grove_api_key
        api_key: str = (
            grove_key_setting.get_secret_value()
            if grove_key_setting is not None
            else (os.getenv("GROVE_API_KEY") or "")
        )
        base_url = settings.grove_base_url or os.getenv("GROVE_BASE_URL")
        if not api_key or not base_url:
            raise ValueError(
                "GROVE_API_KEY and GROVE_BASE_URL required when llm_provider=grove "
                "(set them in .env or your shell)"
            )
        logger.info(f"[INIT] Grove LLM configured: model={settings.grove_model}")
        return create_openai_llm_func(
            api_key=api_key,
            model=settings.grove_model,
            base_url=base_url,
            default_headers=None,
        )

    else:
        logger.error(f"[INIT] Unknown LLM provider: {provider}")
        raise ValueError(f"Unknown LLM provider: {provider}")


@dataclass
class HybridRAG:
    """
    State-of-the-art RAG system with multi-provider support.

    Features:
    - MongoDB Atlas storage (vector + graph + KV)
    - Multiple LLM providers: Anthropic Claude, OpenAI GPT, Google Gemini
    - Voyage AI embeddings (voyage-4-large)
    - Voyage AI reranking (rerank-2.5)
    - Knowledge graph construction and querying
    - Implicit expansion (semantic entity discovery)
    - Entity boosting (structural relevance signal)

    Usage:
        ```python
        from hybridrag import HybridRAG

        rag = HybridRAG()
        await rag.initialize()

        # Ingest documents
        await rag.insert(["Document 1 content...", "Document 2 content..."])

        # Query
        response = await rag.query("What is MongoDB vector search?")
        ```
    """

    settings: Settings = field(default_factory=get_settings)
    working_dir: str = field(default="./hybridrag_workspace")
    _rag_engine: _RAGEngine | None = field(default=None, repr=False)
    _llm_func: Callable | None = field(default=None, repr=False)
    _memory: ConversationMemory | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def _get_pipeline_embed_func(self) -> Callable:
        """Get a reusable embedding function for ingestion pipelines.

        Caches the VoyageEmbedder instance to avoid re-creation per call.
        """
        if not hasattr(self, "_cached_embedder"):
            from ..integrations.voyage import VoyageEmbedder

            self._cached_embedder = VoyageEmbedder(
                api_key=self.settings.voyage_api_key.get_secret_value(),
                embedding_model=self.settings.voyage_embedding_model,
                batch_size=64,
                base_url=self.settings.voyage_base_url,
            )

        embedder = self._cached_embedder

        def pipeline_embed_func(texts: list[str]) -> list[list[float]]:
            """Wrapper to use HybridRAG's cached embedding function."""
            result = embedder.embed_sync(texts, input_type="document")
            return result.tolist() if hasattr(result, "tolist") else list(result)

        return pipeline_embed_func

    async def initialize(self) -> None:
        """
        Initialize HybridRAG with all components.

        Must be called before using insert() or query().
        """
        if self._initialized:
            logger.info("[INIT] HybridRAG already initialized, skipping")
            return

        logger.info("[INIT] ========== Starting HybridRAG Initialization ==========")

        # Pass URI directly to ClientManager via parameter (C1 fix: no env var leak)
        # ClientManager.get_client() now accepts uri + database_name parameters
        # instead of reading from os.environ.
        logger.info(
            f"[INIT] MongoDB configured: database={self.settings.mongodb_database}"
        )

        # Create embedding function based on provider
        logger.info("[INIT] Creating embedding function...")
        embed_func, embedding_dim = _create_embedding_func(self.settings)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=self.settings.max_token_size,
            func=embed_func,
        )
        logger.info(
            f"[INIT] EmbeddingFunc created: dim={embedding_dim}, max_tokens={self.settings.max_token_size}"
        )

        # Create rerank function (Voyage AI - best quality)
        rerank_func = None
        if self.settings.voyage_api_key:
            from ..integrations.rerank_instructions import get_rerank_instructions
            from ..integrations.voyage import create_rerank_func

            logger.info(
                f"[INIT] Creating Voyage reranker: model={self.settings.voyage_rerank_model}"
            )

            # Generate default instructions based on settings
            default_instructions = None
            if self.settings.voyage_rerank_instructions:
                # Custom global instructions take precedence
                default_instructions = self.settings.voyage_rerank_instructions
                logger.info(
                    f"[INIT] Using custom rerank instructions: '{default_instructions[:50]}...'"
                )
            elif self.settings.enable_smart_rerank_instructions:
                # Use intelligent defaults for default query mode
                default_instructions = get_rerank_instructions(
                    query_mode=self.settings.default_query_mode,
                    enable_smart_defaults=True,
                )
                if default_instructions:
                    logger.info(
                        f"[INIT] Using smart rerank instructions for mode '{self.settings.default_query_mode}': '{default_instructions[:50]}...'"
                    )

            base_rerank = create_rerank_func(
                api_key=self.settings.voyage_api_key.get_secret_value(),
                model=self.settings.voyage_rerank_model,
                default_instructions=default_instructions,
                base_url=self.settings.voyage_base_url,
            )

            # Wrap with entity boosting if enabled
            if self.settings.enable_entity_boosting:
                logger.info(
                    f"[INIT] Wrapping with entity boosting: weight={self.settings.entity_boost_weight}"
                )
                rerank_func = create_boosted_rerank_func(
                    base_rerank_func=base_rerank,
                    boost_weight=self.settings.entity_boost_weight,
                )
            else:
                rerank_func = base_rerank
                logger.info("[INIT] Entity boosting disabled, using base reranker")
        else:
            logger.warning("[INIT] No Voyage API key, reranking disabled")

        # Create LLM function based on provider
        logger.info("[INIT] Creating LLM function...")
        llm_func = _create_llm_func(self.settings)
        self._llm_func = llm_func if self.settings.enable_llm else None

        # Initialize RAG engine with MongoDB storage
        logger.info("[INIT] Initializing RAG engine with MongoDB storage...")
        # LightRAG engine storage classes read MONGO_URI/MONGO_DATABASE from env vars
        # during initialization (see engine/kg/mongo_impl.py:90, engine/kg/__init__.py:49).
        # Explicit settings ALWAYS win over stale env/shell state.
        os.environ["MONGO_URI"] = self.settings.mongodb_uri.get_secret_value()
        os.environ["MONGO_DATABASE"] = self.settings.mongodb_database
        self._rag_engine = _RAGEngine(
            working_dir=self.working_dir,
            # MongoDB storage backends
            kv_storage="MongoKVStorage",
            vector_storage="MongoVectorDBStorage",
            graph_storage="MongoGraphStorage",
            doc_status_storage="MongoDocStatusStorage",
            # Embedding integration
            embedding_func=embedding_func,
            rerank_model_func=rerank_func,
            # LLM
            llm_model_func=llm_func,
        )
        logger.info("[INIT] RAG engine instance created")

        # Initialize all storage backends (MongoDB collections)
        logger.info("[INIT] Initializing storage backends...")
        await self._rag_engine.initialize_storages()
        logger.info(
            "[INIT] Storage backends initialized (KV, Vector, Graph, DocStatus)"
        )

        # Initialize pipeline status
        from ..engine.kg.shared_storage import initialize_pipeline_status

        logger.info("[INIT] Initializing pipeline status...")
        await initialize_pipeline_status()

        # Initialize conversation memory for multi-turn conversations
        # Pass LLM function to enable self-compaction (summarization)
        logger.info("[INIT] Initializing conversation memory...")
        self._memory = ConversationMemory(
            mongodb_uri=self.settings.mongodb_uri.get_secret_value(),
            database=self.settings.mongodb_database,
            max_token_limit=32000,  # Compact when exceeds 32K tokens (models support 200K+)
            llm_func=llm_func if self.settings.enable_llm else None,
        )
        await self._memory.initialize()
        if self.settings.enable_llm:
            logger.info(
                "[INIT] Conversation memory initialized (with self-compaction enabled)"
            )
        else:
            logger.info(
                "[INIT] Conversation memory initialized (retrieval-only, compaction disabled)"
            )

        self._initialized = True
        logger.info("[INIT] ========== HybridRAG Initialization Complete ==========")

    def _ensure_initialized(self) -> _RAGEngine:
        """Ensure HybridRAG is initialized and return RAG engine instance."""
        if not self._initialized or self._rag_engine is None:
            raise RuntimeError(
                "HybridRAG not initialized. Call 'await rag.initialize()' first."
            )
        return self._rag_engine

    async def _expand_query_with_entities(
        self,
        query: str,
        rag: _RAGEngine,
        max_entities: int = 5,
    ) -> str:
        """
        Expand query with semantically related entities using vector search.

        This implements implicit expansion - finding entities similar to the query
        that may not have explicit graph connections but are semantically related.

        Args:
            query: Original search query
            rag: RAG engine instance with entities_vdb
            max_entities: Maximum number of entities to add

        Returns:
            Expanded query with related entity names
        """
        try:
            # Query the entity vector database for similar entities
            similar_entities = await rag.entities_vdb.query(query, top_k=max_entities)

            if not similar_entities:
                logger.debug("[IMPLICIT_EXPANSION] No similar entities found")
                return query

            # Extract entity names from results
            # Official LightRAG uses meta_fields={"entity_name", "source_id", "content", "file_path"}
            # MongoDB storage returns "id" mapped from "_id", plus all meta_fields
            entity_names = []
            for entity in similar_entities:
                # Primary: entity_name (official meta_field), fallback: id (from MongoDB _id)
                name = entity.get("entity_name") or entity.get("id")
                if name and name.lower() not in query.lower():
                    entity_names.append(name)

            if not entity_names:
                logger.debug("[IMPLICIT_EXPANSION] No new entities to add")
                return query

            # Augment query with related entity terms
            # Format: "original query (related: entity1, entity2, ...)"
            related_terms = ", ".join(entity_names[:max_entities])
            expanded_query = f"{query} (related concepts: {related_terms})"

            logger.info(
                f"[IMPLICIT_EXPANSION] Expanded query with {len(entity_names)} entities: {related_terms[:100]}..."
            )
            return expanded_query

        except Exception as e:
            logger.warning(f"[IMPLICIT_EXPANSION] Failed to expand query: {e}")
            return query  # Fall back to original query on error

    async def insert(
        self,
        documents: str | Sequence[str],
        ids: Sequence[str] | None = None,
        file_paths: Sequence[str] | None = None,
    ) -> None:
        """
        Insert documents into the RAG system.

        Args:
            documents: Single document or list of documents to ingest
            ids: Optional document IDs
            file_paths: Optional file paths for metadata
        """
        rag = self._ensure_initialized()

        # Normalize to list
        if isinstance(documents, str):
            documents = [documents]

        doc_count = len(documents)
        total_chars = sum(len(d) for d in documents)
        logger.info("[INSERT] ========== Starting Document Insertion ==========")
        logger.info(f"[INSERT] Documents: {doc_count}, Total chars: {total_chars:,}")
        resolved_file_paths = (
            list(file_paths) if file_paths else _build_inline_file_paths(documents, ids)
        )
        logger.info(f"[INSERT] File paths: {resolved_file_paths}")

        import time as _time

        start_time = _time.time()

        try:
            await rag.ainsert(
                input=list(documents),
                ids=list(ids) if ids else None,
                file_paths=resolved_file_paths,
            )
            duration = _time.time() - start_time
            logger.info("[INSERT] ========== Document Insertion Complete ==========")

            # Log to Langfuse if enabled
            if langfuse_enabled():
                log_ingestion(
                    file_name=resolved_file_paths[0]
                    if resolved_file_paths
                    else "unknown",
                    num_chunks=doc_count,
                    num_entities=0,  # Not tracked at this level
                    num_relations=0,  # Not tracked at this level
                    duration_seconds=duration,
                    metadata={
                        "total_chars": total_chars,
                        "doc_count": doc_count,
                    },
                )

        except Exception as e:
            logger.error(f"[INSERT] Error during insertion: {e}")
            raise

    async def ingest_files(
        self,
        folder_path: str | Path,
        config: IngestionConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest documents from a folder using Docling document processor.

        This method uses the new ingestion pipeline that supports:
        - Multi-format documents: PDF, Word, PPT, Excel, HTML, Markdown
        - Audio transcription via Whisper ASR
        - Structure-aware chunking via Docling HybridChunker
        - Automatic embedding generation

        Args:
            folder_path: Path to folder containing documents.
            config: Optional ingestion configuration. Defaults to sensible values.
            progress_callback: Optional callback for progress updates (current, total).

        Returns:
            List of IngestionResult objects with details of each ingestion.

        Example:
            ```python
            rag = HybridRAG()
            await rag.initialize()

            # Ingest all documents from a folder
            results = await rag.ingest_files("./documents")

            # Check results
            for r in results:
                if r.success:
                    print(f"✓ {r.title}: {r.chunks_created} chunks")
                else:
                    print(f"✗ {r.title}: {r.errors}")
            ```
        """
        self._ensure_initialized()

        logger.info("[INGEST_FILES] ========== Starting File Ingestion ==========")
        logger.info(f"[INGEST_FILES] Folder: {folder_path}")

        # Get MongoDB database using shared singleton client (C2 fix)
        from .mongodb_client import get_shared_client

        client = get_shared_client(self.settings)
        db = client[self.settings.mongodb_database]

        # Use cached embedding function for the pipeline
        pipeline_embed_func = self._get_pipeline_embed_func()

        # Use default config if not provided
        if config is None:
            config = IngestionConfig(
                chunking=ChunkingConfig(
                    max_tokens=512,
                    chunk_size=1000,
                    chunk_overlap=200,
                    tokenizer_model="sentence-transformers/all-MiniLM-L6-v2",
                ),
                clean_before_ingest=False,  # Don't clean by default
                batch_size=self.settings.embedding_batch_size,
                enable_audio_transcription=True,
            )

        # Create and run the ingestion pipeline
        pipeline = DocumentIngestionPipeline(
            db=db,
            embedding_func=pipeline_embed_func,
            config=config,
            documents_collection="ingested_documents",
            chunks_collection="ingested_chunks",
        )

        import time as _time

        start_time = _time.time()

        results = await pipeline.ingest_folder(folder_path, progress_callback)

        duration = _time.time() - start_time

        # Summary statistics
        total_docs = len(results)
        successful = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info("[INGEST_FILES] ========== File Ingestion Complete ==========")
        logger.info(f"[INGEST_FILES] Documents: {successful}/{total_docs} successful")
        logger.info(f"[INGEST_FILES] Total chunks: {total_chunks}")
        logger.info(f"[INGEST_FILES] Duration: {duration:.2f}s")
        if total_errors > 0:
            logger.warning(f"[INGEST_FILES] Errors: {total_errors}")

        # Now insert the chunks into the main RAG system for KG extraction
        # Read back ONLY chunks from this ingestion batch (not entire collection)
        # [Rule: query-use-projection] [Rule: query-batch-operations]
        if successful > 0:
            logger.info("[INGEST_FILES] Inserting chunks into RAG for KG extraction...")
            chunks_col = db["ingested_chunks"]
            # Filter by document IDs from this batch to avoid full collection scan
            batch_doc_ids = [
                r.document_id for r in results if r.success and r.document_id
            ]
            from bson import ObjectId

            batch_object_ids = []
            for did in batch_doc_ids:
                try:
                    batch_object_ids.append(ObjectId(did))
                except Exception:
                    logger.warning(f"Could not convert document_id to ObjectId: {did}")

            if batch_object_ids:
                cursor = chunks_col.find(
                    {"document_id": {"$in": batch_object_ids}},
                    {"content": 1, "metadata.source": 1, "_id": 0},
                )
            else:
                # All ObjectId conversions failed - use string IDs instead
                # Do NOT fall back to find({}) which scans entire collection
                cursor = chunks_col.find(
                    {"document_id": {"$in": batch_doc_ids}},
                    {"content": 1, "metadata.source": 1, "_id": 0},
                )
            chunks_data = await cursor.to_list(length=10000)

            if chunks_data:
                # Extract content and file paths for RAG insertion
                contents = [c["content"] for c in chunks_data]
                file_paths = [
                    c.get("metadata", {}).get("source", "unknown") for c in chunks_data
                ]

                # Insert into main RAG (this builds the knowledge graph)
                await self.insert(
                    documents=contents,
                    file_paths=file_paths,
                )
                logger.info(
                    f"[INGEST_FILES] Inserted {len(contents)} chunks into RAG system"
                )

        # Shared client lifecycle managed globally (C3 fix: no per-method close)

        # Log to Langfuse if enabled
        if langfuse_enabled():
            log_ingestion(
                file_name=str(folder_path),
                num_chunks=total_chunks,
                num_entities=0,
                num_relations=0,
                duration_seconds=duration,
                metadata={
                    "total_documents": total_docs,
                    "successful_documents": successful,
                    "total_errors": total_errors,
                    "source": "ingest_files",
                },
            )

        return results

    async def ingest_file(
        self,
        file_path: str | Path,
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        """
        Ingest a single file using Docling document processor.

        Convenience method that wraps ingest_files() for single file ingestion.

        Args:
            file_path: Path to the file to ingest.
            config: Optional ingestion configuration.

        Returns:
            IngestionResult with details of the ingestion.

        Example:
            ```python
            rag = HybridRAG()
            await rag.initialize()

            result = await rag.ingest_file("./report.pdf")
            if result.success:
                print(f"Ingested {result.chunks_created} chunks")
            ```
        """
        from pathlib import Path as PathLib

        file_path = PathLib(file_path).resolve()

        if not file_path.exists():
            return IngestionResult(
                document_id="",
                title=file_path.name,
                chunks_created=0,
                processing_time_ms=0,
                errors=[f"File not found: {file_path}"],
                source=str(file_path),
            )

        # Create a temp folder with just this file (symlink or copy)
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = PathLib(temp_dir) / file_path.name
            shutil.copy(file_path, temp_file)

            results = await self.ingest_files(temp_dir, config)

            if results:
                return results[0]
            else:
                return IngestionResult(
                    document_id="",
                    title=file_path.name,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=["No results returned from ingestion"],
                    source=str(file_path),
                )

    async def ingest_url(
        self,
        url: str,
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        """
        Ingest content from a single URL using Tavily Extract API.

        This method extracts content from a web URL and ingests it into
        HybridRAG using the same pipeline as file ingestion.

        Args:
            url: URL to extract content from.
            config: Optional ingestion configuration.

        Returns:
            IngestionResult with details of the ingestion.

        Example:
            ```python
            rag = HybridRAG()
            await rag.initialize()

            result = await rag.ingest_url("https://docs.mongodb.com/atlas/")
            if result.success:
                print(f"Ingested {result.chunks_created} chunks")
            ```

        Raises:
            ValueError: If Tavily API key is not configured or URL is invalid.
        """
        self._ensure_initialized()

        logger.info("[INGEST_URL] ========== Starting URL Ingestion ==========")
        logger.info(f"[INGEST_URL] URL: {url}")

        # Check if Tavily API key is configured
        if not self.settings.tavily_api_key:
            error_msg = "Tavily API key not configured. Set TAVILY_API_KEY environment variable."
            logger.error(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=0,
                errors=[error_msg],
                source=url,
                format_type="web",
            )

        import time as _time

        start_time = _time.time()

        try:
            # Import Tavily processor
            from ..ingestion.tavily_processor import (
                BadRequestError,
                ForbiddenError,
                InvalidAPIKeyError,
                MissingAPIKeyError,
                TavilyProcessor,
                UsageLimitExceededError,
            )
            from ..ingestion.tavily_processor import (
                TimeoutError as TavilyTimeoutError,
            )

            # Create Tavily processor
            processor = TavilyProcessor(
                api_key=self.settings.tavily_api_key.get_secret_value()
            )

            # Extract content from URL
            processed_doc = await processor.extract_url(url)

            # Use existing ingestion pipeline with shared client (C2/C3 fix)
            from .mongodb_client import get_shared_client

            client = get_shared_client(self.settings)
            db = client[self.settings.mongodb_database]

            # Use cached embedding function for the pipeline
            pipeline_embed_func = self._get_pipeline_embed_func()

            # Use default config if not provided
            if config is None:
                config = IngestionConfig(
                    chunking=ChunkingConfig(
                        max_tokens=512,
                        chunk_size=1000,
                        chunk_overlap=200,
                        tokenizer_model="sentence-transformers/all-MiniLM-L6-v2",
                    ),
                    clean_before_ingest=False,
                    batch_size=self.settings.embedding_batch_size,
                    enable_audio_transcription=False,  # No audio for web content
                )

            # Create pipeline and ingest text
            pipeline = DocumentIngestionPipeline(
                db=db,
                embedding_func=pipeline_embed_func,
                config=config,
                documents_collection="ingested_documents",
                chunks_collection="ingested_chunks",
            )

            result = await pipeline.ingest_text(
                content=processed_doc.content,
                title=processed_doc.title,
                source=processed_doc.source,
                metadata=processed_doc.metadata,
            )

            duration = _time.time() - start_time

            # Insert chunks into RAG for KG extraction
            if result.success:
                logger.info(
                    "[INGEST_URL] Inserting chunks into RAG for KG extraction..."
                )
                chunks_col = db["ingested_chunks"]
                # [Rule: query-use-projection] Only fetch needed fields
                cursor = chunks_col.find(
                    {"document_id": result.document_id},
                    {"content": 1, "metadata.source": 1, "_id": 0},
                )
                chunks_data = await cursor.to_list(length=10000)

                if chunks_data:
                    contents = [c["content"] for c in chunks_data]
                    sources = [
                        c.get("metadata", {}).get("source", url) for c in chunks_data
                    ]

                    await self.insert(documents=contents, file_paths=sources)
                    logger.info(
                        f"[INGEST_URL] Inserted {len(contents)} chunks into RAG system"
                    )

            # Shared client lifecycle managed globally (C3 fix: no per-method close)

            # Log to Langfuse if enabled
            if langfuse_enabled():
                log_ingestion(
                    file_name=url,
                    num_chunks=result.chunks_created,
                    num_entities=0,
                    num_relations=0,
                    duration_seconds=duration,
                    metadata={
                        "source": "ingest_url",
                        "url": url,
                        "api_type": "extract",
                    },
                )

            logger.info("[INGEST_URL] ========== URL Ingestion Complete ==========")
            logger.info(
                f"[INGEST_URL] Success: {result.success}, Chunks: {result.chunks_created}, Duration: {duration:.2f}s"
            )

            return result

        except MissingAPIKeyError as e:
            error_msg = f"Tavily API key not configured: {e}"
            logger.error(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )
        except InvalidAPIKeyError as e:
            error_msg = f"Invalid Tavily API key: {e}"
            logger.error(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )
        except UsageLimitExceededError as e:
            error_msg = f"Tavily rate limit exceeded: {e}"
            logger.warning(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )
        except BadRequestError as e:
            error_msg = f"Invalid URL or request: {e}"
            logger.error(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )
        except TavilyTimeoutError as e:
            error_msg = f"Tavily request timeout: {e}"
            logger.error(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )
        except ForbiddenError as e:
            error_msg = f"Tavily access forbidden: {e}"
            logger.error(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )
        except Exception as e:
            error_msg = f"Unexpected error ingesting URL: {e}"
            logger.exception(f"[INGEST_URL] {error_msg}")
            return IngestionResult(
                document_id="",
                title=url,
                chunks_created=0,
                processing_time_ms=(_time.time() - start_time) * 1000,
                errors=[error_msg],
                source=url,
                format_type="web",
            )

    async def ingest_website(
        self,
        url: str,
        max_pages: int = 10,
        max_depth: int = 2,
        config: IngestionConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest content from a website by crawling multiple pages using Tavily Crawl API.

        This method crawls a website starting from the given URL and ingests
        all crawled pages into HybridRAG.

        Args:
            url: Base URL to start crawling from.
            max_pages: Maximum number of pages to crawl (default: 10).
            max_depth: Maximum crawl depth (default: 2).
            config: Optional ingestion configuration.
            progress_callback: Optional callback for progress updates (current, total).

        Returns:
            List of IngestionResult objects, one per crawled page.

        Example:
            ```python
            rag = HybridRAG()
            await rag.initialize()

            results = await rag.ingest_website(
                "https://docs.mongodb.com/atlas/",
                max_pages=5
            )
            successful = sum(1 for r in results if r.success)
            print(f"Ingested {successful} pages")
            ```

        Raises:
            ValueError: If Tavily API key is not configured or URL is invalid.
        """
        self._ensure_initialized()

        logger.info("[INGEST_WEBSITE] ========== Starting Website Crawl ==========")
        logger.info(
            f"[INGEST_WEBSITE] URL: {url}, max_pages={max_pages}, max_depth={max_depth}"
        )

        # Check if Tavily API key is configured
        if not self.settings.tavily_api_key:
            error_msg = "Tavily API key not configured. Set TAVILY_API_KEY environment variable."
            logger.error(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]

        import time as _time

        start_time = _time.time()

        try:
            # Import Tavily processor
            from ..ingestion.tavily_processor import (
                BadRequestError,
                ForbiddenError,
                InvalidAPIKeyError,
                MissingAPIKeyError,
                TavilyProcessor,
                UsageLimitExceededError,
            )
            from ..ingestion.tavily_processor import (
                TimeoutError as TavilyTimeoutError,
            )

            # Create Tavily processor
            processor = TavilyProcessor(
                api_key=self.settings.tavily_api_key.get_secret_value()
            )

            # Crawl website
            processed_docs = await processor.crawl_website(
                url=url, max_depth=max_depth, max_pages=max_pages
            )

            if not processed_docs:
                error_msg = f"No content extracted from website: {url}"
                logger.error(f"[INGEST_WEBSITE] {error_msg}")
                return [
                    IngestionResult(
                        document_id="",
                        title=url,
                        chunks_created=0,
                        processing_time_ms=(_time.time() - start_time) * 1000,
                        errors=[error_msg],
                        source=url,
                        format_type="web",
                    )
                ]

            logger.info(
                f"[INGEST_WEBSITE] Extracted {len(processed_docs)} pages, processing..."
            )

            # Use existing ingestion pipeline with shared client (C2/C3 fix)
            from .mongodb_client import get_shared_client

            client = get_shared_client(self.settings)
            db = client[self.settings.mongodb_database]

            # Use cached embedding function for the pipeline
            pipeline_embed_func = self._get_pipeline_embed_func()

            # Use default config if not provided
            if config is None:
                config = IngestionConfig(
                    chunking=ChunkingConfig(
                        max_tokens=512,
                        chunk_size=1000,
                        chunk_overlap=200,
                        tokenizer_model="sentence-transformers/all-MiniLM-L6-v2",
                    ),
                    clean_before_ingest=False,
                    batch_size=self.settings.embedding_batch_size,
                    enable_audio_transcription=False,  # No audio for web content
                )

            # Create pipeline
            pipeline = DocumentIngestionPipeline(
                db=db,
                embedding_func=pipeline_embed_func,
                config=config,
                documents_collection="ingested_documents",
                chunks_collection="ingested_chunks",
            )

            # Process each page
            results = []
            all_chunks_data = []

            for i, processed_doc in enumerate(processed_docs):
                if progress_callback:
                    progress_callback(i + 1, len(processed_docs))

                result = await pipeline.ingest_text(
                    content=processed_doc.content,
                    title=processed_doc.title,
                    source=processed_doc.source,
                    metadata=processed_doc.metadata,
                )
                results.append(result)

                # Collect chunks for RAG insertion
                if result.success:
                    chunks_col = db["ingested_chunks"]
                    # [Rule: query-use-projection] Only fetch needed fields
                    cursor = chunks_col.find(
                        {"document_id": result.document_id},
                        {"content": 1, "metadata.source": 1, "_id": 0},
                    )
                    chunks_data = await cursor.to_list(length=10000)
                    all_chunks_data.extend(chunks_data)

            duration = _time.time() - start_time

            # Insert all chunks into RAG for KG extraction
            if all_chunks_data:
                logger.info(
                    "[INGEST_WEBSITE] Inserting chunks into RAG for KG extraction..."
                )
                contents = [c["content"] for c in all_chunks_data]
                sources = [
                    c.get("metadata", {}).get("source", url) for c in all_chunks_data
                ]

                await self.insert(documents=contents, file_paths=sources)
                logger.info(
                    f"[INGEST_WEBSITE] Inserted {len(contents)} chunks into RAG system"
                )

            # Shared client lifecycle managed globally (C3 fix: no per-method close)

            # Summary statistics
            total_docs = len(results)
            successful = sum(1 for r in results if r.success)
            total_chunks = sum(r.chunks_created for r in results)
            total_errors = sum(len(r.errors) for r in results)

            logger.info("[INGEST_WEBSITE] ========== Website Crawl Complete ==========")
            logger.info(f"[INGEST_WEBSITE] Pages: {successful}/{total_docs} successful")
            logger.info(f"[INGEST_WEBSITE] Total chunks: {total_chunks}")
            logger.info(f"[INGEST_WEBSITE] Duration: {duration:.2f}s")
            if total_errors > 0:
                logger.warning(f"[INGEST_WEBSITE] Errors: {total_errors}")

            # Log to Langfuse if enabled
            if langfuse_enabled():
                log_ingestion(
                    file_name=url,
                    num_chunks=total_chunks,
                    num_entities=0,
                    num_relations=0,
                    duration_seconds=duration,
                    metadata={
                        "source": "ingest_website",
                        "url": url,
                        "api_type": "crawl",
                        "total_pages": total_docs,
                        "successful_pages": successful,
                        "max_pages": max_pages,
                        "max_depth": max_depth,
                    },
                )

            return results

        except MissingAPIKeyError as e:
            error_msg = f"Tavily API key not configured: {e}"
            logger.error(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]
        except InvalidAPIKeyError as e:
            error_msg = f"Invalid Tavily API key: {e}"
            logger.error(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]
        except UsageLimitExceededError as e:
            error_msg = f"Tavily rate limit exceeded: {e}"
            logger.warning(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]
        except BadRequestError as e:
            error_msg = f"Invalid URL or request: {e}"
            logger.error(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]
        except TavilyTimeoutError as e:
            error_msg = f"Tavily request timeout: {e}"
            logger.error(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]
        except ForbiddenError as e:
            error_msg = f"Tavily access forbidden: {e}"
            logger.error(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]
        except Exception as e:
            error_msg = f"Unexpected error ingesting website: {e}"
            logger.exception(f"[INGEST_WEBSITE] {error_msg}")
            return [
                IngestionResult(
                    document_id="",
                    title=url,
                    chunks_created=0,
                    processing_time_ms=(_time.time() - start_time) * 1000,
                    errors=[error_msg],
                    source=url,
                    format_type="web",
                )
            ]

    async def query(
        self,
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
        | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        enable_rerank: bool | None = None,
        only_context: bool = False,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        Query the RAG system.

        Args:
            query: Search query
            mode: Query mode (local, global, hybrid, naive, mix, bypass)
            top_k: Number of results to retrieve
            rerank_top_k: Number of results after reranking
            enable_rerank: Whether to enable reranking
            only_context: If True, return only context without LLM response
            system_prompt: Optional system prompt for LLM
            stream: If True, return an async iterator of response chunks

        Returns:
            Generated response, context string, or streaming iterator
        """
        rag = self._ensure_initialized()

        # Validate query length [Rule: input-validation]
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > self.settings.max_query_length:
            raise ValueError(
                f"Query too long ({len(query)} chars). "
                f"Maximum allowed: {self.settings.max_query_length} chars."
            )

        # Resolve parameters with defaults
        resolved_mode = mode or self.settings.default_query_mode
        resolved_top_k = top_k or self.settings.default_top_k
        resolved_rerank_top_k = rerank_top_k or self.settings.default_rerank_top_k
        resolved_enable_rerank = (
            enable_rerank if enable_rerank is not None else self.settings.enable_rerank
        )

        logger.info("[QUERY] ========== Starting Query ==========")
        logger.info(
            f"[QUERY] Query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )
        logger.info(
            f"[QUERY] Mode: {resolved_mode}, top_k: {resolved_top_k}, rerank_top_k: {resolved_rerank_top_k}"
        )
        logger.info(
            f"[QUERY] Rerank enabled: {resolved_enable_rerank}, only_context: {only_context}, stream: {stream}"
        )

        # Apply implicit expansion if enabled
        expanded_query = query
        if self.settings.enable_implicit_expansion and hasattr(rag, "entities_vdb"):
            expanded_query = await self._expand_query_with_entities(query, rag)
            if expanded_query != query:
                logger.info("[QUERY] Implicit expansion applied, query expanded")

        # Build query parameters with defaults from settings
        param = QueryParam(
            mode=resolved_mode,
            top_k=resolved_top_k,
            chunk_top_k=resolved_rerank_top_k,
            enable_rerank=resolved_enable_rerank,
            only_need_context=only_context,
            stream=stream,
        )

        try:
            if stream and only_context:
                raise ValueError("Streaming is not supported when only_context=True")

            if only_context:
                result = await self.query_data(
                    query=expanded_query,
                    mode=resolved_mode,
                    top_k=resolved_top_k,
                    rerank_top_k=resolved_rerank_top_k,
                    enable_rerank=resolved_enable_rerank,
                )
                context = result.get("context", "")
                logger.info(f"[QUERY] Response received: {len(context)} chars")
                logger.info("[QUERY] ========== Query Complete ==========")
                return context

            if not self.settings.enable_llm:
                if stream:
                    raise ValueError(
                        "Streaming requires enable_llm=True on the blessed stack"
                    )
                logger.info(
                    "[QUERY] LLM disabled, returning retrieved context instead of generation"
                )
                result = await self.query_data(
                    query=expanded_query,
                    mode=resolved_mode,
                    top_k=resolved_top_k,
                    rerank_top_k=resolved_rerank_top_k,
                    enable_rerank=resolved_enable_rerank,
                )
                context = result.get("context", "")
                logger.info(f"[QUERY] Response received: {len(context)} chars")
                logger.info("[QUERY] ========== Query Complete ==========")
                return context

            logger.info("[QUERY] Executing query...")
            response = await rag.aquery(
                query=expanded_query,
                param=param._to_internal(),  # Convert to internal QueryParam
                system_prompt=system_prompt,
            )

            if stream and response is not None and not isinstance(response, str):
                logger.info("[QUERY] Streaming response iterator returned")
                logger.info("[QUERY] ========== Query Complete ==========")
                return response

            # Handle None response gracefully
            if response is None:
                logger.warning(
                    "[QUERY] RAG engine returned None response - possible LLM error or empty results"
                )
                response = ""

            response_len = len(response)
            logger.info(f"[QUERY] Response received: {response_len} chars")
            logger.info("[QUERY] ========== Query Complete ==========")
            return response
        except Exception as e:
            logger.error(f"[QUERY] Error during query: {e}")
            raise

    async def query_data(
        self,
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
        | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        enable_rerank: bool | None = None,
    ) -> dict[str, Any]:
        """Return structured retrieval data and the assembled context string."""
        rag = self._ensure_initialized()

        resolved_mode = mode or self.settings.default_query_mode
        resolved_top_k = top_k or self.settings.default_top_k
        resolved_rerank_top_k = rerank_top_k or self.settings.default_rerank_top_k
        resolved_enable_rerank = (
            enable_rerank if enable_rerank is not None else self.settings.enable_rerank
        )

        param = QueryParam(
            mode=resolved_mode,
            top_k=resolved_top_k,
            chunk_top_k=resolved_rerank_top_k,
            enable_rerank=resolved_enable_rerank,
            only_need_context=True,
        )

        result = await rag.aquery_llm(query=query, param=param._to_internal())
        data = dict(result)
        data["context"] = _extract_context_from_query_data(result)
        metadata = dict(result.get("metadata", {}))
        metadata.update(_extract_retrieval_diagnostics(result))
        data["metadata"] = metadata
        return data

    async def query_with_sources(
        self,
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
        | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Query and return both response and source context.

        Optimized to only do retrieval once, then generate response using LLM directly.
        This avoids the MongoDB connection issues that can occur with repeated queries.

        Args:
            query: Search query
            mode: Query mode
            top_k: Number of results

        Returns:
            Dict with 'answer' and 'context' keys
        """
        self._ensure_initialized()
        resolved_mode = mode or self.settings.default_query_mode

        logger.info(
            f"[QUERY_WITH_SOURCES] Starting query with sources: mode={resolved_mode}"
        )

        # Step 1: Get context and structured retrieval data (single retrieval)
        logger.info("[QUERY_WITH_SOURCES] Step 1: Fetching context and references...")
        structured = await self.query_data(
            query=query,
            mode=mode,
            top_k=top_k,
        )
        context = structured.get("context", "")
        context_len = len(context) if context else 0
        logger.info(f"[QUERY_WITH_SOURCES] Context retrieved: {context_len} chars")
        references = _extract_references_from_query_data(structured)
        metadata = structured.get("metadata", {})

        # Step 2: Generate response using LLM directly (no second retrieval)
        logger.info("[QUERY_WITH_SOURCES] Step 2: Generating response from context...")
        if context and self._llm_func:
            # Create prompt for direct LLM generation
            generation_prompt = f"""Based on the following context, please answer the user's question.

---Context---
{context}

---Question---
{query}

---Answer---
Please provide a comprehensive answer based only on the information provided in the context above."""

            try:
                response = await self._llm_func(generation_prompt)
                response = response.strip() if response else ""
            except Exception as e:
                logger.error(f"[QUERY_WITH_SOURCES] LLM generation failed: {e}")
                response = "I apologize, but I was unable to generate a response. Please try again."
        else:
            response = context if context else "No relevant information found."

        response_len = len(response) if response else 0
        logger.info(f"[QUERY_WITH_SOURCES] Response generated: {response_len} chars")

        # Log to Langfuse if enabled
        if langfuse_enabled():
            log_rag_query(
                query=query,
                mode=resolved_mode,
                response=response,
                context=context,
                metadata={
                    "response_length": response_len,
                    "context_length": context_len,
                    "source": "query_with_sources",
                },
            )

        return {
            "answer": response,
            "context": context,
            "query": query,
            "mode": resolved_mode,
            "references": references,
            "metadata": metadata,
        }

    async def stream_query(
        self,
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
        | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        enable_rerank: bool | None = None,
        include_context: bool = False,
        include_references: bool = True,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Return a streaming iterator plus source metadata for a query."""
        rag = self._ensure_initialized()

        if not self.settings.enable_llm:
            raise ValueError("Streaming requires enable_llm=True on the blessed stack")

        resolved_mode = mode or self.settings.default_query_mode
        resolved_top_k = top_k or self.settings.default_top_k
        resolved_rerank_top_k = rerank_top_k or self.settings.default_rerank_top_k
        resolved_enable_rerank = (
            enable_rerank if enable_rerank is not None else self.settings.enable_rerank
        )

        param = QueryParam(
            mode=resolved_mode,
            top_k=resolved_top_k,
            chunk_top_k=resolved_rerank_top_k,
            enable_rerank=resolved_enable_rerank,
            stream=True,
            include_references=include_references,
        )
        result = await rag.aquery_llm(
            query=query,
            param=param._to_internal(),
            system_prompt=system_prompt,
        )

        llm_response = result.get("llm_response", {})
        response_iterator = llm_response.get("response_iterator")

        if not llm_response.get("is_streaming") or response_iterator is None:
            content = llm_response.get("content") or ""

            async def _single_chunk() -> AsyncIterator[str]:
                if content:
                    yield content

            response_iterator = _single_chunk()

        return {
            "query": query,
            "mode": resolved_mode,
            "context": _extract_context_from_query_data(result)
            if include_context
            else None,
            "references": (
                _extract_references_from_query_data(result)
                if include_references
                else []
            ),
            "metadata": {
                **result.get("metadata", {}),
                **_extract_retrieval_diagnostics(result),
            },
            "response_iterator": response_iterator,
        }

    async def query_with_memory(
        self,
        query: str,
        session_id: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
        | None = None,
        top_k: int | None = None,
        max_history_messages: int = 10,
    ) -> dict[str, Any]:
        """
        Query with conversation memory - enables multi-turn conversations.

        This method:
        1. Gets conversation history from the session
        2. Augments the query with conversation context
        3. Retrieves relevant documents using enhanced query
        4. Generates response with full conversation context
        5. Stores user query and assistant response in session

        Args:
            query: User's query
            session_id: Conversation session ID (created if doesn't exist)
            mode: Query mode
            top_k: Number of results
            max_history_messages: Max history messages to include

        Returns:
            Dict with 'answer', 'context', 'session_id', 'history_used'
        """
        self._ensure_initialized()
        resolved_mode = mode or self.settings.default_query_mode

        logger.info(f"[QUERY_WITH_MEMORY] Session: {session_id}, mode={resolved_mode}")

        # Ensure session exists
        session = await self._memory.get_session(session_id)
        if not session:
            session_id = await self._memory.create_session(session_id)
            logger.info(f"[QUERY_WITH_MEMORY] Created new session: {session_id}")

        # Get conversation history
        history = await self._memory.get_history(session_id, max_history_messages)
        history_context = await self._memory.get_context_string(
            session_id, max_history_messages
        )

        logger.info(f"[QUERY_WITH_MEMORY] History messages: {len(history)}")

        # Store user query first
        await self._memory.add_message(session_id, "user", query)

        # Augment query with conversation context for better retrieval
        if history_context:
            augmented_query = f"{history_context}\n\nCurrent question: {query}"
        else:
            augmented_query = query

        # Get context with augmented query
        logger.info("[QUERY_WITH_MEMORY] Fetching context with augmented query...")
        context = await self.query(
            query=augmented_query,
            mode=mode,
            top_k=top_k,
            only_context=True,
        )
        context_len = len(context) if context else 0
        logger.info(f"[QUERY_WITH_MEMORY] Context retrieved: {context_len} chars")

        # Generate response with full conversation context
        if context and self._llm_func:
            # Build conversation history string for LLM
            history_str = ""
            if history:
                history_str = "Previous conversation:\n"
                for msg in history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_str += f"{role}: {msg['content']}\n"
                history_str += "\n"

            generation_prompt = f"""You are a helpful assistant having a conversation about documents.

{history_str}---Retrieved Context---
{context}

---Current Question---
{query}

---Instructions---
Based on the conversation history and retrieved context above, please answer the user's current question.
If this is a follow-up question (like "explain that", "what do you mean", "tell me more"), use the conversation history to understand what the user is referring to.
Provide a helpful, comprehensive answer."""

            try:
                response = await self._llm_func(generation_prompt)
                response = response.strip() if response else ""
            except Exception as e:
                logger.error(f"[QUERY_WITH_MEMORY] LLM generation failed: {e}")
                response = "I apologize, but I was unable to generate a response. Please try again."
        else:
            response = context if context else "No relevant information found."

        # Store assistant response
        await self._memory.add_message(session_id, "assistant", response)

        response_len = len(response) if response else 0
        logger.info(f"[QUERY_WITH_MEMORY] Response generated: {response_len} chars")

        # Log to Langfuse if enabled
        if langfuse_enabled():
            log_rag_query(
                query=query,
                mode=resolved_mode,
                response=response,
                context=context,
                metadata={
                    "response_length": response_len,
                    "context_length": context_len,
                    "session_id": session_id,
                    "history_messages": len(history),
                    "source": "query_with_memory",
                },
            )

        return {
            "answer": response,
            "context": context,
            "query": query,
            "session_id": session_id,
            "mode": resolved_mode,
            "history_used": len(history),
        }

    async def create_conversation_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional custom session ID
            metadata: Optional metadata (e.g., book name, user info)

        Returns:
            Session ID
        """
        self._ensure_initialized()
        return await self._memory.create_session(session_id, metadata)

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID
            limit: Max messages to return

        Returns:
            List of message dicts
        """
        self._ensure_initialized()
        return await self._memory.get_messages(session_id, limit)

    async def clear_conversation(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session ID
        """
        self._ensure_initialized()
        await self._memory.clear_session(session_id)

    async def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the RAG system.

        Args:
            doc_id: Document ID to delete
        """
        rag = self._ensure_initialized()
        await rag.adelete_by_doc_id(doc_id)

    async def get_status(self) -> dict[str, Any]:
        """
        Get system status information.

        Returns:
            Dict with status information
        """
        # Determine current models based on providers
        embedding_model = {
            "voyage": self.settings.voyage_embedding_model,
            "openai": self.settings.openai_embedding_model,
            "gemini": self.settings.gemini_embedding_model,
        }.get(self.settings.embedding_provider, "unknown")

        llm_model = {
            "anthropic": self.settings.anthropic_model,
            "openai": self.settings.openai_model,
            "gemini": self.settings.gemini_model,
        }.get(self.settings.llm_provider, "unknown")

        return {
            "initialized": self._initialized,
            "working_dir": self.working_dir,
            "mongodb_database": self.settings.mongodb_database,
            "mongodb_workspace": self.settings.mongodb_workspace,
            "llm_provider": self.settings.llm_provider,
            "llm_model": llm_model,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": embedding_model,
            "rerank_model": self.settings.voyage_rerank_model
            if self.settings.voyage_api_key
            else None,
            "blessed_stack": {
                "mongodb": "Atlas 8.2+ / atlas-local:preview",
                "embedding": "Voyage",
                "llm": "OpenAI-compatible API",
                "shape": "backend-first",
            },
            "capabilities": {
                "citations": True,
                "streaming": True,
                "conversation_memory": True,
                "source_aware_queries": True,
                "fail_fast": True,
                "supported_query_modes": [
                    "local",
                    "global",
                    "hybrid",
                    "naive",
                    "mix",
                    "bypass",
                ],
            },
            "enhancements": {
                "implicit_expansion": self.settings.enable_implicit_expansion,
                "entity_boosting": self.settings.enable_entity_boosting,
            },
        }

    async def get_knowledge_base_stats(self) -> dict[str, Any]:
        """
        Get knowledge base statistics including documents, entities, and relationships.

        Returns:
            Dict with knowledge base statistics
        """
        rag = self._ensure_initialized()

        stats = {
            "documents": {"total": 0, "by_status": {}},
            "entities": 0,
            "relationships": 0,
            "chunks": 0,
            "recent_documents": [],
        }

        try:
            # Get document counts by status
            doc_counts = await rag.doc_status.get_all_status_counts()
            stats["documents"]["total"] = doc_counts.get("all", 0)
            stats["documents"]["by_status"] = {
                k: v for k, v in doc_counts.items() if k != "all"
            }

            # Get entity count -- use estimated_document_count() for unfiltered
            # counts (avoids collection scan per mongodb-query-optimizer skill)
            entity_count = await rag.entities_vdb._data.estimated_document_count()
            stats["entities"] = entity_count

            # Get relationship count
            relationship_count = (
                await rag.relationships_vdb._data.estimated_document_count()
            )
            stats["relationships"] = relationship_count

            # Get chunk count
            chunk_count = await rag.chunks_vdb._data.estimated_document_count()
            stats["chunks"] = chunk_count

            # Get recent documents (last 10)
            cursor = (
                rag.doc_status._data.find(
                    {},
                    {
                        "_id": 0,
                        "id": 1,
                        "file_path": 1,
                        "status": 1,
                        "created_at": 1,
                        "chunks_count": 1,
                    },
                )
                .sort("created_at", -1)
                .limit(10)
            )

            async for doc in cursor:
                stats["recent_documents"].append(
                    {
                        "id": doc.get("id", "unknown")[:12] + "...",
                        "file": doc.get("file_path", "unknown"),
                        "status": doc.get("status", "unknown"),
                        "chunks": doc.get("chunks_count", 0),
                    }
                )

        except Exception as e:
            logger.error(f"[STATS] Error getting knowledge base stats: {e}")

        return stats


async def create_hybridrag(
    settings: Settings | None = None,
    working_dir: str = "./hybridrag_workspace",
    auto_initialize: bool = True,
) -> HybridRAG:
    """
    Factory function to create and optionally initialize HybridRAG.

    Args:
        settings: Optional settings override
        working_dir: Working directory for cache
        auto_initialize: Whether to initialize automatically

    Returns:
        HybridRAG instance
    """
    rag = HybridRAG(
        settings=settings or get_settings(),
        working_dir=working_dir,
    )

    if auto_initialize:
        await rag.initialize()

    return rag

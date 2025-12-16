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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

# Internal RAG engine imports (bundled in hybridrag.engine)
from ..engine import BaseRAGEngine as _RAGEngine
from ..engine import QueryParam as _QueryParam
from ..engine import EmbeddingFunc


# Re-export QueryParam with our own name
@dataclass
class QueryParam:
    """Query parameters for RAG operations."""
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    top_k: int = 60
    chunk_top_k: int = 10
    enable_rerank: bool = True
    only_need_context: bool = False

    def _to_internal(self) -> _QueryParam:
        """Convert to internal format."""
        return _QueryParam(
            mode=self.mode,
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
            enable_rerank=self.enable_rerank,
            only_need_context=self.only_need_context,
        )

from ..config.settings import Settings, get_settings
from ..enhancements.entity_boosting import create_boosted_rerank_func
from ..enhancements.implicit_expansion import ImplicitExpander
from ..ingestion import (
    DocumentIngestionPipeline,
    IngestionConfig,
    IngestionResult,
    ChunkingConfig,
)
from ..integrations import (
    log_rag_query,
    log_ingestion,
    get_langfuse,
    flush_langfuse,
    langfuse_enabled,
)
from ..memory import ConversationMemory

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Sequence

    import numpy as np

logger = logging.getLogger("hybridrag.core")
logger.setLevel(logging.INFO)


def _create_embedding_func(settings: Settings) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Create Voyage AI embedding function - VOYAGE ONLY, no fallbacks."""
    from ..integrations.voyage import create_embedding_func

    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY is REQUIRED - Voyage AI is the only supported embedding provider")

    embed_func = create_embedding_func(
        api_key=settings.voyage_api_key.get_secret_value(),
        model=settings.voyage_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    logger.info(f"[INIT] Voyage embedding configured: model={settings.voyage_embedding_model}, dim=1024, batch_size={settings.embedding_batch_size}")
    return embed_func, 1024  # voyage-3-large dimension


def _create_llm_func(settings: Settings) -> Callable[..., str]:
    """Create LLM function based on provider setting."""
    provider = settings.llm_provider
    logger.info(f"[INIT] Creating LLM function for provider: {provider}")

    if provider == "anthropic":
        from ..integrations.anthropic import create_llm_func

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required when llm_provider=anthropic")
        logger.info(f"[INIT] Anthropic LLM configured: model={settings.anthropic_model}")
        return create_llm_func(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.anthropic_model,
        )

    elif provider == "openai":
        from ..integrations.openai import create_openai_llm_func

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when llm_provider=openai")
        logger.info(f"[INIT] OpenAI LLM configured: model={settings.openai_model}")
        return create_openai_llm_func(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
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
    - Voyage AI embeddings (voyage-3-large)
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

    async def initialize(self) -> None:
        """
        Initialize HybridRAG with all components.

        Must be called before using insert() or query().
        """
        if self._initialized:
            logger.info("[INIT] HybridRAG already initialized, skipping")
            return

        logger.info("[INIT] ========== Starting HybridRAG Initialization ==========")

        # Set MongoDB environment variables
        os.environ["MONGO_URI"] = self.settings.mongodb_uri.get_secret_value()
        os.environ["MONGO_DATABASE"] = self.settings.mongodb_database
        logger.info(f"[INIT] MongoDB configured: database={self.settings.mongodb_database}")

        # Create embedding function based on provider
        logger.info("[INIT] Creating embedding function...")
        embed_func, embedding_dim = _create_embedding_func(self.settings)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=self.settings.max_token_size,
            func=embed_func,
        )
        logger.info(f"[INIT] EmbeddingFunc created: dim={embedding_dim}, max_tokens={self.settings.max_token_size}")

        # Create rerank function (Voyage AI - best quality)
        rerank_func = None
        if self.settings.voyage_api_key:
            from ..integrations.voyage import create_rerank_func

            logger.info(f"[INIT] Creating Voyage reranker: model={self.settings.voyage_rerank_model}")
            base_rerank = create_rerank_func(
                api_key=self.settings.voyage_api_key.get_secret_value(),
                model=self.settings.voyage_rerank_model,
            )

            # Wrap with entity boosting if enabled
            if self.settings.enable_entity_boosting:
                logger.info(f"[INIT] Wrapping with entity boosting: weight={self.settings.entity_boost_weight}")
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
        self._llm_func = llm_func  # Store for direct use in query_with_sources

        # Initialize RAG engine with MongoDB storage
        logger.info("[INIT] Initializing RAG engine with MongoDB storage...")
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
        logger.info("[INIT] Storage backends initialized (KV, Vector, Graph, DocStatus)")

        # Initialize pipeline status
        from ..engine.kg.shared_storage import initialize_pipeline_status
        logger.info("[INIT] Initializing pipeline status...")
        await initialize_pipeline_status()

        # Initialize conversation memory for multi-turn conversations
        logger.info("[INIT] Initializing conversation memory...")
        self._memory = ConversationMemory(
            mongodb_uri=self.settings.mongodb_uri.get_secret_value(),
            database=self.settings.mongodb_database,
        )
        await self._memory.initialize()
        logger.info("[INIT] Conversation memory initialized")

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
        logger.info(f"[INSERT] ========== Starting Document Insertion ==========")
        logger.info(f"[INSERT] Documents: {doc_count}, Total chars: {total_chars:,}")
        if file_paths:
            logger.info(f"[INSERT] File paths: {file_paths}")

        import time as _time
        start_time = _time.time()

        try:
            await rag.ainsert(
                input=list(documents),
                ids=list(ids) if ids else None,
                file_paths=list(file_paths) if file_paths else None,
            )
            duration = _time.time() - start_time
            logger.info(f"[INSERT] ========== Document Insertion Complete ==========")

            # Log to Langfuse if enabled
            if langfuse_enabled():
                log_ingestion(
                    file_name=file_paths[0] if file_paths else "unknown",
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
        folder_path: str | "Path",
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

        logger.info(f"[INGEST_FILES] ========== Starting File Ingestion ==========")
        logger.info(f"[INGEST_FILES] Folder: {folder_path}")

        # Get MongoDB database for the pipeline
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(self.settings.mongodb_uri.get_secret_value())
        db = client[self.settings.mongodb_database]

        # Create embedding function wrapper for the pipeline
        def pipeline_embed_func(texts: list[str]) -> list[list[float]]:
            """Wrapper to use HybridRAG's embedding function."""
            from ..integrations.voyage import create_embedding_func

            embed_func = create_embedding_func(
                api_key=self.settings.voyage_api_key.get_secret_value(),
                model=self.settings.voyage_embedding_model,
                batch_size=self.settings.embedding_batch_size,
            )
            # embed_func returns np.ndarray, convert to list
            result = embed_func(texts)
            return result.tolist() if hasattr(result, "tolist") else list(result)

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

        logger.info(f"[INGEST_FILES] ========== File Ingestion Complete ==========")
        logger.info(f"[INGEST_FILES] Documents: {successful}/{total_docs} successful")
        logger.info(f"[INGEST_FILES] Total chunks: {total_chunks}")
        logger.info(f"[INGEST_FILES] Duration: {duration:.2f}s")
        if total_errors > 0:
            logger.warning(f"[INGEST_FILES] Errors: {total_errors}")

        # Now insert the chunks into the main RAG system for KG extraction
        # Read back the chunks from MongoDB and insert them
        if successful > 0:
            logger.info("[INGEST_FILES] Inserting chunks into RAG for KG extraction...")
            chunks_col = db["ingested_chunks"]
            cursor = chunks_col.find({})
            chunks_data = await cursor.to_list(length=None)

            if chunks_data:
                # Extract content and file paths for RAG insertion
                contents = [c["content"] for c in chunks_data]
                file_paths = [c.get("metadata", {}).get("source", "unknown") for c in chunks_data]

                # Insert into main RAG (this builds the knowledge graph)
                await self.insert(
                    documents=contents,
                    file_paths=file_paths,
                )
                logger.info(f"[INGEST_FILES] Inserted {len(contents)} chunks into RAG system")

        # Close the motor client
        client.close()

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
        file_path: str | "Path",
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
        import tempfile
        import shutil

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

    async def query(
        self,
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        enable_rerank: bool | None = None,
        only_context: bool = False,
        system_prompt: str | None = None,
    ) -> str:
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

        Returns:
            Generated response or context string
        """
        rag = self._ensure_initialized()

        # Resolve parameters with defaults
        resolved_mode = mode or self.settings.default_query_mode
        resolved_top_k = top_k or self.settings.default_top_k
        resolved_rerank_top_k = rerank_top_k or self.settings.default_rerank_top_k
        resolved_enable_rerank = enable_rerank if enable_rerank is not None else self.settings.enable_rerank

        logger.info(f"[QUERY] ========== Starting Query ==========")
        logger.info(f"[QUERY] Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        logger.info(f"[QUERY] Mode: {resolved_mode}, top_k: {resolved_top_k}, rerank_top_k: {resolved_rerank_top_k}")
        logger.info(f"[QUERY] Rerank enabled: {resolved_enable_rerank}, only_context: {only_context}")

        # Apply implicit expansion if enabled
        expanded_query = query
        if self.settings.enable_implicit_expansion and hasattr(rag, 'entities_vdb'):
            expanded_query = await self._expand_query_with_entities(query, rag)
            if expanded_query != query:
                logger.info(f"[QUERY] Implicit expansion applied, query expanded")

        # Build query parameters with defaults from settings
        param = QueryParam(
            mode=resolved_mode,
            top_k=resolved_top_k,
            chunk_top_k=resolved_rerank_top_k,
            enable_rerank=resolved_enable_rerank,
            only_need_context=only_context,
        )

        try:
            logger.info("[QUERY] Executing query...")
            response = await rag.aquery(
                query=expanded_query,
                param=param._to_internal(),  # Convert to internal QueryParam
                system_prompt=system_prompt,
            )

            # Handle None response gracefully
            if response is None:
                logger.warning("[QUERY] RAG engine returned None response - possible LLM error or empty results")
                response = ""

            response_len = len(response)
            logger.info(f"[QUERY] Response received: {response_len} chars")
            logger.info(f"[QUERY] ========== Query Complete ==========")
            return response
        except Exception as e:
            logger.error(f"[QUERY] Error during query: {e}")
            raise

    async def query_with_sources(
        self,
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None,
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

        logger.info(f"[QUERY_WITH_SOURCES] Starting query with sources: mode={resolved_mode}")

        # Step 1: Get context (single retrieval)
        logger.info("[QUERY_WITH_SOURCES] Step 1: Fetching context...")
        context = await self.query(
            query=query,
            mode=mode,
            top_k=top_k,
            only_context=True,
        )
        context_len = len(context) if context else 0
        logger.info(f"[QUERY_WITH_SOURCES] Context retrieved: {context_len} chars")

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
        }

    async def query_with_memory(
        self,
        query: str,
        session_id: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None,
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
        history_context = await self._memory.get_context_string(session_id, max_history_messages)

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
            "llm_provider": self.settings.llm_provider,
            "llm_model": llm_model,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": embedding_model,
            "rerank_model": self.settings.voyage_rerank_model if self.settings.voyage_api_key else None,
            "enhancements": {
                "implicit_expansion": self.settings.enable_implicit_expansion,
                "entity_boosting": self.settings.enable_entity_boosting,
            },
        }


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

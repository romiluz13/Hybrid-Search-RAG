"""
Document Ingestion Pipeline for HybridRAG.

This module provides the full ingestion pipeline that:
1. Discovers documents in a folder
2. Processes documents to markdown (via DocumentProcessor)
3. Chunks documents (via DoclingHybridChunker)
4. Generates embeddings
5. Stores in MongoDB

This is the main entry point for batch document ingestion.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hybridrag.core.transaction_helper import run_with_transaction

from .chunker import create_chunker
from .document_processor import DocumentProcessor, create_document_processor
from .types import (
    DocumentChunk,
    IngestionConfig,
    IngestionResult,
    ProcessedDocument,
)

if TYPE_CHECKING:
    from pymongo.asynchronous.database import AsyncDatabase

logger = logging.getLogger("hybridrag.ingestion.pipeline")


class DocumentIngestionPipeline:
    """
    Full document ingestion pipeline for HybridRAG.

    This pipeline handles:
    - Multi-format document discovery
    - Document processing (conversion to markdown)
    - Structure-aware chunking (Docling HybridChunker)
    - Embedding generation
    - MongoDB storage

    Usage:
        ```python
        pipeline = DocumentIngestionPipeline(
            db=mongo_db,
            embedding_func=voyage_embed_func,
            config=IngestionConfig()
        )

        results = await pipeline.ingest_folder("./documents")
        ```
    """

    def __init__(
        self,
        db: AsyncDatabase,
        embedding_func: Callable[[list[str]], list[list[float]]],
        config: IngestionConfig | None = None,
        documents_collection: str = "documents",
        chunks_collection: str = "chunks",
        vector_embedding_backend: str = "client",
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            db: MongoDB async database instance.
            embedding_func: Function to generate embeddings for text.
            config: Ingestion configuration.
            documents_collection: Name of documents collection.
            chunks_collection: Name of chunks collection.
            vector_embedding_backend: ``"client"`` for client-side Voyage
                embeddings or ``"automated"`` for MongoDB Automated Embedding.
                When ``"automated"``, embeddings are generated server-side by
                MongoDB Atlas and the pipeline skips client-side generation.
        """
        self.db = db
        self.embedding_func = embedding_func
        self.config = config or IngestionConfig()
        self.documents_collection = documents_collection
        self.chunks_collection = chunks_collection
        self.vector_embedding_backend = vector_embedding_backend

        # Initialize components
        self.processor = create_document_processor(
            enable_audio=self.config.enable_audio_transcription
        )
        self.chunker = create_chunker(self.config.chunking)
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        """
        Create indexes on chunks collection for optimal query performance.

        Per MongoDB best practices:
        - Rule 1.3: Index everything in $lookup foreign fields
        - Rule 1.4: Index fields used in filtering/sorting

        These indexes ensure plug-and-play performance for any new clone.
        """
        if self._indexes_created:
            return

        chunks_col = self.db[self.chunks_collection]
        documents_col = self.db[self.documents_collection]

        try:
            # Chunks collection indexes
            # 1. document_id - for document lookups and joins
            await chunks_col.create_index("document_id")

            # 2. metadata.source - for filtering by source file
            await chunks_col.create_index("metadata.source")

            # 3. chunk_index - for ordering chunks within a document
            await chunks_col.create_index(
                [("document_id", 1), ("chunk_index", 1)],
            )

            # 4. created_at - for time-based queries (descending for recent first)
            await chunks_col.create_index(
                [("created_at", -1)],
            )

            # Documents collection indexes
            # 1. source - for filtering by source file
            await documents_col.create_index("source")

            # 2. created_at - for time-based queries
            await documents_col.create_index(
                [("created_at", -1)],
            )

            logger.debug(
                f"[PIPELINE] Created indexes on {self.chunks_collection} "
                f"and {self.documents_collection}"
            )
            self._indexes_created = True

        except Exception as e:
            # Log but don't fail - indexes are optimization, not required
            logger.warning(f"[PIPELINE] Error creating indexes: {e}")

    async def ingest_folder(
        self,
        folder_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest all supported documents from a folder.

        Args:
            folder_path: Path to the folder containing documents.
            progress_callback: Optional callback for progress updates (current, total).

        Returns:
            List of ingestion results for each document.
        """
        folder_path = Path(folder_path).resolve()

        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []

        # Ensure indexes exist for optimal performance (idempotent)
        await self._ensure_indexes()

        # Discover all supported files
        files = self._discover_files(folder_path)

        if not files:
            logger.warning(f"No supported files found in {folder_path}")
            return []

        logger.info(f"Found {len(files)} documents to process")

        # Clean existing data if requested
        if self.config.clean_before_ingest:
            await self._clean_collections()

        # Process each file
        results = []
        for i, file_path in enumerate(files):
            try:
                result = await self.ingest_file(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(files))

            except Exception as e:
                logger.exception(f"Failed to ingest {file_path}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=file_path.name,
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                        source=str(file_path),
                    )
                )

        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)
        successful = sum(1 for r in results if r.success)

        logger.info(
            f"Ingestion complete: {successful}/{len(results)} successful, "
            f"{total_chunks} total chunks, {total_errors} errors"
        )

        return results

    async def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            IngestionResult with details of the ingestion.
        """
        file_path = Path(file_path).resolve()
        start_time = datetime.now(UTC)
        errors = []

        try:
            # Step 1: Process document
            logger.info(f"Processing: {file_path.name}")
            processed = self.processor.process_file(file_path)

            # Step 2: Chunk document
            chunks = await self.chunker.chunk_document(
                content=processed.content,
                title=processed.title,
                source=processed.source,
                metadata=processed.metadata,
                docling_doc=processed.docling_document,
            )

            if not chunks:
                errors.append("No chunks created")
                return IngestionResult(
                    document_id="",
                    title=processed.title,
                    chunks_created=0,
                    processing_time_ms=self._calc_time_ms(start_time),
                    errors=errors,
                    source=str(file_path),
                    format_type=processed.format_type,
                )

            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Generate embeddings (skip if MongoDB Automated Embedding)
            if self.vector_embedding_backend == "client":
                chunks = await self._generate_embeddings(chunks)
                logger.info(f"Generated embeddings for {len(chunks)} chunks")
            else:
                logger.info(
                    "Skipping client-side embeddings "
                    "(MongoDB Automated Embedding enabled)"
                )

            # Step 4: Store in MongoDB
            document_id = await self._store_document(processed, chunks)
            logger.info(f"Stored document with ID: {document_id}")

            return IngestionResult(
                document_id=document_id,
                title=processed.title,
                chunks_created=len(chunks),
                processing_time_ms=self._calc_time_ms(start_time),
                errors=[],
                source=str(file_path),
                format_type=processed.format_type,
            )

        except Exception as e:
            logger.exception(f"Ingestion failed for {file_path}: {e}")
            return IngestionResult(
                document_id="",
                title=file_path.stem,
                chunks_created=0,
                processing_time_ms=self._calc_time_ms(start_time),
                errors=[str(e)],
                source=str(file_path),
            )

    async def ingest_text(
        self,
        content: str,
        title: str = "document",
        source: str = "text_input",
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """
        Ingest raw text content (no file processing).

        Args:
            content: Text content to ingest.
            title: Document title.
            source: Document source identifier.
            metadata: Additional metadata.

        Returns:
            IngestionResult with details of the ingestion.
        """
        start_time = datetime.now(UTC)

        try:
            # Create ProcessedDocument from text
            processed = ProcessedDocument(
                content=content,
                title=title,
                source=source,
                metadata=metadata or {},
                docling_document=None,  # No DoclingDocument for raw text
                format_type="text",
            )

            # Chunk using fallback (no DoclingDocument)
            chunks = await self.chunker.chunk_document(
                content=processed.content,
                title=processed.title,
                source=processed.source,
                metadata=processed.metadata,
                docling_doc=None,
            )

            if not chunks:
                return IngestionResult(
                    document_id="",
                    title=title,
                    chunks_created=0,
                    processing_time_ms=self._calc_time_ms(start_time),
                    errors=["No chunks created"],
                    source=source,
                    format_type="text",
                )

            # Generate embeddings and store (skip if MongoDB Automated Embedding)
            if self.vector_embedding_backend == "client":
                chunks = await self._generate_embeddings(chunks)
            else:
                logger.info(
                    "Skipping client-side embeddings "
                    "(MongoDB Automated Embedding enabled)"
                )
            document_id = await self._store_document(processed, chunks)

            return IngestionResult(
                document_id=document_id,
                title=title,
                chunks_created=len(chunks),
                processing_time_ms=self._calc_time_ms(start_time),
                errors=[],
                source=source,
                format_type="text",
            )

        except Exception as e:
            logger.exception(f"Text ingestion failed: {e}")
            return IngestionResult(
                document_id="",
                title=title,
                chunks_created=0,
                processing_time_ms=self._calc_time_ms(start_time),
                errors=[str(e)],
                source=source,
                format_type="text",
            )

    def _discover_files(self, folder_path: Path) -> list[Path]:
        """Discover all supported files in a folder recursively."""
        supported_extensions = DocumentProcessor.get_supported_extensions()

        files = []
        for ext in supported_extensions:
            pattern = f"**/*{ext}"
            files.extend(folder_path.glob(pattern))

        # Sort for consistent ordering
        return sorted(files)

    async def _generate_embeddings(
        self, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Generate embeddings for chunks in batches."""
        batch_size = self.config.batch_size
        contents = [chunk.content for chunk in chunks]

        all_embeddings = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i : i + batch_size]
            embeddings = await asyncio.to_thread(self.embedding_func, batch)
            all_embeddings.extend(embeddings)

        # Attach embeddings to chunks and validate dimension [L16]
        for chunk, embedding in zip(chunks, all_embeddings, strict=False):
            chunk.embedding = embedding

        # [L16] Validate embedding dimensions are consistent
        if all_embeddings:
            expected_dim = len(all_embeddings[0])
            for i, embedding in enumerate(all_embeddings):
                if len(embedding) != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch at chunk {i}: "
                        f"expected {expected_dim}, got {len(embedding)}"
                    )

        return chunks

    async def _store_document(
        self,
        processed: ProcessedDocument,
        chunks: list[DocumentChunk],
    ) -> str:
        """Store document and chunks in MongoDB atomically.

        Uses run_with_transaction to ensure parent document and chunks are
        inserted together. If chunk insertion fails, the parent document
        is rolled back -- no orphaned parents.

        Falls back to non-transactional execution on standalone/M0 deployments
        that do not support transactions.
        """
        documents_col = self.db[self.documents_collection]
        chunks_col = self.db[self.chunks_collection]

        # M30: Compute content hash for duplicate detection
        # [Rule: mongodb-schema-design] Prevent duplicate ingestion via content_hash
        content_hash = hashlib.sha256(processed.content.encode()).hexdigest()

        # Check for existing document with same content hash
        existing = await documents_col.find_one({"content_hash": content_hash})
        if existing:
            logger.info(
                f"[PIPELINE] Duplicate document detected: {content_hash[:12]}. "
                f"Returning existing ID: {existing['_id']}"
            )
            return str(existing["_id"])

        # Build document dict
        doc_dict = {
            "title": processed.title,
            "source": processed.source,
            "content": processed.content,
            "content_hash": content_hash,
            "metadata": processed.metadata,
            "format_type": processed.format_type,
            "created_at": datetime.now(UTC),
        }

        # Build chunk dicts (document_id assigned inside transaction)
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "content": chunk.content,
                "chunk_index": chunk.index,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
                "created_at": datetime.now(UTC),
            }
            # Only store the embedding field when using client-side embeddings.
            # With automated embedding, MongoDB generates vectors from
            # the content field at index time.
            if self.vector_embedding_backend == "client":
                chunk_dict["embedding"] = chunk.embedding
            chunk_dicts.append(chunk_dict)

        async def _do_insert(session=None):
            result = await documents_col.insert_one(doc_dict, session=session)
            document_id = result.inserted_id
            for chunk_dict in chunk_dicts:
                chunk_dict["document_id"] = document_id
            if chunk_dicts:
                await chunks_col.insert_many(
                    chunk_dicts, ordered=False, session=session
                )
            return str(document_id)

        client = self.db.client
        return await run_with_transaction(client, _do_insert)

    async def _clean_collections(self) -> None:
        """Clean existing data from collections atomically.

        M31: Uses run_with_transaction to ensure both collections are
        cleaned atomically. Falls back to non-transactional on standalone/M0.
        """
        logger.warning("Cleaning existing data from MongoDB...")

        documents_col = self.db[self.documents_collection]
        chunks_col = self.db[self.chunks_collection]

        async def _do_clean(session=None):
            chunks_result = await chunks_col.delete_many({}, session=session)
            logger.info(f"Deleted {chunks_result.deleted_count} chunks")

            docs_result = await documents_col.delete_many({}, session=session)
            logger.info(f"Deleted {docs_result.deleted_count} documents")

        client = self.db.client
        await run_with_transaction(client, _do_clean)

    def _calc_time_ms(self, start_time: datetime) -> float:
        """Calculate elapsed time in milliseconds."""
        return (datetime.now(UTC) - start_time).total_seconds() * 1000


def create_ingestion_pipeline(
    db: AsyncDatabase,
    embedding_func: Callable[[list[str]], list[list[float]]],
    config: IngestionConfig | None = None,
    vector_embedding_backend: str = "client",
) -> DocumentIngestionPipeline:
    """
    Create a document ingestion pipeline.

    Args:
        db: MongoDB async database instance.
        embedding_func: Function to generate embeddings.
        config: Ingestion configuration.
        vector_embedding_backend: ``"client"`` or ``"automated"``.

    Returns:
        DocumentIngestionPipeline instance.
    """
    return DocumentIngestionPipeline(
        db=db,
        embedding_func=embedding_func,
        config=config,
        vector_embedding_backend=vector_embedding_backend,
    )

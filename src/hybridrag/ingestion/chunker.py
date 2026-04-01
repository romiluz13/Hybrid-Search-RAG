"""
Docling HybridChunker implementation for intelligent document splitting.

This module uses Docling's built-in HybridChunker which combines:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)

Benefits over basic token-based chunking:
- Fast (no LLM API calls)
- Token-precise (not character-based estimates)
- Better for RAG (chunks include document context)
- Battle-tested (maintained by Docling team)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .types import ChunkingConfig, DocumentChunk

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

logger = logging.getLogger("hybridrag.ingestion.chunker")


class DoclingHybridChunker:
    """
    Docling HybridChunker wrapper for intelligent document splitting.

    This chunker uses Docling's built-in HybridChunker which:
    - Respects document structure (sections, paragraphs, tables)
    - Is token-aware (fits embedding model limits)
    - Preserves semantic coherence
    - Includes heading context in chunks

    Falls back to simple token-based chunking when:
    - DoclingDocument is not available
    - HybridChunker fails for any reason
    """

    def __init__(self, config: ChunkingConfig) -> None:
        """
        Initialize chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config
        self._tokenizer = None
        self._chunker = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of tokenizer and chunker."""
        if self._initialized:
            return

        try:
            from docling.chunking import HybridChunker
            from transformers import AutoTokenizer

            logger.info(f"Initializing tokenizer: {self.config.tokenizer_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_model)

            self._chunker = HybridChunker(
                tokenizer=self._tokenizer,
                max_tokens=self.config.max_tokens,
                merge_peers=self.config.merge_peers,
            )

            logger.info(
                f"HybridChunker initialized (max_tokens={self.config.max_tokens})"
            )
            self._initialized = True

        except ImportError as e:
            logger.warning(
                f"Docling/transformers not available, will use fallback chunking: {e}"
            )
            self._initialized = True  # Mark as initialized but without Docling

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        docling_doc: DoclingDocument | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a document using Docling's HybridChunker.

        Args:
            content: Document content (markdown format).
            title: Document title.
            source: Document source path or identifier.
            metadata: Additional metadata.
            docling_doc: Optional pre-converted DoclingDocument (for efficiency).

        Returns:
            List of document chunks with contextualized content.
        """
        if not content.strip():
            return []

        self._ensure_initialized()

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {}),
        }

        # If no DoclingDocument or chunker not available, use fallback
        if docling_doc is None or self._chunker is None:
            if docling_doc is None:
                logger.debug(
                    "No DoclingDocument provided, using simple chunking fallback"
                )
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            # Use HybridChunker to chunk the DoclingDocument
            chunk_iter = self._chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)

            # Convert Docling chunks to DocumentChunk objects
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = self._chunker.contextualize(chunk=chunk)

                # Count actual tokens
                token_count = len(self._tokenizer.encode(contextualized_text))

                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True,  # Flag indicating contextualized chunk
                }

                # Estimate character positions
                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(
                    DocumentChunk(
                        content=contextualized_text.strip(),
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                        token_count=token_count,
                    )
                )

                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self,
        content: str,
        base_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """
        Simple fallback chunking when HybridChunker can't be used.

        This is used when:
        - No DoclingDocument is provided
        - HybridChunker fails
        - Docling/transformers not installed

        Args:
            content: Content to chunk.
            base_metadata: Base metadata for chunks.

        Returns:
            List of document chunks.
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Try to use tokenizer if available for better token counting
        use_tokenizer = self._tokenizer is not None

        # Simple sliding window approach
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunk_text = content[start:]
            else:
                # Try to end at sentence boundary
                chunk_end = end
                for i in range(
                    end, max(start + self.config.min_chunk_size, end - 200), -1
                ):
                    if i < len(content) and content[i] in ".!?\n":
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                if use_tokenizer:
                    token_count = len(self._tokenizer.encode(chunk_text))
                else:
                    token_count = len(chunk_text) // 4  # Rough estimate

                chunks.append(
                    DocumentChunk(
                        content=chunk_text.strip(),
                        index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **base_metadata,
                            "chunk_method": "simple_fallback",
                            "total_chunks": -1,  # Will update after
                        },
                        token_count=token_count,
                    )
                )

                chunk_index += 1

            # Move forward with overlap
            start = end - overlap

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks

    def chunk_text_sync(
        self,
        content: str,
        title: str = "document",
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Synchronous version of chunk_document for compatibility.

        Uses only simple fallback chunking (no DoclingDocument support).

        Args:
            content: Content to chunk.
            title: Document title.
            source: Document source.
            metadata: Additional metadata.

        Returns:
            List of document chunks.
        """
        self._ensure_initialized()

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "simple_fallback",
            **(metadata or {}),
        }

        return self._simple_fallback_chunk(content, base_metadata)


def create_chunker(config: ChunkingConfig | None = None) -> DoclingHybridChunker:
    """
    Create a DoclingHybridChunker with the given configuration.

    Args:
        config: Chunking configuration. Uses defaults if not provided.

    Returns:
        DoclingHybridChunker instance.
    """
    if config is None:
        config = ChunkingConfig()
    return DoclingHybridChunker(config)

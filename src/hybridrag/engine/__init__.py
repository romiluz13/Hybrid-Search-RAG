"""
HybridRAG Engine - Core RAG functionality.

This module provides the core RAG engine with MongoDB Atlas storage support.
"""

from .base_engine import BaseRAGEngine
RAGEngine = BaseRAGEngine
from .base import QueryParam
from .base_engine import chunking_by_token_size, chunking_by_docling
from .base import EmbeddingFunc

__all__ = [
    "BaseRAGEngine",
    "RAGEngine",
    "QueryParam",
    "EmbeddingFunc",
    "chunking_by_token_size",
    "chunking_by_docling",
]

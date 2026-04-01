"""
HybridRAG Engine - Core RAG functionality.

This module provides the core RAG engine with MongoDB Atlas storage support.
"""

from .base import EmbeddingFunc, QueryParam
from .base_engine import BaseRAGEngine
from .operate import chunking_by_docling, chunking_by_token_size

RAGEngine = BaseRAGEngine

__all__ = [
    "BaseRAGEngine",
    "RAGEngine",
    "QueryParam",
    "EmbeddingFunc",
    "chunking_by_token_size",
    "chunking_by_docling",
]

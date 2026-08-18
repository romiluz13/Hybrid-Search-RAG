"""
HybridRAG - State-of-the-art RAG with MongoDB Atlas + Voyage AI.

This package provides a production-ready RAG system with:
- MongoDB Atlas storage (vector + graph + KV)
- Voyage AI embeddings (voyage-4-large) and reranking (rerank-2.5)
- Multi-provider LLM support (Gemini, Claude, OpenAI)
- Knowledge graph construction and querying
- Entity Boosting enhancement
- Implicit Expansion enhancement

Quick Start:
    ```python
    from hybridrag import create_hybridrag

    # Create and initialize
    rag = await create_hybridrag()

    # Ingest documents
    await rag.insert(["Your document text..."])

    # Query
    response = await rag.query("Your question?")
    ```

For API usage:
    ```bash
    uvicorn hybridrag.api.main:app --host 0.0.0.0 --port 8000
    ```
"""

# Core exports
from .config.settings import Settings, get_settings
from .core.rag import HybridRAG, create_hybridrag

# Enhancement exports
from .enhancements.entity_boosting import (
    EntityBoostingReranker,
    create_boosted_rerank_func,
)
from .enhancements.filters import (
    AtlasSearchFilterConfig,
    FilterConfig,
    FilterPredicate,
    FuzzyFilter,
    GeoFilter,
    LexicalPrefilterConfig,
    PhraseFilter,
    QueryStringFilter,
    RetrievalSecurityContext,
    TextFilter,
    VectorSearchFilterConfig,
    WildcardFilter,
    build_lexical_prefilters,
    build_search_vector_search_stage,
)
from .enhancements.implicit_expansion import ImplicitExpander

# Ingestion exports (document processing and chunking)
from .ingestion import (
    ChunkingConfig,
    DoclingHybridChunker,
    DocumentChunk,
    DocumentIngestionPipeline,
    DocumentProcessor,
    IngestionConfig,
    IngestionResult,
    ProcessedDocument,
)

# Observability exports (Langfuse)
from .integrations import (
    flush_langfuse,
    get_langfuse,
    langfuse_enabled,
    langfuse_status,
    log_ingestion,
    log_rag_query,
)

# Integration exports
from .integrations.voyage import (
    VoyageEmbedder,
    VoyageReranker,
    create_embedding_func,
    create_rerank_func,
)

# Memory exports (conversation session management)
from .memory import ConversationMemory, ConversationSession

# Evaluation exports (RAGAS)
try:
    from .evaluation import RAGEvaluator, run_evaluation

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    RAGEvaluator = None
    run_evaluation = None

# Prompts exports (system prompts, reranking, entity extraction)
# Query parameters for RAG operations
from .core.rag import QueryParam
from .prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    QueryType,
    create_system_prompt,
    detect_query_type,
    select_rerank_instruction,
)


# CLI exports (lazy import to avoid rich dependency)
def run_cli():
    """Run the HybridRAG CLI."""
    from .cli import run_cli as _run_cli

    return _run_cli()


__version__ = "0.4.0"
__all__ = [
    # Core
    "HybridRAG",
    "create_hybridrag",
    "QueryParam",
    # Config
    "Settings",
    "get_settings",
    # Integrations
    "VoyageEmbedder",
    "VoyageReranker",
    "create_embedding_func",
    "create_rerank_func",
    # Enhancements
    "EntityBoostingReranker",
    "create_boosted_rerank_func",
    "ImplicitExpander",
    # Filter Configs
    "FilterConfig",
    "FilterPredicate",
    "RetrievalSecurityContext",
    "LexicalPrefilterConfig",
    "VectorSearchFilterConfig",
    "AtlasSearchFilterConfig",
    # Filter Builders
    "build_lexical_prefilters",
    "build_search_vector_search_stage",
    # Filter Types
    "TextFilter",
    "FuzzyFilter",
    "PhraseFilter",
    "WildcardFilter",
    "GeoFilter",
    "QueryStringFilter",
    # Memory
    "ConversationMemory",
    "ConversationSession",
    # Ingestion (document processing)
    "DocumentIngestionPipeline",
    "DocumentProcessor",
    "DoclingHybridChunker",
    "IngestionConfig",
    "IngestionResult",
    "ChunkingConfig",
    "DocumentChunk",
    "ProcessedDocument",
    # Observability (Langfuse)
    "get_langfuse",
    "flush_langfuse",
    "langfuse_enabled",
    "langfuse_status",
    "log_rag_query",
    "log_ingestion",
    # Evaluation (RAGAS)
    "RAGEvaluator",
    "run_evaluation",
    # Prompts
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_COMPACT",
    "create_system_prompt",
    "detect_query_type",
    "select_rerank_instruction",
    "QueryType",
    # CLI
    "run_cli",
]

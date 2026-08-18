# Changelog

All notable changes to HybridRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-08-19

### Added
- Comprehensive cookbook documentation (8 recipes)
- Production deployment guide with scaling strategies
- Public metadata ingestion, typed filters, configured dual-index mappings, and server-owned retrieval constraints
- Native reranking with preserved fusion and rerank scores
- Exact vector execution, flat/HNSW index planning, quantization, explicit readiness waiting, and rollback
- Optional MongoDB Automated Embedding for chunk retrieval
- Redacted server explanations and multi-storage search-index inventory
- Explicit text-search index planning, application, and rollback
- Schema-validation desired-versus-effective diagnostics

### Changed

- Score fusion is now the default; rank fusion requires explicit selection.
- Native reranking is the default when reranking is enabled; external reranking requires explicit selection.
- Capability support is determined by execution, not numeric server-version gates.
- Low-level hybrid execution helpers are no longer exported from the package root; use `HybridRAG` query methods for the supported retrieval contract.

### Security

- Retrieval now fails closed instead of dropping filters or substituting rank, manual RRF, vector-only retrieval, or empty context.
- Query cache identity includes caller filters, server constraints, fusion strategy, vector mode, and native or external reranker identity.
- Constrained queries skip unscoped implicit entity expansion.
- Constrained KG retrieval fails closed until graph evidence carries matching provenance.
- HTTP diagnostics require configured operator authentication and sensitive filters are redacted from parameter logs.
- UUID and date filter values now use stable BSON representations.
- HTTP principals resolve to trusted mandatory tenant predicates, and streaming errors no longer expose backend exception text.

## [0.3.0] - 2025-01-20

### Added
- **MongoDB 8.2 Lexical Prefilters** - Full support for `$search.vectorSearch` with Atlas Search operators
  - `TextFilter` - Text matching with optional fuzzy
  - `FuzzyFilter` - Typo-tolerant search (maxEdits, prefixLength)
  - `PhraseFilter` - Exact phrase matching with slop
  - `WildcardFilter` - Pattern matching (`*`, `?`)
  - `GeoFilter` - Geospatial filtering (within, near, intersects)
  - `QueryStringFilter` - Full Lucene query syntax
- `LexicalPrefilterConfig` dataclass for configuring prefilters
- `build_lexical_prefilters()` - Build compound filter from config
- `build_search_vector_search_stage()` - Complete $search stage builder
- ADR-0006 documenting lexical prefilters architecture decision
- Voyage AI `voyage-4-large` embedding model support (1024 dimensions)
- Voyage AI `rerank-2.5` reranking model support

### Changed
- Default embedding model upgraded to `voyage-4-large`
- Default reranking model upgraded to `rerank-2.5`
- Improved numCandidates calculation (20x limit for optimal recall)

### Fixed
- Score field consistency across different MongoDB operators
- Fallback chain for clusters without $rankFusion support

## [0.2.0] - 2025-01-10

### Added
- **Native $rankFusion Support** - MongoDB 8.2 weighted hybrid search
  - `hybrid_search_with_rank_fusion()` - Primary hybrid search function
  - `vector_search_with_lexical_prefilters()` - Vector + prefilters
  - `manual_hybrid_search_with_rrf()` - Fallback RRF implementation
  - `reciprocal_rank_fusion()` - Score calculation utility
- **$scoreFusion Support** - Score-based fusion alternative
- `MongoDBHybridSearchConfig` - Comprehensive configuration dataclass
- `SearchResult` Pydantic model for type-safe results
- Three filter system architecture:
  - `VectorSearchFilterConfig` - MQL filters for $vectorSearch
  - `AtlasSearchFilterConfig` - Atlas Search filters
  - `LexicalPrefilterConfig` - Lexical prefilters (MongoDB 8.2+)
- ADR-0005 documenting filter builder systems
- Graceful fallback chain: $rankFusion -> manual RRF -> vector-only

### Changed
- Refactored hybrid search to use native MongoDB operators
- Improved error handling with detailed logging
- Better type hints throughout enhancements module

## [0.1.0] - 2025-01-01

### Added
- **Core HybridRAG Framework**
  - `HybridRAG` class - Main entry point
  - `create_hybridrag()` - Factory function
  - Async/await throughout
- **Conversation Memory**
  - `ConversationMemory` - MongoDB-backed session storage
  - `ConversationSession` - Session data model
  - Self-compaction via LLM summarization
  - Configurable history size and token limits
- **Document Ingestion**
  - PDF, Markdown, Text support
  - URL ingestion with Tavily
  - Automatic chunking with overlap
- **Knowledge Graph**
  - Entity extraction (SpaCy + LLM)
  - Relationship extraction
  - `$graphLookup` traversal
  - Entity boosting for reranking
- **Query Modes**
  - `mix` - Knowledge Graph + Vector + Keyword
  - `hybrid` - Vector + Keyword ($rankFusion)
  - `local` - Entity-focused retrieval
  - `global` - Community summaries
  - `naive` - Vector search only
- **Integrations**
  - Voyage AI embeddings (`voyage-3-large`)
  - Voyage AI reranking (`rerank-2`)
  - Anthropic Claude (claude-3-5-sonnet)
  - OpenAI GPT-4
  - Google Gemini
  - Langfuse observability
- **CLI Application**
  - `hybridrag chat` - Interactive chat
  - `hybridrag ingest` - Document ingestion
  - `hybridrag query` - Single query
  - `hybridrag status` - System status
  - `hybridrag benchmark` - Performance testing
- **Web Interfaces**
  - Chainlit UI
  - FastAPI REST API
- **Documentation**
  - ADR-0001: MongoDB Single Database
  - ADR-0002: Voyage AI Embeddings
  - ADR-0003: Hybrid Search RRF
  - ADR-0004: Prompts Module Architecture
  - Installation and configuration guides
  - Jupyter notebooks (5)
  - Example scripts (8)

### Architecture Decisions
- Single MongoDB database (no sync between multiple stores)
- Voyage AI as primary embedding provider
- Reciprocal Rank Fusion for hybrid search
- Modular prompts system
- Filter builder pattern for query construction

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.4.0 | 2026-08-19 | Latest-first fusion, native reranking, typed filters, index lifecycle, Automated Embedding |
| 0.3.0 | 2025-01-20 | MongoDB 8.2 Lexical Prefilters, voyage-4-large |
| 0.2.0 | 2025-01-10 | Native $rankFusion, Three Filter Systems |
| 0.1.0 | 2025-01-01 | Initial release, Core RAG framework |

---

[Unreleased]: https://github.com/romiluz13/Hybrid-Search-RAG/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/romiluz13/Hybrid-Search-RAG/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/romiluz13/Hybrid-Search-RAG/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/romiluz13/Hybrid-Search-RAG/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/romiluz13/Hybrid-Search-RAG/releases/tag/v0.1.0

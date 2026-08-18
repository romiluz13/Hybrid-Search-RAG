# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting key technical decisions for HybridRAG.

## What is an ADR?

An ADR captures an important architectural decision along with its context and consequences.

## Format

Each ADR follows this structure:
- **Status**: Accepted, Proposed, Deprecated, Superseded
- **Context**: What is the issue being decided?
- **Decision**: What is the change or decision?
- **Consequences**: What are the results (positive and negative)?

## ADRs

| ID | Title | Status | Date |
|----|-------|--------|------|
| [0001](0001-mongodb-single-database.md) | MongoDB Single Database Architecture | Accepted | 2026-01-20 |
| [0002](0002-voyage-ai-embeddings.md) | Voyage AI for Embeddings | Accepted | 2026-01-20 |
| [0003](0003-hybrid-search-rrf.md) | Hybrid Search with $rankFusion | Superseded by 0008 | 2026-01-20 |
| [0004](0004-prompts-module-architecture.md) | Prompts Module Architecture | Accepted | 2026-01-20 |
| [0005](0005-filter-builder-systems.md) | Dual Filter Builder Systems | Accepted | 2026-01-20 |
| [0008](0008-latest-first-search-capabilities.md) | Latest-First Search Capabilities | Accepted | 2026-08-18 |

## Contributing

When proposing a new ADR:
1. Copy the template
2. Fill in the sections
3. Assign the next sequential number
4. Submit for review
5. Update this README

# HybridRAG overview

HybridRAG is a backend-first Python library for retrieval-augmented generation (RAG) built on MongoDB Atlas. It combines vector search, full-text keyword search, and knowledge graph traversal in a single MongoDB database, avoiding the synchronization problems of multi-store architectures. The library targets developers building AI applications who want MongoDB-native search capabilities without managing separate vector databases, graph stores, or caching layers.

The project uses MongoDB 8.2+ features including `$rankFusion`, `$scoreFusion`, `$vectorSearch`, and `$search.vectorSearch` with lexical prefilters. Embeddings come from Voyage AI (voyage-4-large), and the LLM layer supports OpenAI, Anthropic Claude, Google Gemini, and OpenAI-compatible endpoints. The blessed reference path is intentionally narrow: one database, one embedding provider, one LLM interface, exercised by a deterministic release gate.

HybridRAG exposes four interfaces: a Python SDK (`HybridRAG` class), a FastAPI REST API, a Typer/Rich CLI, and a Chainlit chat UI. The core query flow supports six modes: `mix` (KG + vector/keyword fusion), `hybrid` (local + global KG), `local` (entity-focused), `global` (relationship-focused), `naive` (vector + keyword, no KG), and `bypass` (direct LLM).

## Quick links

| What | Where |
| ---- | ----- |
| Architecture diagram | [Architecture](architecture.md) |
| Setup and installation | [Getting started](getting-started.md) |
| Project terms | [Glossary](glossary.md) |
| Codebase statistics | [By the numbers](../by-the-numbers.md) |
| Project history | [Lore](../lore.md) |
| How to contribute | [Contributing](../how-to-contribute/index.md) |
| Hybrid search deep dive | [Hybrid search](../features/hybrid-search.md) |
| Filter systems | [Filters](../features/filters.md) |
| REST API reference | [API](../api/index.md) |
| Configuration reference | [Configuration](../reference/configuration.md) |

## Key source files

| File | Purpose |
| ---- | ------- |
| `src/hybridrag/core/rag.py` | Main `HybridRAG` class, public API, query orchestration |
| `src/hybridrag/enhancements/mongodb_hybrid_search.py` | `$rankFusion` and `$scoreFusion` pipeline builders |
| `src/hybridrag/engine/kg/mongo_impl.py` | MongoDB storage layer, vector/text/graph operations |
| `src/hybridrag/engine/operate.py` | Internal query execution pipeline |
| `src/hybridrag/config/settings.py` | Pydantic settings from environment variables |
| `src/hybridrag/api/main.py` | FastAPI application and route registration |
| `src/hybridrag/cli/app.py` | Typer CLI commands |

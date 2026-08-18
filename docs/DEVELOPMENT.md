# HybridRAG Development Guide

HybridRAG is a Python library that combines MongoDB search, knowledge-graph retrieval, Voyage embeddings/reranking, and pluggable LLM generation. The LightRAG-derived engine is bundled under `src/hybridrag/engine/`; there is no separate fork checkout in the current development workflow.

## Architecture

The public `HybridRAG` facade owns initialization, ingestion, query validation, conversation memory, diagnostics, and lifecycle operations. It delegates retrieval and generation to the bundled engine.

MongoDB provides the storage and retrieval layers:

- document/chunk, vector, graph, KV, and processing-status storage
- vector and lexical search with score fusion by default
- optional score fusion on supported MongoDB versions
- graph traversal and namespace isolation
- explicit search-index lifecycle operations

The enhancement layer contains public search/filter models, reranking, query optimization, entity boosting, and implicit expansion. Backend-specific MongoDB filter builders are implementation details behind the public `FilterConfig` contract.

## Source Layout

```text
src/hybridrag/
├── core/           # Public facade and MongoDB client lifecycle
├── engine/         # Bundled retrieval, KG, storage, prompt, and API engine
├── enhancements/   # Filters, fusion helpers, boosting, optimization
├── ingestion/      # Document, URL, chunking, and embedding pipelines
├── integrations/   # Voyage, LLM providers, Langfuse
├── memory/         # Conversation memory
├── migrations/     # Explicit schema/index migration helpers
├── prompts/        # Public prompt modules
├── api/            # Lean FastAPI surface
├── cli/            # Typer CLI
└── ui/             # Chainlit interface
```

## Development Commands

Use the project environment rather than a global Python installation.

```bash
# Focused tests
.venv/bin/pytest tests/path/to/test.py -q

# Fast project suite
make test-quick

# Full tests and lint
make test
make lint
```

Integration and benchmark tests are marked separately because some require MongoDB Atlas or other external services.

## Working Rules

- Add public behavior test-first at the highest stable seam.
- Preserve existing public parameter order, defaults, and return types in compatible releases.
- Keep Vector Search MQL, Atlas Search, and lexical-prefilter syntaxes separate behind public translators.
- Never trigger a production search-index rebuild implicitly; expose a plan and require explicit apply intent.
- Use timezone-aware datetimes.
- Keep credentials and `.env*` files out of commits.
- Match current code patterns before introducing new abstractions.

## Query Modes

| Mode | Retrieval behavior |
| --- | --- |
| `local` | Entity-focused KG retrieval |
| `global` | Relationship-focused KG retrieval |
| `hybrid` | Combined local and global KG retrieval |
| `mix` | KG retrieval plus vector/keyword chunk fusion |
| `naive` | Vector/keyword chunk fusion without KG context |
| `bypass` | Direct LLM generation without retrieval |

Public metadata filters are initially supported on `naive`, where both vector and lexical evidence can enforce the same boundary. KG-backed filtering remains research-gated until entity and relationship provenance can prevent excluded sources from influencing context.

## Testing Priorities

1. Public SDK behavior and backwards compatibility
2. MongoDB pipeline generation with mocked collections
3. Both HTTP request/response contracts
4. Index planning without implicit mutation
5. Conversation and source-reference behavior
6. Documentation examples that reflect real query modes and model defaults

## Documentation Ownership

Executable behavior and tests define query semantics. The README explains supported user flows. The capability specification records the accepted roadmap boundary. ADRs preserve decisions and explicit supersessions. Deployment documentation owns Atlas backup, monitoring, security, administration, and production index operations.

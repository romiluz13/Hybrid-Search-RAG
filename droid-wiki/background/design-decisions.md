# Design decisions

HybridRAG records major choices under `docs/adr/`. The ADRs explain why the code uses MongoDB for every persistence role, treats filter grammars separately, and prefers explicit capability errors over silent search fallbacks.

## Decision map

| ADR | Status | Decision |
| --- | --- | --- |
| `docs/adr/0001-mongodb-single-database.md` | Accepted | Use MongoDB for document, vector, graph, and status storage |
| `docs/adr/0002-voyage-ai-embeddings.md` | Accepted, model updated | Use Voyage AI; current default is `voyage-4-large` with 1,024 dimensions |
| `docs/adr/0003-hybrid-search-rrf.md` | Superseded | Originally selected rank fusion as the hybrid default |
| `docs/adr/0004-prompts-module-architecture.md` | Accepted | Centralize prompts under `src/hybridrag/prompts/` |
| `docs/adr/0005-filter-builder-systems.md` | Accepted | Keep vector and Atlas Search filter builders separate |
| `docs/adr/0006-lexical-prefilters.md` | Superseded | Introduced MongoDB 8.2 lexical prefilters |
| `docs/adr/0007-collection-naming-strategy.md` | Accepted direction | Prefer tenant fields and shared collections over collection-per-tenant proliferation |
| `docs/adr/0008-latest-first-search-capabilities.md` | Accepted | Use capability execution/probing, score fusion by default, and fail closed |
| `docs/adr/005-automated-embedding-consideration.md` | Superseded | Originally deferred Automated Embedding |

## Single-database architecture

MongoDB is the only database. `MongoKVStorage`, `MongoDocStatusStorage`, `MongoGraphStorage`, and `MongoVectorDBStorage` in `src/hybridrag/engine/kg/mongo_impl.py` cover documents, cache records, processing status, graph nodes and edges, and vectors.

This avoids cross-database synchronization and gives the engine one operational surface. The trade-off is tight coupling to MongoDB Search and graph aggregation features.

## MongoDB 8.2+ and the blessed stack

The project targets current MongoDB Search capabilities, including `$rankFusion`, score fusion, lexical prefilters, native reranking, and index diagnostics. `docs/adr/0008-latest-first-search-capabilities.md` replaces version-number feature guessing with probing or executing the requested operation.

A requested strategy must either run with its declared semantics or raise a stable capability or execution error. This is particularly important when filters represent tenant, ACL, or evidence boundaries.

The repository's release gates in `Makefile` and `.github/workflows/test.yml` validate the blessed local MongoDB stack and configured real providers. See [Testing](../how-to-contribute/testing.md).

## Filter translation

MongoDB's vector and Atlas Search stages use different filter languages:

```text
$vectorSearch filter       -> MQL operators: $eq, $gte, $in
$search compound.filter    -> Atlas operators: equals, range, text
```

ADR-0005 therefore created separate typed builders under `src/hybridrag/enhancements/filters/`. The backend-neutral `FilterConfig` is translated at the boundary, while `RetrievalSecurityContext` adds mandatory server-owned predicates. Mixing syntaxes is treated as a correctness error.

## Async-first design

Public query and ingestion operations are asynchronous because MongoDB access, embedding, reranking, and LLM generation are I/O-bound. The main facade in `src/hybridrag/core/rag.py` awaits the engine, and all four MongoDB storage implementations use PyMongo's asynchronous client types.

Synchronous wrappers exist for compatibility in parts of `src/hybridrag/engine/base_engine.py`, but new public behavior should preserve async APIs and avoid blocking provider or database calls.

## Embedding provider boundary

ADR-0002 standardizes on Voyage AI for client embeddings. Configuration still exposes model names and dimensions because the Atlas vector index must match the selected embedding output.

Automated Embedding was initially deferred by ADR-005 due to model pinning, batching, dual ingestion/KG paths, and cost visibility. ADR-0008 superseded the blanket deferral: it is now supported as a separate chunk path, while knowledge-graph embeddings remain a distinct design problem.

## Collection naming

The engine currently prefixes collections with `MONGODB_WORKSPACE`, as implemented by `resolve_workspace()` in `src/hybridrag/engine/kg/mongo_impl.py`. ADR-0007 recommends eventual consolidation into shared collections with an `engine_id` or tenant discriminator to reduce collection and index proliferation.

Because current code still uses prefixed collections, contributors should treat the ADR's consolidation map as migration direction. Do not assume it is already the deployed schema.

## Search-index lifecycle

Search indexes are planned, compared, applied, waited on, and rolled back through explicit methods in `src/hybridrag/engine/kg/mongo_impl.py`. Latest-first support does not permit automatic destructive rebuilding at startup. Operators can inspect readiness and run a functional sync probe after seeding.

See [Data models](../reference/data-models.md) for the index definitions and [Security](../security.md) for the fail-closed rule.

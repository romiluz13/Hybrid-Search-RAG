# HybridRAG Capability Catalog

The executable SDK and API contracts are authoritative. Capability support is detected by executing MongoDB operations; the library does not use numeric server-version policy gates.

## Supported Library Capabilities

### Retrieval

- Six query modes: `local`, `global`, `hybrid`, `mix`, `naive`, and `bypass`
- Score fusion by default; rank fusion only when explicitly selected
- Native MongoDB reranking by default when reranking is enabled
- Explicit external-provider reranking as an alternative
- Stable `fusion_score` and `rerank_score` fields
- ANN and exact vector execution
- Backend-neutral, typed metadata filters for `naive` retrieval
- Server-owned `RetrievalSecurityContext` predicates conjoined with caller filters
- Fail-closed rejection of KG retrieval when server-owned constraints cannot be preserved
- Fail-closed capability and execution errors; no semantic fallback

### Ingestion and indexes

- Validated per-document metadata through the SDK and both HTTP surfaces
- Metadata propagation to full documents and every generated chunk
- Configured vector-filter and Atlas Search mappings for the same metadata paths
- Client-generated vectors or MongoDB Automated Embedding for chunk retrieval
- A separate client-vector path for knowledge-graph entity and relationship vectors
- Flat and HNSW vector indexes, including HNSW options
- Scalar and binary quantization

### Diagnostics and operations

- Deterministic local query-plan compilation
- Redacted MongoDB server explanations
- Search-index inventory across chunk, entity, relationship, and graph stores
- Preserved upstream status and orthogonal readiness fields
- Explicit vector and text index plan, apply, observe, wait, and rollback operations
- Schema-validation creation/update parity for `error` and `errorAndLog`
- Desired-versus-effective schema diagnostics

### Interfaces

- Python SDK
- Lean and engine FastAPI surfaces with query, streaming, source, and data parity
- CLI and Ollama-compatible API
- Conversation memory, evaluation, tracing, and knowledge-graph visualization

## Constraints

- Caller filters are limited to `naive`; KG-backed filtering needs provenance-aware graph data before it can preserve the same evidence boundary.
- Filterable metadata paths must be configured before ingestion and indexed in both retrieval indexes.
- Search-index mutations are explicit operator actions. They are not triggered by startup or queries.
- Detailed HTTP diagnostics require a configured operator API key.
- Native capability failure is returned as a typed error. Rank, manual RRF, vector-only retrieval, and external reranking are not outage fallbacks.
- Query-plan and explanation APIs currently support `naive` only; `mix` diagnostics wait for a typed full KG call graph.

## Deployment Responsibilities

The application or platform owner controls credentials, network isolation, API authentication, backup, monitoring, capacity, model API keys, native-reranking enablement, index-change authorization, and live acceptance tests.

## Deliberately Deferred Data-Model Work

These require a concrete HybridRAG document-model use case and acceptance test:

- nested embeddings and `parentFilter`
- multiple vector fields
- indexes on transformed views
- `storedSource` optimization
- BSON binary vector storage
- arrays of embeddings

Preview or beta maturity alone is never a reason to exclude a capability.

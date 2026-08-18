# Latest-First Search Migration

This release changes retrieval semantics. Review ranking and deployment capabilities before upgrading.

## Required actions

1. Configure every filterable field in `FILTERABLE_METADATA_FIELDS`.
2. Inspect and explicitly apply all Search and Vector Search index plans. Startup no longer creates them.
3. Pass the returned applied plans to `wait_for_search_indexes(applied)` before serving traffic.
4. Enable MongoDB native reranking and its model credentials, or pass `rerank_strategy="external"` explicitly.
5. Run relevance evaluation because omitted `fusion_strategy` now means score fusion.
6. Warm new caches only after deployment; the retrieval cache namespace changed.
7. If HTTP authentication is enabled, configure the trusted API-key tenant map or validated JWT tenant claim.

## Behavior changes

- `fusion_strategy=None` selects score fusion. Use `fusion_strategy="rank"` for reciprocal-rank fusion.
- `enable_rerank=True` selects native reranking unless `rerank_strategy="external"` is passed.
- Failed requested capabilities raise typed errors. There is no rank, manual RRF, vector-only, external-reranker, or empty-context fallback.
- Filters accept only configured metadata paths and now constrain vector and lexical branches together.
- Filtered requests no longer perform unscoped implicit entity expansion.
- A configured server security context rejects KG-backed retrieval until graph evidence carries equivalent provenance.
- UUID filters use standard BSON binary UUIDs; date filters use UTC-midnight BSON datetimes.
- Query explanations execute on MongoDB and redact query vectors. Use `compile_query_plan()` for local compilation.
- HTTP explanation and index diagnostics require a configured `HYBRIDRAG_OPERATOR_API_KEY`.
- Metadata and filter values are checked against configured mapping types and backend bounds.

## Index changes

Use `plan_search_indexes()` before `apply_search_index_plans()`. This covers chunk vector/text, entity vector, relationship vector, and graph Search indexes. Keep the returned plans until readiness is confirmed; each contains the previous definition required by `rollback_search_index_plans()`.

Automated Embedding is opt-in through `VECTOR_EMBEDDING_BACKEND=automated`. It applies to chunk content only; entity and relationship indexes keep client-generated vectors.

Schema validation now targets the same workspace-prefixed runtime collection names as the engine. Run the migration with `--workspace` matching the serving process.

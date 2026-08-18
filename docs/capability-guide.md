# Search Capability Guide

## Metadata ingestion and filters

Configure every public or server-owned field before building the indexes:

```env
FILTERABLE_METADATA_FIELDS={"metadata.category":"token","metadata.year":"number","metadata.tenant_id":"token"}
```

Metadata enters through normal ingestion and is copied to every chunk:

```python
await rag.insert(
    ["First document", "Second document"],
    metadata=[
        {"category": "docs", "year": 2026},
        {"category": "guides", "year": 2025},
    ],
)
```

Use the same configured paths when querying:

```python
from hybridrag import FilterConfig, FilterPredicate

filters = FilterConfig(
    predicates=[
        FilterPredicate(field="metadata.category", operator="eq", value="docs"),
        FilterPredicate(field="metadata.year", operator="gte", value=2025),
    ]
)

answer = await rag.query(
    "What changed?",
    mode="naive",
    filter_config=filters,
)
```

Operators are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`, and `exists`. Values support strings, finite numbers, booleans, dates, timezone-aware datetimes, ObjectIds, and UUIDs. JSON clients use tagged values such as `{"type":"objectId","value":"..."}`, `{"type":"uuid","value":"..."}`, `{"type":"date","value":"2026-08-18"}`, and `{"type":"datetime","value":"2026-08-18T12:00:00Z"}`.

UUIDs are stored and queried as standard BSON binary UUIDs. Dates are normalized to UTC midnight so every accepted value has a stable BSON representation. Membership lists must use one BSON type. Values are checked against the configured mapping: range operators require number/date mappings, integers must fit BSON int64, and token values are limited to 8,181 UTF-8 bytes.

## Mandatory tenant or ACL constraints

Host applications inject server-owned predicates on the configured engine. They are not accepted from HTTP request bodies and are always conjoined with caller filters:

```python
from hybridrag import FilterConfig, FilterPredicate, HybridRAG, RetrievalSecurityContext

rag = HybridRAG(
    retrieval_security_context=RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field="metadata.tenant_id",
                    operator="eq",
                    value=authenticated_tenant,
                )
            ]
        )
    )
)
```

Constrained requests skip unscoped implicit entity expansion. The cache fingerprint includes the mandatory filter, caller filter, fusion strategy, vector mode, and native or external reranker identity.

For HTTP deployments, configure `HYBRIDRAG_TENANT_FIELD` and map trusted API keys with `HYBRIDRAG_API_KEY_TENANTS`, or select a validated JWT claim with `HYBRIDRAG_TENANT_CLAIM`. Request bodies never supply the mandatory tenant predicate.

Until graph entities and relationships carry filter provenance, a configured security context permits `naive` retrieval and `bypass` only. KG-backed retrieval fails with `RetrievalCapabilityError` instead of consulting unscoped evidence.

## Fusion, reranking, and vector execution

Score fusion is the default. Rank fusion remains an explicit choice:

```python
result = await rag.query_data(
    "Compare semantic and exact matches",
    mode="naive",
    fusion_strategy="rank",
)
```

When `enable_rerank=True`, `rerank_strategy="native"` is the default. MongoDB native reranking runs after fusion and returns both `fusion_score` and `rerank_score`. Select `rerank_strategy="external"` to use the configured provider. A native failure never switches strategies.

Use `vector_search_mode="exact"` for exact-nearest-neighbor evaluation; ANN is the normal default.

## Automated Embedding

Chunk retrieval can use MongoDB Automated Embedding while entity and relationship vectors remain on the client-generated path:

```env
VECTOR_EMBEDDING_BACKEND=automated
AUTOMATED_EMBEDDING_MODEL=voyage-4-large
```

The chunk index then uses `autoEmbed` on `content`, writes store text without invoking the client embedding provider, and queries send `query: {text: ...}`. Index plans accept current Automated Embedding options including dimensions, similarity, flat/HNSW, HNSW options, and `binaryNoRescore`. Capability or deployment configuration failures are explicit.

## Diagnostics

`compile_query_plan()` returns a deterministic local plan. `explain_query()` executes MongoDB explain and redacts query vectors. Both currently accept `mode="naive"` only.

```python
plan = await rag.compile_query_plan("How will this run?", mode="naive")
explanation = await rag.explain_query("How did this run?", mode="naive")
```

The HTTP diagnostic endpoints require a separate `HYBRIDRAG_OPERATOR_API_KEY` to be configured and supplied as `X-API-Key`. They return `503` when operator authentication has not been configured.

## Index lifecycle

For complete deployment reconciliation, operate on all five indexes together:

```python
plans = await rag.plan_search_indexes()
applied = await rag.apply_search_index_plans()
statuses = await rag.wait_for_search_indexes(
    applied
)

# Retain `applied` until deployment validation succeeds.
rolled_back = await rag.rollback_search_index_plans(applied)
```

Use the chunk-only methods for a targeted vector configuration change:

```python
plan = await rag.plan_vector_index(
    quantization="scalar",
    indexing_method="hnsw",
    hnsw_options={"maxEdges": 32, "numEdgeCandidates": 200},
    similarity="cosine",
)
applied = await rag.apply_vector_index_plan(
    quantization="scalar",
    indexing_method="hnsw",
    hnsw_options={"maxEdges": 32, "numEdgeCandidates": 200},
)
text_plan = await rag.plan_text_search_index()
text_applied = await rag.apply_text_search_index_plan()
status = await rag.wait_for_search_index(applied["index_name"])
rollback = await rag.rollback_vector_index(applied)
text_rollback = await rag.rollback_text_search_index(text_applied)
```

Startup and planning never mutate Search or Vector Search indexes. Apply and rollback are explicit. If a multi-index apply fails, `SearchIndexApplyError.applied_plans` retains accepted mutations and their rollback material. `list_search_indexes()` exposes `exists`, `fresh`, `transitioning`, `queryable`, `healthy`, `status_detail`, `main_index`, and `staged_index` separately. Pass applied plans to `wait_for_search_indexes()` so readiness is checked against the exact requested definition.

## Schema validation

The migration supports `error` and `errorAndLog` for both collection creation and update. It derives workspaced `full_docs`, `text_chunks`, `chunks`, and `doc_status` names from the runtime namespace rules, applies the requested policy, then compares desired and effective collection options. Unsupported capabilities and authorization failures raise `SchemaValidationError`.

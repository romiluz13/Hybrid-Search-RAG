# HybridRAG Latest-First Primary-Source Validation

**Date:** 2026-08-18

**Scope:** Validate `/tmp/hybridrag-codex-latest-first-handoff-2026-08-18.md` against the current working tree and current MongoDB primary documentation.

**Policy:** Latest-first. Preview or beta status is not an exclusion. Numeric deployment-version gates are not product policy. Unsupported requested capabilities must fail explicitly rather than change semantics.

## Executive verdict

The handoff is directionally correct, but it is no longer an accurate statement of the current tree. A substantial part of its primary-path remediation has already been implemented: filters are applied to both native fusion inputs, primary retrieval now fails closed, cache identity includes the effective filter boundary and major retrieval choices, constrained implicit expansion is disabled, public metadata reaches chunks, the engine API imports, and the `$scoreFusion` variable syntax is correct.

The release is still not ready. The most serious remaining problem is architectural: unsafe legacy retrieval functions remain publicly exported and documented. Those paths can discard filters, accept different security predicates for the two fusion inputs, substitute manual RRF or vector-only retrieval, accept a single surviving branch, or turn backend failures into an empty result. The primary `HybridRAG` path is hardened; the full public package surface is not.

Other release-blocking gaps are:

1. HTTP authentication is not connected to a per-request mandatory tenant/ACL retrieval context.
2. Schema validation targets generic collections rather than the workspaced collections used by the engine.
3. Automated Embedding is broken in normal public mix retrieval and uses a query shape contradicted by the current canonical field schema.
4. Multi-index apply can partially mutate indexes and then discard the rollback records needed for recovery.
5. Index drift/readiness comparison can treat behavior-changing scalar/flat/HNSW options as harmless server defaults.
6. Filter validation does not enforce configured mapping type/operator compatibility, while legacy filter builders contain fail-open and incorrect Boolean/geo translations.
7. Live Atlas release evidence is still missing for the newest native capabilities and lifecycle behaviors.

## Validation method

Five independent read-only reviewers covered:

- native score/rank fusion, exact/ANN search, native reranking, score preservation, and fallbacks;
- Vector Search definitions, Automated Embedding, quantization, flat/HNSW, PyMongo index APIs, and lifecycle state;
- public filters, BSON values, metadata ingestion, Atlas/MQL translation, and evidence-boundary equivalence;
- SDK/lean HTTP/engine HTTP contracts, authentication, cache isolation, diagnostics, and schema validation;
- the entire handoff phase-by-phase, including documentation, packaging, release evidence, and obsolete claims.

External claims were checked against current MongoDB documentation and the MongoDB Assistant knowledge base. Repository findings were checked against the current dirty working tree. No production file was changed during validation.

## Requirements matrix

| Handoff phase | Status | Current result |
|---|---|---|
| 0. Supersede the contract | Partial | ADR 0008 establishes latest-first policy and older material is partly marked superseded. Active API/cookbook/deployment/tracker material still contradicts it. |
| 1. Security and correctness | Incorrect across the complete public surface | The primary storage path is substantially fail-closed. Root-exported legacy enhancement paths are not. HTTP identity is not converted into mandatory per-request retrieval predicates. |
| 2. Metadata and filters | Partial | Public ingestion, chunk propagation, allowlisting, and matching vector/text paths exist. Mapping-type compatibility, backend bounds, status metadata preservation, and legacy builders remain defective. |
| 3. One retrieval contract | Partial | Option forwarding improved and both servers import. SDK, lean HTTP, and engine HTTP still own divergent models and behavior. |
| 4. Diagnostics and index lifecycle | Partial | Plan/apply/wait/rollback and operator diagnostics exist. Definition comparison, partial-apply recovery, main/staged generation modeling, and stable typed diagnostics remain incomplete. |
| 5. Latest retrieval capabilities | Partial/Incorrect | Native fusion, rerank, exact/ANN, quantization, and flat/HNSW are present. Automated Embedding is not operational through the public path and current tuning options are rejected. |
| 6. Schema validation | Incorrect target | Create/update policy and typed failures exist, but validators target collections the engine does not use. Only mocked proof exists. |
| 7. Documentation truth | Partial | Core policy/capability docs improved. Public legacy recipes, API references, changelog claims, trackers, and dependency declarations are stale. |
| 8. Release evidence | Partial | Focused offline, local, and artifact evidence exists. Required live native-capability, lifecycle, schema, recall, and exact-SHA CI evidence does not. |

## Release-blocking findings

### RB-1 — Public legacy retrieval paths still fail open

**Severity:** Critical

The package publicly exposes retrieval implementations whose semantics conflict with the adopted policy:

- `src/hybridrag/enhancements/mongodb_hybrid_search.py:257` accepts independent vector, Atlas, and lexical filter configurations. Vector and text fusion inputs can therefore operate over different evidence boundaries.
- `src/hybridrag/enhancements/mongodb_hybrid_search.py:917` retries text search without its supplied filter.
- `src/hybridrag/enhancements/mongodb_hybrid_search.py:1097` permits hybrid retrieval to degrade to one successful leg.
- `src/hybridrag/enhancements/mongodb_hybrid_search.py:1274` converts vector-search failures to `[]`.
- `src/hybridrag/enhancements/mongodb_hybrid_search.py:1424` drops lexical prefilter semantics by retrying ordinary vector search and can also return `[]`.
- `src/hybridrag/enhancements/mix_mode_search.py:190` changes native rank fusion into manual RRF.
- `src/hybridrag/enhancements/mix_mode_search.py:261` swallows graph failures.
- `src/hybridrag/enhancements/mix_mode_search.py:337` returns `[]` when all paths fail.
- `src/hybridrag/enhancements/__init__.py:41` and `src/hybridrag/__init__.py:58` keep legacy/manual helpers on public import surfaces.

MongoDB executes fusion input pipelines independently, so mandatory predicates must be equivalent in every input. A surviving unfiltered branch is not a valid partial success.

Primary sources:

- <https://www.mongodb.com/docs/manual/reference/operator/aggregation/scoreFusion/>
- <https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/>
- <https://www.mongodb.com/docs/search/query/operators-collectors/compound/>
- <https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/>

**Required resolution:** Make one canonical planner/executor the only supported public retrieval route. Until legacy APIs delegate to it with identical effective predicates and typed errors, stop exporting and documenting them as supported latest-first behavior. Do not replace one requested algorithm with another.

### RB-2 — HTTP authentication does not create a tenant/ACL evidence boundary

**Severity:** High

- `src/hybridrag/core/rag.py:381` holds a static in-process retrieval security context.
- The lean API factory at `src/hybridrag/core/rag.py:2710` does not accept a request-derived security context.
- `src/hybridrag/engine/api/utils_api.py:105` validates JWT metadata and discards it.
- `src/hybridrag/engine/api/rag_server.py:1050` constructs the server without a per-request `retrieval_security_context`.

Internal mandatory predicates are correctly ANDed with public filters once configured, but neither HTTP surface derives those predicates from the authenticated principal. Authentication and data authorization are therefore disconnected.

**Required resolution:** Introduce a trusted server-side principal-to-`RetrievalSecurityContext` resolver at the HTTP boundary. Never accept mandatory tenant/ACL predicates from the request body. Add warm-cache tenant A/B tests through both HTTP surfaces.

### RB-3 — Schema validation is applied to the wrong collections

**Severity:** High

- `src/hybridrag/migrations/migrate_schema_validation.py:126` targets `ingested_documents` and `ingested_chunks`.
- Runtime namespaces at `src/hybridrag/engine/namespace.py:6` use `full_docs`, `text_chunks`, `chunks`, entity, relationship, graph, and status collections.
- `src/hybridrag/engine/kg/mongo_impl.py:2429` applies workspace prefixes to runtime collections.
- `tests/test_schema_validation.py:17` uses mocks; it does not prove invalid writes against the actual runtime collections.

MongoDB validation is collection-specific. A successful migration against unused collection names does not protect HybridRAG data.

Primary sources:

- <https://www.mongodb.com/docs/manual/core/schema-validation/handle-invalid-documents/>
- <https://www.mongodb.com/docs/manual/reference/command/collMod/>

**Required resolution:** Generate validator targets from the same workspace/namespace mechanism as runtime storage. Prove create and `collMod` behavior with actual invalid writes and observable typed failure.

### RB-4 — Automated Embedding is not operational through normal retrieval

**Severity:** High

- `src/hybridrag/engine/operate.py:3656` always computes a client query vector for mix retrieval.
- `src/hybridrag/engine/operate.py:3709` passes that vector to storage.
- `src/hybridrag/engine/kg/mongo_impl.py:3251` rejects supplied query embeddings when Automated Embedding is enabled.
- `src/hybridrag/engine/kg/mongo_impl.py:3290` emits `"query": query`.
- `tests/core/test_public_retrieval_options.py:1033` asserts that string shape and therefore encodes the defect.
- `src/hybridrag/engine/kg/mongo_impl.py:2917` also invokes client embedding during every upsert and stores the client vector even in automated mode.

The current canonical `$vectorSearch` field schema and MongoDB MCP aggregation schema define Automated Embedding queries as:

```json
"query": { "text": "<query-text>" }
```

The MongoDB Automated Embedding overview currently contains a conflicting older example using a bare string. The canonical stage field table explicitly defines `query` as an object and `query.text` as the required string, so this validation adopts the field table/MCP schema and records the overview inconsistency rather than treating the string example as authoritative.

Primary sources:

- <https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/>
- <https://www.mongodb.com/docs/vector-search/crud-embeddings/automated-embedding/>

**Required resolution:** Branch orchestration before client embedding, emit `query.text`, and add a public-path integration test rather than testing storage in isolation. Decide explicitly whether any client vector remains necessary for a separate KG algorithm; do not make it an implicit Automated Embedding requirement.

### RB-5 — Multi-index apply can lose rollback material

**Severity:** High

- `src/hybridrag/core/rag.py:2239` applies multiple independent index operations sequentially.
- If a later operation fails, earlier mutations remain but the method raises without returning their applied records.
- `src/hybridrag/core/rag.py:2255` needs those records to restore prior definitions.

Search-index operations are asynchronous. Command acceptance is not build success, and a multi-index convenience method cannot promise recoverability while discarding partial progress.

Primary source: <https://www.mongodb.com/docs/languages/python/pymongo-driver/current/indexes/>

**Required resolution:** Return or attach a structured partial-apply result containing every accepted mutation and previous definition, or compensate safely. Preserve the exact recovery material even when a later command fails.

### RB-6 — Definition freshness can accept the wrong search behavior

**Severity:** High

- `src/hybridrag/engine/kg/mongo_impl.py:122` treats every extra observed definition key as a harmless server default.
- `src/hybridrag/engine/kg/mongo_impl.py:2600` rebuilds a wait-time desired definition without the options used by the applied plan.
- `src/hybridrag/engine/kg/mongo_impl.py:2646` can therefore mark an index fresh even when quantization, indexing method, or HNSW options differ.

Omitted client-vector quantization and indexing method have meaningful defaults. Scalar versus none and flat versus HNSW are behavioral differences, not ignorable server decoration.

Primary sources:

- <https://www.mongodb.com/docs/vector-search/vector-search-type/>
- <https://www.mongodb.com/docs/manual/reference/operator/aggregation/listSearchIndexes/>

**Required resolution:** Compare a normalized allowlist of behavior-defining fields and pass the applied plan's exact desired definition into readiness observation. Ignore only known server-owned fields.

## Additional high-value findings

### F-1 — Legacy filter builders can remove or broaden constraints

**Severity:** High

- `src/hybridrag/enhancements/filters/atlas_search_filters.py:107` silently skips an empty membership constraint.
- `src/hybridrag/enhancements/filters/lexical_prefilters.py:132` and related branches skip malformed or incomplete filters rather than rejecting them.
- `src/hybridrag/enhancements/filters/vector_search_filters.py:465` merges equality and membership on one field into a broader OR-like `$in`; the correct conjunction may match nothing.
- `src/hybridrag/enhancements/filters/lexical_prefilters.py:210` emits nonexistent `geoIntersects` and collapses `contains`/`disjoint` into `geoWithin`.

MongoDB Search geospatial relations use `geoShape` with an explicit `relation` value.

Primary source: <https://www.mongodb.com/docs/search/query/operators-collectors/geoShape/>

### F-2 — Main filter validation ignores configured mapping semantics

**Severity:** Medium

- `src/hybridrag/enhancements/filters/vector_search_filters.py:120` allows scalar/operator combinations without field-type knowledge.
- Its Atlas compiler maps comparisons to `range` without confirming that the configured mapping supports that operation.
- `src/hybridrag/engine/kg/mongo_impl.py:3237` validates the field allowlist but not operator/value compatibility.
- Heterogeneous `in` values are accepted even though Atlas Search requires one BSON type.

Examples currently accepted locally include Boolean or UUID range comparisons and mixed string/number/Boolean membership lists.

Primary sources:

- <https://www.mongodb.com/docs/search/query/operators-collectors/range/>
- <https://www.mongodb.com/docs/search/query/operators-collectors/in/>

### F-3 — Ingested metadata is not validated against mapping types and backend bounds

**Severity:** Medium

- `src/hybridrag/enhancements/filters/metadata.py:13` validates generic scalar shape without the configured field mapping.
- `src/hybridrag/config/settings.py:128` holds mapping types separately.
- Oversized token strings and integers outside BSON int64 can pass early validation and diverge or fail later.
- `src/hybridrag/engine/base_engine.py:1931` replaces document-status metadata with processing timestamps, so status responses lose caller metadata even though full documents and chunks retain it.

Primary sources:

- <https://www.mongodb.com/docs/search/field-types/token-type/>
- <https://www.mongodb.com/docs/search/define-field-mappings/>

### F-4 — Current Automated Embedding tuning is rejected

**Severity:** Medium

- `src/hybridrag/engine/kg/mongo_impl.py:2516` rejects quantization, indexing method, and HNSW options for `autoEmbed`.
- Current documentation permits Automated Embedding quantization, dimensions, similarity, flat/HNSW, and HNSW options, including `binaryNoRescore` where documented.
- `src/hybridrag/engine/kg/mongo_impl.py:2510` checks only positive HNSW integers rather than documented bounds (`maxEdges` 16–64 and `numEdgeCandidates` 100–3200).

Primary source: <https://www.mongodb.com/docs/vector-search/vector-search-type/>

### F-5 — Index lifecycle is preserved internally but flattened publicly

**Severity:** Medium

- `src/hybridrag/engine/kg/mongo_impl.py:2591` does not expose typed main/staged generations, definition versions, or status detail.
- `src/hybridrag/engine/kg/mongo_impl.py:2636` labels any `statusDetail` as a failure even though status detail is also present during normal builds.
- A `BUILDING` or `FAILED` index can still be queryable through an older main definition, while `STALE` is queryable. `queryable` and lifecycle status are independent.

Primary source: <https://www.mongodb.com/docs/manual/reference/operator/aggregation/listSearchIndexes/>

### F-6 — Fusion and rerank edge contracts remain inaccurate

**Severity:** Medium

- `src/hybridrag/engine/kg/mongo_impl.py:3450` and legacy equivalents classify every `OperationFailure` as a missing capability, including invalid syntax, authorization failures, and ordinary execution failures.
- `src/hybridrag/engine/kg/mongo_impl.py:3353` reads `scoreDetails.value`, although MongoDB states that `scoreDetails` format is not stable. Preserve the preceding score directly with `$meta: "score"` before reranking.
- `src/hybridrag/engine/kg/mongo_impl.py:3302` and legacy configuration can exceed the documented ANN `numCandidates` maximum.
- `src/hybridrag/enhancements/mongodb_hybrid_search.py:490` promises a weighted sum, but its `$scoreFusion` shape uses the default averaging method rather than an expression.

Primary sources:

- <https://www.mongodb.com/docs/manual/reference/operator/aggregation/scoreFusion/>
- <https://www.mongodb.com/docs/vector-search/query/aggregation-stages/rerank/>
- <https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/>
- <https://pymongo.readthedocs.io/en/stable/api/pymongo/errors.html#pymongo.errors.OperationFailure>

### F-7 — Retrieval contracts are forwarded, not unified

**Severity:** Medium

- Lean HTTP: `src/hybridrag/api/models.py:56`.
- Engine HTTP: `src/hybridrag/engine/api/routers/query_routes.py:42`.
- SDK signatures: `src/hybridrag/core/rag.py:1497`, `:1679`, `:1751`, and `:1871`.

These models differ in validation, defaults, token/history/prompt options, and response shapes. There is no one request model or parameterized suite proving the same retrieval semantics across SDK query/data/sources/stream and both HTTP APIs.

### F-8 — Error/cache details remain inconsistent

**Severity:** Medium

- `src/hybridrag/engine/api/routers/query_routes.py:778` streams `str(exception)` to clients, while lean streaming uses a generic message.
- `src/hybridrag/engine/operate.py:95` includes filters, mandatory security filters, fusion strategy, exact/ANN, and native rerank model in cache v3, but omits the external reranker provider/model/instruction identity.
- `src/hybridrag/api/models.py:159` declares a flat error schema while runtime retrieval errors use different status codes and a nested `detail` body.

The existing `test_query_cache_isolated_by_effective_filter` does exercise a warmed cache with different effective filters. That handoff requirement is no longer missing; it should be extended through the HTTP principal boundary once RB-2 is implemented.

### F-9 — Documentation and dependency truth remain incomplete

**Severity:** Medium

- `docs/enhanced-search.md:142` and cookbook material still advertise fallback-prone legacy paths.
- `docs/api.md:240` omits current filter/fusion/vector/rerank options.
- `CHANGELOG.md:27` overstates package-wide fail-closed behavior.
- Scratch issue trackers retain superseded defaults/gates while appearing complete.
- `requirements.txt` and `pyproject.toml` disagree on dependency baselines and optionality.
- `.pi/` and `.pi-subagents/` remain unignored runtime artifacts and must never be staged.

## Handoff claims that are now stale

The following original findings are fixed or obsolete in the current tree and must not be reimplemented:

- `$scoreFusion` variables now use `$$vectorPipeline` and `$$textPipeline`, matching the current official syntax.
- The primary `MongoVectorDBStorage.hybrid_query()` path applies one effective caller/security expression to both vector and text branches.
- Primary retrieval failures propagate typed errors rather than becoming empty context.
- Final-answer cache v3 includes public filters, mandatory filters, fusion strategy, vector mode, rerank strategy, and native model.
- Constrained retrieval skips implicit cross-boundary entity expansion.
- Public metadata ingestion reaches SDK, lean API, engine API, full documents, and chunks with cardinality checks.
- Engine API imports successfully; the earlier broken-relative-import blocker is fixed.
- Lean operator diagnostics require a separately configured key and use constant-time comparison.
- PyMongo exposes the required asynchronous Search-index management methods; lifecycle observation through `list_search_indexes()` is still necessary.
- Package smoke evidence now exists; the handoff's blanket “no package smoke” statement is obsolete.

## Validated-correct current behavior

- Current `$scoreFusion` variable syntax, stage placement, and primary input shapes are structurally correct.
- `$rankFusion` input pipelines and score-details placement are structurally correct.
- Exact vector search uses `exact: true` and omits `numCandidates`.
- Native `$rerank` placement, model selection, document cap, missing-content guard, and rerank-score extraction are structurally correct.
- Primary native hybrid retrieval fails explicitly rather than silently changing fusion strategy.
- Mandatory server predicates are ANDed with caller predicates and applied to both primary fusion inputs.
- UUID values use BSON subtype 4; ObjectId remains ObjectId; aware datetime handling is UTC-safe.
- Vector and text index definitions include the same configured metadata paths.
- Metadata reaches full documents and chunks.
- Engine API import is fixed.
- Operator diagnostic endpoints have a distinct authorization boundary.
- Individual Search-index create/update/drop rollback records are usable when no concurrent operator changes the same index.
- Online update is used instead of destructive drop/recreate on the active index-management path.
- Schema-validation actions and typed errors are implemented without numeric product gating; collection targeting and live proof remain unresolved.

## Forward remediation plan

### 1. Close every public evidence-boundary escape

Make the canonical primary planner/executor the only supported implementation. Route legacy exports through it or deprecate/remove them from supported public documentation. Reject independent vector/text security filters. Delete semantic fallback behavior only where it is part of the requested remediation; preserve unrelated dead code until its ownership is explicit.

Acceptance evidence:

- forced native fusion failure returns a typed error on every public import path;
- an identical effective mandatory predicate appears in every fusion input;
- empty membership means explicit rejection or match-none, never filter removal;
- no backend exception becomes `[]` or a single surviving hybrid leg.

### 2. Bind authenticated principals to mandatory retrieval constraints

Add a trusted request-scoped security-context resolver to both HTTP servers. Extend cache-isolation tests through authenticated HTTP requests. Keep public filters separate from server-owned predicates.

### 3. Repair schema target derivation

Use runtime workspace namespaces to enumerate validator targets. Add real collection create/update and invalid-write tests. Report desired versus effective validation action per actual collection.

### 4. Make Automated Embedding genuinely end-to-end

Skip client query embedding and client upsert embedding when server-managed embedding is selected, unless a separately modeled KG dependency requires a client vector. Emit `query.text`. Support the current documented tuning surface, validate HNSW bounds, and test through `HybridRAG.query(mode="mix")` against a capable deployment.

### 5. Make index actions recoverable and lifecycle-aware

Carry exact desired definitions through apply and wait. Normalize only documented server-owned fields. Return structured partial results on multi-index failure. Model `mainIndex` and `stagedIndex` separately, preserve `queryable` independently, and stop treating all `statusDetail` as a failure.

### 6. Enforce one typed metadata/filter contract

Bind allowed operators and BSON values to the configured mapping type. Enforce homogeneous membership lists, BSON numeric range, token/string bounds, and metadata type agreement during ingestion. Preserve user metadata in status records. Remove or repair legacy builder semantics.

### 7. Unify public query/error/cache models

Use one retrieval-options type across SDK and both APIs. Add a parameterized parity suite for query, data, sources, streaming, lean HTTP, and engine HTTP. Redact streaming failures. Include external reranker identity in cache fingerprints.

### 8. Produce live release evidence and reconcile docs

Run real native fusion/rerank, Automated Embedding, explain, lifecycle transition/rollback, invalid schema write, and exact-versus-ANN quantized recall tests. Record exact artifact/SHA evidence. Then update API/cookbook/changelog/tracker/dependency truth to match the tested public surface.

## Definition of done for this remediation

The release is ready only when:

- every supported public retrieval path preserves the same evidence boundary and selected semantics;
- authenticated HTTP principals produce mandatory server-owned predicates;
- actual workspaced runtime collections carry the intended validators;
- Automated Embedding works from public ingestion through public retrieval without hidden client embedding;
- multi-index failures retain recovery material and lifecycle state exposes serving versus staged generations;
- metadata values, filter operations, and index mappings share one typed contract;
- all SDK and HTTP surfaces pass one parity suite;
- live capable-deployment evidence proves native capabilities and failure paths;
- documentation and changelog claims describe only behavior that the tests prove.

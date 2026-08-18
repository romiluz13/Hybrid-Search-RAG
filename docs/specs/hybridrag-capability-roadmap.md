# HybridRAG Capability Roadmap — Accepted Specification

**Status:** Superseded by [ADR 0008](../adr/0008-latest-first-search-capabilities.md)
**Policy note:** Numeric version gates, rank-first defaults, and Preview exclusions below are retained only as historical context.
**Parent map:** <https://github.com/romiluz13/Hybrid-Search-RAG/issues/7>
**Evidence base:** capability war room, reviewed decision records for ownership, program boundary, and public compatibility

## Problem Statement

HybridRAG already contains MongoDB search, fusion, filtering, index-management, schema-validation, and observability building blocks, but important capabilities are unreachable through the main public API, duplicated across internal paths, or described incorrectly in the documentation. Users cannot reliably apply filters through `HybridRAG.query()`, select the implemented `$scoreFusion` path, inspect search pipelines and index readiness through stable APIs, or configure vector quantization without dropping below the library abstraction. At the same time, the README misstates query-mode behavior and contributor documentation describes an older architecture.

The capability investigation found 70 MongoDB-related candidates. Treating all 70 as a feature backlog would optimize for vendor breadth rather than user value. This specification therefore implements the eleven evidence-backed, reversible items and preserves the rest as research, monitor, reference, deployment-guidance, or rejected work.

## Solution

Deliver a backward-compatible v1 capability release with five outcomes:

1. A stable, backend-neutral filter model exposed through the primary SDK and supported end-to-end for the retrieval mode where semantics are unambiguous.
2. An explicit public fusion strategy that makes the existing `$scoreFusion` branch selectable without changing the `$rankFusion` default.
3. Structured diagnostics for search explanation and search-index readiness.
4. Explicit vector-quantization and schema-validation configuration with safe operational boundaries—no implicit production index rebuilds.
5. Documentation that accurately describes query modes, embedding models, architecture, hidden capabilities, deployment responsibilities, and feature maturity.

The release keeps MongoDB 8.2 as the baseline. MongoDB 8.3-only behavior is runtime-gated. Preview capabilities remain non-default and outside implementation scope.

## User Stories

1. As a HybridRAG user, I want to filter retrieval with one stable filter model so that I do not need to understand three incompatible MongoDB filter syntaxes.
2. As a HybridRAG user, I want invalid filter expressions rejected before a database query runs so that mistakes fail predictably.
3. As a HybridRAG user, I want filters to have explicitly documented mode support so that unfiltered KG evidence cannot silently bypass my constraints.
4. As a HybridRAG user, I want `bypass` and unsupported retrieval modes to reject filters so that options are never silently ignored.
5. As a HybridRAG user, I want ObjectId and UUID filter values supported so that metadata filters work with real MongoDB identifiers.
6. As a HybridRAG user, I want logical negation and the documented comparison operators translated consistently so that the public filter language is useful without exposing backend syntax.
7. As a HybridRAG user, I want current query calls to keep their positional behavior and defaults so that upgrading does not break my application.
8. As a HybridRAG user, I want `$rankFusion` to remain the default so that an upgrade does not change result ranking.
9. As a HybridRAG user, I want to opt into score fusion with `fusion_strategy="score"` so that I can evaluate the already-implemented score-based path.
10. As a HybridRAG user on MongoDB 8.2, I want a clear error when requesting an 8.3-only strategy so that unsupported behavior is not hidden.
11. As a HybridRAG user, I want invalid fusion/mode combinations rejected so that I know which retrieval path actually ran.
12. As a HybridRAG user, I want an explanation API with a stable structured result so that debugging does not alter the normal query return type.
13. As a HybridRAG operator, I want to inspect search-index readiness so that startup, migration, and incident tooling can wait for usable indexes.
14. As a HybridRAG operator, I want vector quantization represented in typed configuration so that supported indexes can reduce memory use without hand-editing definitions.
15. As a HybridRAG operator, I want the library to show the difference between desired and existing index definitions so that I understand whether a rebuild is required.
16. As a HybridRAG operator, I want production index rebuilds to require an explicit action so that configuration changes cannot trigger expensive mutations implicitly.
17. As a HybridRAG operator, I want schema validation to support `errorAndLog` so that validation policy can be tightened with observable failures.
18. As an HTTP API client, I want supported query fields available on both API surfaces with the same defaults and validation so that behavior does not depend on the server entry point.
19. As an HTTP API client, I want known new fields explicitly modeled so that Pydantic does not silently discard them.
20. As a CLI user, I want existing commands and flags to remain compatible so that the release does not force migration.
21. As a documentation reader, I want query modes described according to the production call graph so that I select `mix`, `naive`, or `hybrid` correctly.
22. As a documentation reader, I want the active Voyage model documented accurately so that configuration examples match runtime behavior.
23. As a contributor, I want current architecture documentation so that I do not design against an obsolete LightRAG-fork structure.
24. As a contributor, I want hidden capabilities discoverable from a maintained catalog so that implemented features are not repeatedly rediscovered.
25. As a deployment owner, I want backup, monitoring, security, and index-rebuild responsibilities separated from Python-library behavior so that operational authority is explicit.
26. As a maintainer, I want research-gated and Preview candidates excluded from this release so that the implementation stays bounded and evidence-driven.

## Implementation Decisions

### Program boundary

The implementation program contains eleven items: Explain, search-index status, `errorAndLog`, public filtering, vector-quantization configuration, missing filter operator/type support, public score-fusion selection, and four documentation corrections. Research-gated, monitor, reference, rejected, and deployment-guidance candidates remain outside implementation tickets.

### Backward compatibility

The release is additive. Existing parameters retain their names, order, positional availability, and defaults. New parameters are appended rather than inserted. Keyword-only enforcement, strict rejection of unknown HTTP fields, removals, renames, and a v2 query API require a future major-version decision and deprecation cycle.

The core SDK is the behavior authority. HTTP surfaces must explicitly model supported fields and use shared validation/translation. A public surface that accepts an option must forward it to the real retrieval path; otherwise it must explicitly reject or document the unsupported option.

### Public filter contract

Expose one backend-neutral `FilterConfig` expression model. Backend-specific Vector Search, Atlas Search, and lexical-prefilter forms are translation details and must not be mixed by callers.

Version one supports filters end-to-end on `naive` retrieval, where both vector and lexical branches operate over document chunks and can enforce the same evidence boundary. `local`, `global`, `hybrid`, and `mix` are rejected for filtered queries in v1 because graph/entity provenance is not yet rich enough to prove that excluded sources cannot influence generated context. `bypass` also rejects filters because it performs no retrieval.

Supporting filters on KG-backed modes requires a separate research-gated design that propagates source metadata through entity, relationship, graph traversal, and final chunk evidence. This safety boundary is preferable to apparently filtered results that still include unfiltered graph context.

The initial public filter language covers equality/inequality, ordered comparison, membership, logical conjunction/disjunction/negation, and supported scalar identifier values including ObjectId and UUID. Translators must fail when a public expression cannot be represented by the selected backend.

### Fusion contract

Expose `fusion_strategy` with public values `rank` and `score`; do not expose the internal negative boolean. Omission preserves the current rank-fusion default. Fusion selection is valid only on modes whose retrieval path actually invokes vector/lexical fusion. Invalid mode combinations fail deterministically. Score fusion requires MongoDB 8.3 or newer and fails clearly when unavailable.

Making score fusion selectable does not make it the default. Changing the default requires benchmark evidence and a separate compatibility decision.

### Diagnostics contract

Provide a separate structured explanation operation rather than an `explain` flag on `query()`. Normal query return types remain unchanged. Explanation output identifies the effective mode, fusion strategy, generated pipeline or explain document, relevant scores/score details when available, capability/version information, and timing needed for diagnosis.

Provide a search-index status operation that returns stable readiness information suitable for startup and migration workflows. These APIs are debugging and lifecycle primitives; they do not replace Langfuse or Atlas Monitoring.

### Index lifecycle and quantization

Expose typed vector-index options for quantization values supported by the deployed MongoDB environment. The library compares desired and observed definitions and reports whether an index rebuild is required. It never automatically rebuilds an existing production index because rebuilds have resource, storage, and rollback consequences.

Deployment explicitly authorizes apply/rebuild operations, monitors readiness, and owns rollback. Preview HNSW customization and `storedSource` remain research-gated and are not part of this configuration surface.

### Schema validation

Extend schema-validation operations to support `errorAndLog` where the server supports it. Existing policy remains unchanged unless the deployment explicitly selects the new action. Validation failures remain observable and testable.

### Release and maturity policy

MongoDB 8.2 remains the baseline. Features that require 8.3 or a particular Atlas tier use runtime capability checks and clear errors. Preview features are never defaults or acceptance dependencies in this release.

### Documentation truth

Executable behavior and tests are authoritative for query semantics. The README explains supported user flows and mode behavior; the capability catalog records the broader surface; ADRs preserve decisions with explicit supersession; contributor documentation describes the current architecture. Deployment guidance owns backup, monitoring, security, Atlas administration, and production index operations.

## Testing Decisions

- Test externally observable SDK and API behavior rather than private translator implementation.
- Preserve regression tests for every existing query mode with all new options omitted.
- Test public filter expressions against representative equality, range, membership, negation, ObjectId, and UUID cases.
- Verify `naive` filters constrain both vector and lexical branches and that unsupported modes fail before querying.
- Verify `fusion_strategy="rank"`, `fusion_strategy="score"`, omitted strategy, invalid modes, and unsupported server versions.
- Verify normal query return types are unchanged and explanation output uses its own stable structured result.
- Verify both HTTP API request models accept and forward supported fields with matching defaults.
- Verify search-index status distinguishes building, ready, failed, and missing states using mocked MongoDB responses.
- Verify index-definition diffing detects quantization changes without applying them implicitly.
- Verify explicit index lifecycle operations are idempotent and surface readiness/rollback information.
- Verify `errorAndLog` pipeline/config generation and unchanged default validation policy.
- Add documentation assertions or focused tests for the six canonical query-mode names and active embedding model where practical.
- Reuse existing mocked MongoDB enhancement tests, public RAG API tests, integration query tests, and CLI/API contract tests as prior art.
- Run the project’s full unit, lint, and type-check gates before review; integration tests requiring Atlas remain explicitly separated.

## Out of Scope

- Filters for `local`, `global`, `hybrid`, or `mix` until KG provenance can enforce the same evidence boundary.
- Changing `$scoreFusion` to the default.
- Native `$rerank`, custom HNSW options, flat indexes, vector arrays, Queryable Encryption text search, or other Preview/experimental features.
- `storedSource`, automated embeddings, BSON binary vectors, synonyms, highlighting, bulk-write optimization, extra Voyage models, Atlas Trigger deployment, or change-stream ingestion without their research gates.
- Automatic production index rebuilds.
- Consolidating the two HTTP API implementations.
- Changing Pydantic’s unknown-field policy from ignore to forbid.
- A v2 query API or keyword-only conversion.
- Implementing Atlas backup, monitoring, security, administration, or capacity management inside the Python library.
- Re-platforming onto EOL Atlas App Services or Device Sync.

## Further Notes

The reviewed war-room corpus remains the evidence appendix at `/tmp/hybridrag-capability-war-room-20260812/`. The parent Wayfinder map should be treated as superseded by this specification for implementation decisions; unresolved map issues become historical pointers rather than build blockers.

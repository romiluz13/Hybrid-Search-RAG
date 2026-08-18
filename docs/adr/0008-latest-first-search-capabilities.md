# ADR-0008: Latest-First Search Capabilities

## Status

Accepted

## Context

HybridRAG's retrieval behavior is described by several documents that disagree
about defaults, capability maturity, and fallback behavior. Some paths also
change retrieval semantics or remove filters after a backend failure. That is
unsafe when filters represent tenant, ACL, or other evidence constraints.

HybridRAG is intentionally an innovation-first library. Upstream preview or
beta labels describe maturity, but do not by themselves exclude a useful
capability from the supported product surface.

## Decision

HybridRAG targets the newest configured MongoDB Search capabilities and checks
support by probing or executing the requested operation rather than deriving
product behavior from a numeric server version.

- Score fusion is the default vector-and-lexical fusion strategy.
- Rank fusion remains an explicit caller-selected strategy.
- Native reranking is a supported retrieval stage.
- A requested capability either executes with its declared semantics or raises
  a stable typed capability or execution error.
- Retrieval never silently substitutes another fusion, reranking, or search
  strategy.
- Public filters and server-owned tenant or ACL predicates are mandatory
  evidence constraints. Every retrieval, expansion, cache, and generation path
  must preserve them or fail closed.
- Filterable metadata must be accepted through public ingestion, propagated to
  chunks, stored in a stable BSON form, and mapped in every relevant index.
- Search-index changes require an explicit operator action. Latest-first does
  not authorize startup-time creation or implicit destructive rebuilds.
- Raw backend diagnostics may be retained as opaque data, while public models
  expose stable fields and errors.
- Detailed HTTP diagnostics require configured operator authentication and
  redact query vectors.

## Consequences

**Positive:** Callers can rely on declared retrieval semantics, security
constraints remain intact during failures, and new MongoDB Search capabilities
can become normal product behavior without compatibility matrices.

**Negative:** Deployments that do not enable a requested capability receive an
explicit error instead of a degraded result. Changing the default fusion
strategy can change ranking and requires release notes and relevance evidence.
KG-backed retrieval is unavailable under mandatory predicates until graph
evidence carries equivalent provenance; failing closed is preferred to an
unscoped partial answer.

## Supersedes

- ADR-0003 as the default-fusion decision. Its historical rank-fusion rationale
  remains valid for callers that select rank fusion explicitly.
- ADR-005 as a blanket deferral of Automated Embedding. Automated Embedding is
  now an active implementation track, with chunk and knowledge-graph embedding
  paths designed separately.
- Compatibility-first and preview-exclusion statements in the historical
  capability roadmap and release-hardening handoff.

## Date

2026-08-18

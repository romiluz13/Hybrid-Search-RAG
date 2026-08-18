# ADR-0002: Voyage AI for Embeddings

## Status

Accepted; model selection updated in August 2026

## Context

HybridRAG needs high-quality embeddings for semantic retrieval. The original decision selected Voyage AI `voyage-3-large` over OpenAI and Cohere. The production configuration later moved to the Voyage 4 family while preserving the same provider boundary.

## Decision

Use Voyage AI as the embedding provider. The current default is `voyage-4-large` with 1024 dimensions. Keep the model name and dimensions configurable so deployments can adopt supported Voyage models without changing the provider interface.

This update supersedes only the original model identifier; the provider decision remains accepted.

## Consequences

**Positive:** Strong retrieval quality, explicit input types, and a stable provider integration shared by ingestion and query paths.

**Negative:** Vendor dependency, model-specific cost/latency trade-offs, and a requirement to keep index dimensions aligned with the configured model.

## Date

2026-01-20; model update recorded 2026-08-13

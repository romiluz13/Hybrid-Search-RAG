# ADR-005: Server-Side Automated Embedding

## Status

Deferred - Considered but not adopted at this time.

## Context

MongoDB Atlas provides [Automated Embedding](https://www.mongodb.com/docs/atlas/atlas-vector-search/automated-embedding/)
via the `index-automated-embedding` rule, which generates embeddings server-side
using Voyage AI integration. This eliminates the need for client-side embedding
computation during ingestion.

Currently, HybridRAG generates embeddings client-side using:
- `VoyageEmbedder` for ingestion pipeline (`pipeline_embed_func`)
- `EmbeddingFunc` in the RAG engine for vector storage upserts

## Decision

We defer adoption of automated embedding for now because:

1. **Model version pinning**: Client-side embedding gives us explicit control over
   which Voyage model version is used. Automated embedding may auto-upgrade.

2. **Batch control**: We need fine-grained control over batching (64 per batch)
   to avoid Voyage API token limits per request.

3. **Dual-path embedding**: HybridRAG embeds at ingestion (pipeline) AND at
   KG extraction (RAG engine). Automated embedding covers only one path.

4. **Cost visibility**: Client-side calls provide clear API usage tracking
   through Langfuse observability.

## Consequences

- We continue to manage embedding generation client-side.
- We accept the latency overhead of client-side API calls.
- We revisit this decision when MongoDB supports Voyage model version pinning
  and batch size configuration in automated embedding.

## References

- MongoDB Skill Rule: `index-automated-embedding`
- HybridRAG embedding flow: `src/hybridrag/core/rag.py` (lines 74-93)
- Voyage AI batching: `src/hybridrag/integrations/voyage.py`

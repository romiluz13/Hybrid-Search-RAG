# ADR-005: Server-Side Automated Embedding

## Status

Superseded by ADR-0008. The constraints below remain design inputs for the
separate chunk and knowledge-graph embedding paths.

**Update (2026-09-01):** ADR-0008's "active implementation track" is now the
default. The ingestion pipeline gap (reason 3 below — "Dual-path embedding")
has been closed: `DocumentIngestionPipeline` skips client-side embedding
generation when `vector_embedding_backend="automated"`. The chunk embedding
path now defaults to `"automated"` in `settings.py` and `.env.example`. The
knowledge-graph embedding path remains client-side by design (ADR-0008).
Reasons 1, 2, and 4 remain as noted trade-offs.

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

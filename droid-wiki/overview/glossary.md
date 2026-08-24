# Glossary

| Term | Definition |
| ---- | ---------- |
| **Blessed stack** | The single supported reference path: MongoDB + Voyage AI + OpenAI-compatible LLM. Other integrations exist but are not part of the release gate. |
| **$rankFusion** | MongoDB 8.0+ aggregation stage that combines results from multiple input pipelines using Reciprocal Rank Fusion (RRF). Formula: `score = Σ (weight / (60 + rank))`. |
| **$scoreFusion** | MongoDB 8.3+ aggregation stage that combines results using weighted score normalization (sigmoid or min-max). |
| **$vectorSearch** | MongoDB Atlas Vector Search aggregation stage for approximate nearest neighbor (ANN) search over vector embeddings. |
| **$search.vectorSearch** | Atlas Search variant of vector search that supports lexical prefilters (fuzzy, phrase, wildcard, geo) before vector similarity. |
| **Lexical prefilters** | Atlas Search operators applied before vector search: fuzzy matching, phrase matching, wildcard patterns, geo filtering, and range filters. |
| **numCandidates** | `$vectorSearch` parameter controlling how many nearest neighbors to approximate. MongoDB recommends 20x the limit. Must be >= limit, max 1000. |
| **RRF** | Reciprocal Rank Fusion. A rank-based fusion method that combines ranked result lists by summing `1 / (constant + rank)` for each list. |
| **Mix mode** | Query mode that combines knowledge graph retrieval with vector/keyword fusion. The recommended default mode. |
| **Naive mode** | Query mode using vector + keyword fusion without knowledge graph involvement. Used for filtered document retrieval. |
| **Bypass mode** | Query mode that skips retrieval entirely and sends the query directly to the LLM. |
| **Entity boosting** | Enhancement that uses knowledge graph entity relationships to boost reranking scores for results connected to query-relevant entities. |
| **Implicit expansion** | Enhancement that semantically expands queries by discovering related entities through embedding similarity. |
| **Self-compacting memory** | Conversation memory system that auto-summarizes older turns to stay within token limits while preserving context. |
| **Search index probe** | Functional test that fires minimal `$vectorSearch` and `$search` queries to verify Atlas Search indexes have ingested recently seeded documents, accounting for eventual consistency. |
| **Workspace** | MongoDB collection name prefix used for multi-tenant isolation. Collections are namespaced as `{workspace}_kg_edges`, `{workspace}_chunks`, etc. |
| **FilterConfig** | Public, backend-neutral filter specification that translates to MongoDB-specific syntax (MQL, Atlas Search compound, or lexical prefilters). |
| **scoreDetails** | MongoDB metadata field exposed by `$rankFusion` and `$scoreFusion` that contains per-pipeline scores for debugging. |
| **atlas-local** | Docker image (`mongodb/mongodb-atlas-local:preview`) providing local MongoDB with Search and Vector Search for development. |
| **Grove** | MongoDB internal OpenAI-compatible LLM gateway. Used by solutions architects without direct OpenAI/Anthropic keys. |
| **BSON** | Binary JSON format used by MongoDB. Extended JSON defines canonical representations for BSON types like ObjectId, Decimal128, and Timestamp. |
| **Embedding dimension** | The size of the vector produced by an embedding model. Voyage voyage-4-large produces 1024-dimensional vectors. |
| **Over-fetch** | Technique where each fusion input branch fetches more candidates than the final top_k, giving the fusion stage more material to rank. Computed as `max(top_k * factor, floor)`. |

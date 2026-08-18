# Atlas Live Capability Validation

Date: 2026-08-18

This record closes the provider-dependent validation gaps from the latest-first release hardening work. The test used an isolated, temporary database on MongoDB Atlas. It did not modify existing application databases, and the database was dropped after the run.

No connection string, credential, username, hostname, or cluster identifier is stored in this repository.

## Validated capabilities

| Capability | Result | Evidence |
| --- | --- | --- |
| Search-index lifecycle | Passed | Vector and text indexes reached `READY` with `queryable: true`. |
| `$scoreFusion` | Passed | Weighted sigmoid-normalized vector and lexical fusion returned three documents with numeric scores; the expected hybrid document ranked first. |
| Native `$rerank` | Passed | `rerank-2.5` returned three documents; both `fusion_score` and `rerank_score` remained numeric for every result. |
| Automated Embedding index | Passed | An `autoEmbed` index using `voyage-4-large` reached `READY` and was queryable. |
| Automated Embedding query | Passed | `$vectorSearch` with `query: {text: ...}` returned two documents with numeric vector-search scores. |
| Cleanup | Passed | The isolated validation database was dropped successfully. |

## Validation shape

- Client vector index: eight dimensions, cosine similarity, HNSW, no quantization.
- Text index: static `content` string mapping.
- Fusion: vector weight `0.6`, lexical weight `0.4`, sigmoid normalization, expression combination.
- Reranking: native `$rerank` after fusion, preserving the pre-rerank score before capturing the rerank score.
- Automated Embedding: source text stored in `content`; no client-generated vector was written.

The run exercised the same stage shapes and score-preservation contract used by HybridRAG. Deployment support was determined by executing each capability rather than by a numeric version gate.

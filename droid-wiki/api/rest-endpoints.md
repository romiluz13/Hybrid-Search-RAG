# REST endpoints

This page catalogs routes from the reference application in `src/hybridrag/api/main.py` and the full engine server under `src/hybridrag/engine/api/`. Paths shown for engine document and Ollama routes include their mounted prefixes.

## Reference API

This is the application started by `make run-api`.

| Method | Path | Purpose | Access |
| --- | --- | --- | --- |
| GET | `/health` | Report API, RAG, and MongoDB health plus package version | Public |
| GET | `/ready` | Kubernetes-style readiness result | Public |
| POST | `/v1/ingest` | Chunk, embed, extract graph data, and store documents | Request guard |
| POST | `/v1/query` | Return a non-streaming RAG answer, optionally with context and references | Request guard |
| POST | `/v1/query/stream` | Return metadata followed by answer chunks as NDJSON | Request guard |
| POST | `/v1/query/explain` | Run a redacted MongoDB explanation for the effective query pipeline | Operator key |
| GET | `/v1/search-indexes` | List stable vector and text search-index status records | Operator key |
| GET | `/v1/search-indexes/sync` | Probe whether newly written data is visible to vector and text search | Operator key |
| DELETE | `/v1/documents/{doc_id}` | Delete one document after ObjectId and ownership checks | Request guard |
| GET | `/v1/status` | Return RAG initialization and configuration status | Public |

`POST /v1/query` supports `local`, `global`, `hybrid`, `naive`, `mix`, and `bypass` modes. Streaming callers must use `/v1/query/stream`; the normal endpoint rejects `stream=true`.

The request guard in `src/hybridrag/api/main.py` applies optional public API-key authentication, per-process rate limiting, and request-scoped tenant constraints. The operator guard requires a separately configured key and uses constant-time comparison.

## Engine server shell

`src/hybridrag/engine/api/rag_server.py` creates the full engine application and mounts the routers below.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Redirect to the Web UI when available, otherwise API docs |
| GET | `/docs` | Serve custom Swagger UI |
| GET | `/docs/oauth2-redirect` | Complete Swagger OAuth2 redirection |
| GET | `/auth-status` | Report auth status and return a guest token when auth is disabled |
| POST | `/login` | Authenticate a configured account and issue a token |
| GET | `/health` | Report server and Web UI status |
| GET | `/webui` | Redirect to docs when the Web UI is unavailable |
| GET | `/webui/` | Slash-preserving form of `/webui` |

## Query router

Routes are defined in `src/hybridrag/engine/api/routers/query_routes.py`.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/query` | Execute a comprehensive non-streaming RAG query; a request `stream` flag is ignored |
| POST | `/query/stream` | Stream an answer with selectable streaming behavior |
| POST | `/query/data` | Return structured retrieval data for analysis rather than only generated prose |

## Document router

`src/hybridrag/engine/api/routers/document_routes.py` declares a `/documents` prefix.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/documents/scan` | Scan the configured input directory and enqueue new documents |
| POST | `/documents/upload` | Upload a file and index it |
| POST | `/documents/text` | Insert one text document |
| POST | `/documents/texts` | Insert multiple text documents |
| DELETE | `/documents` | Clear all documents; forbidden for scoped principals |
| GET | `/documents/pipeline_status` | Return current indexing-pipeline state and history |
| GET | `/documents` | List document processing statuses |
| DELETE | `/documents/delete_document` | Delete selected document IDs in the background |
| POST | `/documents/clear_cache` | Clear LLM response cache data |
| DELETE | `/documents/delete_entity` | Delete an entity and its graph relationships |
| DELETE | `/documents/delete_relation` | Delete a relationship between two entities |
| GET | `/documents/track_status/{track_id}` | Return statuses for an ingestion tracking ID |
| POST | `/documents/paginated` | Return paginated, filtered, and sorted document statuses |
| GET | `/documents/status_counts` | Count documents by processing status |
| POST | `/documents/reprocess_failed` | Re-enqueue failed and pending documents |
| POST | `/documents/cancel_pipeline` | Request cancellation of the active indexing pipeline |

## Graph router

Routes are defined in `src/hybridrag/engine/api/routers/graph_routes.py`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/graph/label/list` | List graph labels |
| GET | `/graph/label/popular` | Rank labels by node degree |
| GET | `/graph/label/search` | Fuzzy-search labels |
| GET | `/graphs` | Return a bounded connected subgraph for a label |
| GET | `/graph/entity/exists` | Check whether an entity exists |
| POST | `/graph/entity/edit` | Update an entity |
| POST | `/graph/relation/edit` | Update a relation |
| POST | `/graph/entity/create` | Create an entity |
| POST | `/graph/relation/create` | Create a relation between entities |
| POST | `/graph/entities/merge` | Merge entities while preserving relationships |

## Ollama compatibility router

`src/hybridrag/engine/api/routers/ollama_api.py` is mounted at `/api`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/version` | Return the emulated Ollama API version |
| GET | `/api/tags` | List the emulated HybridRAG model |
| GET | `/api/ps` | List the currently exposed model as running |
| POST | `/api/generate` | Handle Ollama-compatible generation requests |
| POST | `/api/chat` | Handle Ollama-compatible chat and route eligible user messages through RAG |

All engine-router routes use the combined auth dependency. A configured tenant field disables unauthenticated whitelist bypass for `/query`, `/query/*`, and `/api/*`.

## Errors and serialization

Reference retrieval validation errors map to HTTP 400, unsupported capabilities to 422, and backend retrieval failures to 502. Generic internal failures are logged and returned without exception details. BSON values pass through `bson_to_jsonable()` in `src/hybridrag/engine/utils.py`.

See [Configuration](../reference/configuration.md) for environment fields and [Security](../security.md) for tenant and operator controls.

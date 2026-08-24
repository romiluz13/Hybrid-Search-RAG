# Debugging

Debug HybridRAG by narrowing the failing layer: public API, retrieval orchestration, generated MongoDB pipeline, provider call, or storage. The commands below follow the repository guidance in `AGENTS.md`.

## Enable debug logging

For a small script or test:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

Logs from the reference API use the `hybridrag.api` logger in `src/hybridrag/api/main.py`; engine code uses the shared logger imported throughout `src/hybridrag/engine/`.

## Run the smallest failing test

```bash
pytest tests/path/to/test_file.py::test_function_name -v -s
```

`-s` exposes print output and live logs. Move outward only after the focused case is stable: file, subsystem, then `make test`.

## Inspect MongoDB pipelines

When a search returns unexpected results, print or log the pipeline immediately before `aggregate()`:

```python
import pprint

pprint.pprint(pipeline)
```

Check the first stage and its filter grammar:

- `$vectorSearch` uses MongoDB query operators such as `$eq`, `$in`, `$gte`, and `$lte`.
- `$search` uses Atlas Search operators such as `equals`, `range`, `text`, and `compound`.
- vector candidate count should be derived from `top_k` and passed through the selected search path.
- server-owned predicates must remain conjoined with caller filters.

Pipeline builders live in `src/hybridrag/enhancements/mongodb_hybrid_search.py` and `src/hybridrag/enhancements/filters/`. Storage execution lives in `src/hybridrag/engine/kg/mongo_impl.py`.

## Check service state

```bash
make atlas-check
make atlas-indexes
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

The API intentionally starts in degraded mode if MongoDB or provider configuration is unavailable. `/health` reports component state; `/ready` reports whether the RAG instance initialized.

Search indexes are eventually consistent. A `queryable` status does not prove freshly written documents are visible. Use the authenticated `GET /v1/search-indexes/sync` probe or the corresponding `HybridRAG.verify_index_sync()` method for a functional check.

## Common failure patterns

| Symptom | Check |
| --- | --- |
| Empty vector results | Embedding dimensions, index name, `numCandidates`, score threshold, and prefilter values |
| Atlas Search error | Atlas operator shape and mapped metadata types |
| `503 RAG system not initialized` | `.env`, MongoDB connectivity, and required provider keys |
| Cross-tenant result missing | `HYBRIDRAG_TENANT_FIELD`, API-key mapping, and mandatory predicate propagation |
| Integration test skipped or failing | Local MongoDB on port 27018 and the `integration` marker selection |
| Streaming appears buffered | Proxy buffering and the `application/x-ndjson` response path |

For environment fields, see [Configuration](../reference/configuration.md). For test selection, see [Testing](testing.md).

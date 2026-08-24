# Patterns and conventions

## Coding style

- **Formatter**: Black, 88-character line length
- **Import sorting**: isort with Black profile, `hybridrag` as first-party
- **Linter**: Ruff (E, W, F, I, B, C4, UP rules)
- **Type hints**: Required for all public functions. Use `list[T]` not `List[T]` (Python 3.11+).
- **Docstrings**: Google-style with Args, Returns, Raises sections

## Filter syntax (critical)

The codebase has three distinct filter syntaxes. Never mix them:

1. **Vector Search filters** (MQL): `{"metadata.category": {"$eq": "features"}}` — used with `$vectorSearch`
2. **Atlas Search filters** (compound): `{"equals": {"path": "metadata.category", "value": "features"}}` — used with `$search`
3. **Lexical prefilters**: `{"fuzzy": {"path": "content", "query": "text", "maxEdits": 2}}` — used with `$search.vectorSearch`

The public `FilterConfig` translates to the correct backend syntax automatically. See [Filters](../features/filters.md).

## Async patterns

All public APIs are async. Use `await` for all database and LLM calls:

```python
# Correct
results = await rag.query(query="test", mode="mix")

# Wrong — will return a coroutine, not results
results = rag.query(query="test", mode="mix")
```

## Datetime handling

Always use timezone-aware datetimes:

```python
# Correct
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)

# Wrong — deprecated, no timezone
timestamp = datetime.utcnow()
```

## Configuration

Settings come from environment variables via `pydantic-settings`. Never hardcode connection strings or API keys:

```python
from hybridrag.config.settings import get_settings
settings = get_settings()
client = AsyncMongoClient(settings.mongodb_uri.get_secret_value())
```

See [Configuration](../reference/configuration.md) for all settings.

## MongoDB conventions

- Use `allowDiskUse=True` on aggregate pipelines that may exceed 100MB memory
- Use `maxTimeMS` to bound query execution time
- Use `bson_to_jsonable()` from `src/hybridrag/engine/utils.py` before returning MongoDB documents through the API
- numCandidates should be 20x the limit (per MongoDB docs), capped at 1000
- Atlas Search indexes are eventually consistent — use `probe_index_sync()` to verify data ingestion

## Error handling

- Raise specific exceptions from `src/hybridrag/engine/exceptions.py` (`RetrievalCapabilityError`, `RetrievalExecutionError`, `RetrievalValidationError`)
- Unsupported capabilities should fail explicitly, not silently degrade
- Use `try/except` with `logger.debug()` for expected retry conditions (e.g., index sync polling)
- API endpoints catch exceptions and return appropriate HTTP status codes

## Naming conventions

- Collection names: `{workspace}_{type}` (e.g., `default_chunks`, `default_kg_edges`)
- Index names: `vector_knn_index_{collection}`, `text_search_index_{collection}`
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Import structure

- Public API exports flow through `src/hybridrag/__init__.py`
- Internal engine code lives in `src/hybridrag/engine/`
- The `HybridRAG` class in `core/rag.py` wraps the internal `BaseRAGEngine` from `engine/base_engine.py`
- Lazy imports for optional dependencies (CLI, UI, evaluation)

## Testing patterns

- Test markers: `p1` (critical), `p2` (important), `p3` (edge cases), `integration` (needs MongoDB), `benchmark` (performance)
- Unit tests mock MongoDB and API calls
- Integration tests require a live MongoDB on `localhost:27018`
- See [Testing](testing.md) for details.

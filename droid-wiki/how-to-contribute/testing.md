# Testing

HybridRAG uses pytest with automatic asyncio support. Unit tests isolate provider and database behavior, while integration, live, and benchmark tests are selected explicitly.

## Pytest configuration

`pyproject.toml` configures:

- `asyncio_mode = "auto"`;
- discovery under `tests/`;
- files named `test_*.py` and functions named `test_*`;
- verbose output with short tracebacks;
- coverage over `src/hybridrag`, excluding tests, caches, virtual environments, and the vendored-style engine API subtree.

There is no fixed coverage threshold. Coverage is reported to reveal gaps, not used as a single release criterion.

## Test layout

| Path | Focus |
| --- | --- |
| `tests/api/` | Public FastAPI query contracts |
| `tests/core/` | Public `HybridRAG`, retrieval options, diagnostics, and workspace behavior |
| `tests/enhancements/` | Filter builders, search pipelines, graph search, MongoDB helpers, and index probes |
| `tests/examples/` | Import and execution smoke tests for examples |
| `tests/integration/` | MongoDB-backed end-to-end pipelines and memory |
| `tests/benchmarks/` | Search performance |
| `tests/e2e_real_test.py` | Deterministic live-provider release gate |
| `tests/conftest.py` | Shared fixtures and test environment setup |

## Markers

| Marker | Meaning |
| --- | --- |
| `p1` | Critical functionality |
| `p2` | Important features |
| `p3` | Edge cases and lower-priority behavior |
| `integration` | Requires MongoDB and possibly external services |
| `benchmark` | Performance test, excluded from regular runs |

Examples:

```bash
pytest -m p1
pytest -m "p1 or p2"
pytest -m "not integration and not benchmark"
```

## Common commands

```bash
make test                 # Unit suite; excludes integration and benchmarks
make test-quick           # Fast enhancement tests
make test-cov             # HTML coverage report in htmlcov/
make test-integration     # MongoDB-backed integration tests
make example-smoke        # Example and API contract smoke tests
make contract-tests       # Canonical API and integration contracts
make benchmark            # Performance benchmarks
make release-gate-fast    # Compile, Ruff, and focused release tests
make release-gate-live    # Real-provider E2E gate
```

Run a focused test directly while iterating:

```bash
pytest tests/core/test_public_retrieval_options.py -v
pytest tests/core/test_public_retrieval_options.py::test_name -v -s
```

## Writing tests

Place a test beside the subsystem it covers, use `test_*.py` and `test_*`, and prefer pytest fixtures for setup. Async tests can be written directly because pytest-asyncio runs in auto mode:

```python
async def test_query_preserves_filter(rag):
    result = await rag.query(
        "release notes",
        mode="hybrid",
        filter_config=filter_config,
    )
    assert result
```

Mock MongoDB and provider APIs in unit tests. Mark tests `integration` when they require a real MongoDB feature such as `$vectorSearch`, `$search`, or `$rankFusion`. Mark timing-sensitive performance tests `benchmark`.

For search changes, assert the generated aggregation stage, not only the returned answer. For tenant-aware changes, test both authorized data and cross-tenant denial. For a bug fix, first add a test that reproduces the failure.

## Integration environment

Start the local MongoDB Atlas image on port 27018:

```bash
make mongo-up
make test-integration
```

Provider-backed tests may also require `VOYAGE_API_KEY` and an LLM key. Keep such tests out of the default unit path. The daily jobs in `.github/workflows/test.yml` separate the full unit suite, live release gate, optional Atlas cloud smoke test, and benchmarks.

See [Debugging](debugging.md) when a test needs pipeline-level inspection.

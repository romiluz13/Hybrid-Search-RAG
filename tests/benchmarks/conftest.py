"""
Benchmark configuration.

Provides fixtures for performance benchmarking. The `rag` fixture is shared with
the integration tests so benchmarks run against a real MongoDB (atlas-local on
localhost:27018). If MongoDB is unavailable, benchmark tests are skipped.
"""

import pytest

# Reuse the shared MongoDB-backed `rag` fixture from the integration conftest so
# benchmarks and integration tests exercise the same blessed stack. Importing it
# here makes the fixture resolvable from tests/benchmarks/ (conftest scope is
# hierarchical upward, not across sibling dirs).
from tests.integration.conftest import rag  # noqa: F401  (re-exported fixture)


@pytest.fixture(scope="session")
def benchmark_queries():
    """Sample queries for benchmarking."""
    return [
        "What is MongoDB Atlas?",
        "How does vector search work?",
        "mongodb hybrid search configuration",
        "atlas search full text features",
        "knowledge graph mongodb",
    ]

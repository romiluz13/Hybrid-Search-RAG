"""
Phase 2E: Infrastructure Findings (H18, H19, H26)

Tests verify:
- H18: Docker port bound to localhost only (127.0.0.1:27017:27017)
- H19: CI test.yml has --ignore flags for integration/benchmark/e2e tests
- H26: E2E test default DB name is test-specific, not production
"""

import os
import re

# ──────────────────────────────────────────────────────────────────────
# H18: Docker port exposed to all interfaces
# ──────────────────────────────────────────────────────────────────────


def test_h18_docker_port_bound_to_localhost():
    """Docker-compose must bind MongoDB port to 127.0.0.1 only."""
    compose_path = os.path.join(
        os.path.dirname(__file__), "..", "docker", "docker-compose.local.yml"
    )
    with open(compose_path) as f:
        content = f.read()

    # Must contain the localhost-bound port mapping
    assert "127.0.0.1:27017:27017" in content, (
        "Docker port must be bound to 127.0.0.1, not exposed to all interfaces"
    )

    # Must NOT contain a bare port mapping that exposes to 0.0.0.0
    # Match lines like '- "27017:27017"' or "- '27017:27017'" without 127.0.0.1
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") and "27017:27017" in stripped:
            assert "127.0.0.1" in stripped, (
                f"Port mapping must include 127.0.0.1 binding: {stripped}"
            )


# ──────────────────────────────────────────────────────────────────────
# H19: CI test suite no --ignore flags
# ──────────────────────────────────────────────────────────────────────


def test_h19_ci_test_yml_ignores_integration_tests():
    """CI test.yml must --ignore integration tests that need credentials."""
    workflow_path = os.path.join(
        os.path.dirname(__file__), "..", ".github", "workflows", "test.yml"
    )
    with open(workflow_path) as f:
        content = f.read()

    assert "--ignore=tests/integration/" in content, (
        "CI must --ignore tests/integration/ (requires MongoDB credentials)"
    )


def test_h19_ci_test_yml_ignores_benchmark_tests():
    """CI test.yml must --ignore benchmark tests that need credentials."""
    workflow_path = os.path.join(
        os.path.dirname(__file__), "..", ".github", "workflows", "test.yml"
    )
    with open(workflow_path) as f:
        content = f.read()

    assert "--ignore=tests/benchmarks/" in content, (
        "CI must --ignore tests/benchmarks/ (requires MongoDB credentials)"
    )


def test_h19_ci_test_yml_ignores_e2e_test():
    """CI test.yml must --ignore e2e_real_test.py that needs real APIs."""
    workflow_path = os.path.join(
        os.path.dirname(__file__), "..", ".github", "workflows", "test.yml"
    )
    with open(workflow_path) as f:
        content = f.read()

    assert "--ignore=tests/e2e_real_test.py" in content, (
        "CI must --ignore tests/e2e_real_test.py (requires real API credentials)"
    )


def test_h19_ci_test_yml_preserves_coverage():
    """CI test.yml must still have coverage reporting after adding --ignore flags."""
    workflow_path = os.path.join(
        os.path.dirname(__file__), "..", ".github", "workflows", "test.yml"
    )
    with open(workflow_path) as f:
        content = f.read()

    assert "--cov=src/hybridrag" in content, (
        "CI must preserve --cov=src/hybridrag coverage flag"
    )
    assert "--cov-report=term" in content, "CI must preserve --cov-report=term flag"


# ──────────────────────────────────────────────────────────────────────
# H26: E2E test production-like DB name
# ──────────────────────────────────────────────────────────────────────


def test_h26_e2e_test_db_name_not_production():
    """E2E test default DB name must NOT be 'hybridrag' (production name)."""
    e2e_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "e2e_real_test.py"
    )
    with open(e2e_path) as f:
        content = f.read()

    # Find the TEST_DB_NAME default value assignment
    # Pattern: os.environ.get("HYBRIDRAG_TEST_DB", "...")
    match = re.search(
        r'os\.environ\.get\(\s*"HYBRIDRAG_TEST_DB"\s*,\s*"([^"]+)"\s*\)',
        content,
    )
    assert match is not None, "Could not find HYBRIDRAG_TEST_DB env var default"
    default_name = match.group(1)
    assert default_name != "hybridrag", (
        f"E2E test default DB name must not be 'hybridrag' (production name), "
        f"got '{default_name}'"
    )
    assert "test" in default_name.lower() or "e2e" in default_name.lower(), (
        f"E2E test default DB name should contain 'test' or 'e2e', got '{default_name}'"
    )


def test_h26_e2e_test_db_env_override_still_works():
    """E2E test must still support HYBRIDRAG_TEST_DB env var override."""
    e2e_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "e2e_real_test.py"
    )
    with open(e2e_path) as f:
        content = f.read()

    assert "HYBRIDRAG_TEST_DB" in content, (
        "E2E test must still support HYBRIDRAG_TEST_DB env var for overriding DB name"
    )

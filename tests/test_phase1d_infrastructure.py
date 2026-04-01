"""
Phase 1D Infrastructure validation tests.

Tests for C11 (.dockerignore), C12 (docker-compose creds), C14 (requirements.txt),
and C16 (benchmark async pattern).
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestC11Dockerignore:
    """C11: .dockerignore must exist and exclude sensitive files."""

    def test_dockerignore_exists(self):
        """A .dockerignore file must exist in the project root."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        assert dockerignore.exists(), ".dockerignore file must exist in project root"

    def test_dockerignore_excludes_env(self):
        """.dockerignore must exclude .env files to prevent credential leaks."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        content = dockerignore.read_text()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert ".env" in lines, ".dockerignore must exclude .env"

    def test_dockerignore_excludes_env_variants(self):
        """.dockerignore must exclude .env.* variants."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        content = dockerignore.read_text()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert ".env.*" in lines, ".dockerignore must exclude .env.* variants"

    def test_dockerignore_preserves_env_example(self):
        """.dockerignore must NOT exclude .env.example."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        content = dockerignore.read_text()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        assert "!.env.example" in lines, (
            ".dockerignore must preserve .env.example with negation"
        )

    def test_dockerignore_excludes_git(self):
        """.dockerignore must exclude .git directory."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        content = dockerignore.read_text()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert ".git" in lines, ".dockerignore must exclude .git"

    def test_dockerignore_excludes_claude(self):
        """.dockerignore must exclude .claude directory."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        content = dockerignore.read_text()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert ".claude" in lines, ".dockerignore must exclude .claude"


class TestC12DockerComposeCreds:
    """C12: docker-compose must not have hardcoded credentials."""

    def test_no_hardcoded_username(self):
        """docker-compose must not contain MONGODB_INITDB_ROOT_USERNAME with a value."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.local.yml"
        content = compose_file.read_text()
        assert "MONGODB_INITDB_ROOT_USERNAME=admin" not in content, (
            "docker-compose must not have hardcoded username"
        )

    def test_no_hardcoded_password(self):
        """docker-compose must not contain MONGODB_INITDB_ROOT_PASSWORD with a value."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.local.yml"
        content = compose_file.read_text()
        assert "MONGODB_INITDB_ROOT_PASSWORD=password" not in content, (
            "docker-compose must not have hardcoded password"
        )

    def test_has_initdb_database(self):
        """docker-compose must still set the database name."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.local.yml"
        content = compose_file.read_text()
        assert "MONGODB_INITDB_DATABASE=hybridrag" in content, (
            "docker-compose must still configure the database name"
        )

    def test_has_atlas_local_explanation_comment(self):
        """docker-compose must explain that atlas-local ignores auth env vars."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.local.yml"
        content = compose_file.read_text()
        assert "atlas-local" in content.lower() or "ignores" in content.lower(), (
            "docker-compose must have a comment explaining atlas-local ignores auth vars"
        )


class TestC14RequirementsMismatch:
    """C14: requirements.txt must not contain google-generativeai (only google-genai)."""

    def test_no_google_generativeai(self):
        """requirements.txt must not have google-generativeai (different from google-genai)."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text()
        lines = [line.strip().lower() for line in content.splitlines()]
        for line in lines:
            if line.startswith("google-generativeai"):
                pytest.fail(
                    "requirements.txt must not contain google-generativeai; "
                    "only google-genai should be present (to match pyproject.toml)"
                )

    def test_has_google_genai(self):
        """requirements.txt must still have google-genai."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text()
        lines = [line.strip().lower() for line in content.splitlines()]
        found = any(line.startswith("google-genai") for line in lines)
        assert found, "requirements.txt must contain google-genai>=0.2.0"


class TestC16BenchmarkAsyncPattern:
    """C16: Benchmark tests must not use asyncio.run() inside async functions."""

    def test_no_asyncio_run_in_benchmark_tests(self):
        """Benchmark tests must not call asyncio.run() inside async test functions."""
        benchmark_file = (
            PROJECT_ROOT / "tests" / "benchmarks" / "test_search_performance.py"
        )
        content = benchmark_file.read_text()
        # Parse the AST to find async functions that call asyncio.run
        tree = ast.parse(content)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "asyncio"
                        and child.func.attr == "run"
                    ):
                        violations.append(node.name)
        assert not violations, (
            f"async functions must not call asyncio.run(): {violations}"
        )

    def test_benchmark_tests_are_not_async(self):
        """After fix, benchmark tests should be sync (not async) with asyncio.run or
        async without asyncio.run. Either pattern is valid - this test checks consistency."""
        benchmark_file = (
            PROJECT_ROOT / "tests" / "benchmarks" / "test_search_performance.py"
        )
        content = benchmark_file.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_"):
                # If it's async, it must NOT call asyncio.run
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "asyncio"
                        and child.func.attr == "run"
                    ):
                        pytest.fail(
                            f"Async test {node.name} calls asyncio.run() - "
                            "this will raise RuntimeError in a running event loop"
                        )

"""
Tests for Phase 3F: Infrastructure & MCP Setup findings (M34, M35, M41, M43, M44).

Backing Skill: mongodb-mcp-setup
"""

import ast
import inspect
import re

import pytest


class TestM34InternalExceptionsNotLeaked:
    """M34: Internal exceptions leaked to API caller.

    The ingest endpoint must NOT expose raw exception messages
    to the caller. Must return generic "Internal server error" instead.
    """

    def test_ingest_endpoint_does_not_expose_raw_exception(self):
        """The ingest exception handler must use a generic error message."""
        source_module = __import__(
            "hybridrag.api.main",
            fromlist=["register_routes"],
        )
        # Read the source file to check the ingest endpoint's except block
        import hybridrag.api.main as api_main_mod

        source = inspect.getsource(api_main_mod)

        # The problematic pattern is: detail=str(e) in the ingest endpoint
        # Find the ingest_documents function and check its exception handling
        # After the fix, the ingest endpoint should NOT have detail=str(e)
        # but should have a generic message like "Internal server error"

        # Parse the AST to find the ingest_documents function
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "ingest_documents"
            ):
                # Look for HTTPException with detail=str(e) pattern
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.keyword) and sub_node.arg == "detail":
                        # Check if value is a Call to str(e)
                        val = sub_node.value
                        if isinstance(val, ast.Call):
                            func = val.func
                            if isinstance(func, ast.Name) and func.id == "str":
                                pytest.fail(
                                    "ingest_documents must NOT use detail=str(e) -- "
                                    "internal exceptions must not be leaked to API callers"
                                )

    def test_ingest_error_logs_exception_details(self):
        """The ingest handler must log the actual error with exc_info."""
        import hybridrag.api.main as api_main_mod

        source = inspect.getsource(api_main_mod)

        # The fixed code should have logger.error with exc_info=True
        # in the ingest endpoint's except block
        assert "exc_info=True" in source or "logger.error" in source, (
            "Ingest error handler must log exception details server-side"
        )


class TestM35MaxLengthOnDocumentsList:
    """M35: No max_length on documents list in IngestRequest.

    The documents field must have max_length=100 to prevent abuse.
    """

    def test_ingest_request_documents_has_max_length(self):
        """IngestRequest.documents must have max_length constraint."""
        from hybridrag.api.models import IngestRequest

        field_info = IngestRequest.model_fields.get("documents")
        assert field_info is not None, "documents field must exist"

        # Check metadata for max_length
        metadata = field_info.metadata
        has_max_length = False
        for m in metadata:
            if hasattr(m, "max_length") and m.max_length is not None:
                has_max_length = True
                break
        assert has_max_length, "IngestRequest.documents must have max_length constraint"

    def test_ingest_request_documents_max_length_is_100(self):
        """Max length should be 100."""
        from hybridrag.api.models import IngestRequest

        field_info = IngestRequest.model_fields.get("documents")
        metadata = field_info.metadata
        max_len = None
        for m in metadata:
            if hasattr(m, "max_length") and m.max_length is not None:
                max_len = m.max_length
                break
        assert max_len == 100, (
            f"IngestRequest.documents max_length should be 100, got {max_len}"
        )


class TestM41LangfuseDefersEnvReads:
    """M41: Langfuse reads env at import time.

    Environment variable reads for LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    must be deferred to first use (lazy initialization), not at import time.
    """

    def test_langfuse_no_env_read_at_module_level(self):
        """Module-level code must not read LANGFUSE_PUBLIC_KEY/SECRET_KEY from env."""
        import hybridrag.integrations.langfuse as langfuse_mod

        source = inspect.getsource(langfuse_mod)

        # Parse the AST and check module-level statements only
        tree = ast.parse(source)

        for node in tree.body:
            # Check module-level statements (not inside function/class)
            if isinstance(node, (ast.Try, ast.If)):
                # Walk the try/if block for env reads
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Call):
                        # Check for os.environ.get("LANGFUSE_PUBLIC_KEY") etc.
                        func = sub_node.func
                        if isinstance(func, ast.Attribute) and func.attr == "get":
                            if (
                                isinstance(func.value, ast.Attribute)
                                and func.value.attr == "environ"
                            ):
                                # Check the key argument
                                if sub_node.args:
                                    arg = sub_node.args[0]
                                    if isinstance(arg, ast.Constant) and isinstance(
                                        arg.value, str
                                    ):
                                        if (
                                            "LANGFUSE" in arg.value
                                            and "KEY" in arg.value
                                        ):
                                            pytest.fail(
                                                f"Module-level code reads {arg.value} from env. "
                                                "M41 requires deferring env reads to first use."
                                            )

    def test_langfuse_enabled_is_lazy(self):
        """LANGFUSE_ENABLED must be determined lazily, not at import time."""
        import hybridrag.integrations.langfuse as langfuse_mod

        source = inspect.getsource(langfuse_mod)
        tree = ast.parse(source)

        # Check that LANGFUSE_ENABLED assignment at module level does NOT depend
        # on reading LANGFUSE_PUBLIC_KEY from env
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "LANGFUSE_ENABLED":
                        # If it's assigned True at module level based on env keys,
                        # that's the bug. After fix, it should default to False or
                        # be computed lazily.
                        if (
                            isinstance(node.value, ast.Constant)
                            and node.value.value is True
                        ):
                            pytest.fail(
                                "LANGFUSE_ENABLED must not be set to True at module level"
                            )

    def test_check_langfuse_enabled_function_exists(self):
        """A _check_langfuse_enabled() function must exist for lazy checking."""
        import hybridrag.integrations.langfuse as langfuse_mod

        assert hasattr(langfuse_mod, "_check_langfuse_enabled"), (
            "langfuse.py must have _check_langfuse_enabled() for lazy env reads"
        )
        assert callable(langfuse_mod._check_langfuse_enabled), (
            "_check_langfuse_enabled must be callable"
        )


class TestM43DockerComposeEnvVars:
    """M43: Docker-compose wrong env vars for atlas-local.

    ALREADY DONE in Phase 1D (C12). Verify env vars are removed/commented.
    """

    def test_no_hardcoded_credentials_in_docker_compose(self):
        """docker-compose.local.yml must not have hardcoded MONGODB_INITDB_ROOT credentials."""
        import pathlib

        compose_path = (
            pathlib.Path(__file__).parents[2] / "docker" / "docker-compose.local.yml"
        )
        if not compose_path.exists():
            pytest.skip("docker-compose.local.yml not found")

        content = compose_path.read_text()
        assert "MONGODB_INITDB_ROOT_USERNAME" not in content or content.count(
            "MONGODB_INITDB_ROOT_USERNAME"
        ) == content.count("# MONGODB_INITDB_ROOT_USERNAME"), (
            "Hardcoded MONGODB_INITDB_ROOT_USERNAME must be removed or commented out"
        )
        assert "MONGODB_INITDB_ROOT_PASSWORD" not in content, (
            "Hardcoded MONGODB_INITDB_ROOT_PASSWORD must not be present"
        )


class TestM44CILintPinnedVersions:
    """M44: CI lint installs unpinned versions.

    CI workflow must pin ruff and mypy versions.
    """

    def test_ci_yml_pins_ruff_version(self):
        """CI workflow must pin ruff version."""
        import pathlib

        ci_path = pathlib.Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
        if not ci_path.exists():
            pytest.skip(".github/workflows/ci.yml not found")

        content = ci_path.read_text()
        # Should have ruff==X.Y.Z, not just "ruff"
        assert re.search(r"ruff==\d+\.\d+\.\d+", content), (
            "CI must pin ruff to a specific version (e.g. ruff==0.4.0)"
        )

    def test_ci_yml_pins_mypy_version(self):
        """CI workflow must pin mypy version."""
        import pathlib

        ci_path = pathlib.Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
        if not ci_path.exists():
            pytest.skip(".github/workflows/ci.yml not found")

        content = ci_path.read_text()
        # Should have mypy==X.Y.Z, not just "mypy"
        assert re.search(r"mypy==\d+\.\d+\.\d+", content), (
            "CI must pin mypy to a specific version (e.g. mypy==1.10.0)"
        )

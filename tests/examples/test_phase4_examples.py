"""Tests for Phase 4 example fixes (C6, C7, M3, M4)."""

import ast

import pytest


@pytest.fixture
def ragas_source():
    """Read 08_evaluation_ragas.py source."""
    import pathlib

    path = pathlib.Path(__file__).parents[2] / "examples" / "08_evaluation_ragas.py"
    return path.read_text()


@pytest.fixture
def filters_source():
    """Read 07_custom_filters.py source."""
    import pathlib

    path = pathlib.Path(__file__).parents[2] / "examples" / "07_custom_filters.py"
    return path.read_text()


class TestC6RagasContextsFormat:
    """C6: RAGAS expects list[list[str]] with individual chunks."""

    def test_no_single_blob_contexts(self, ragas_source):
        """Contexts must NOT wrap context as a single element list [[context]]."""
        # The old pattern: [[context] if context else []]
        # This wraps the entire blob as one string in a list
        assert "[[context]" not in ragas_source, (
            "Found [[context]] pattern — RAGAS needs split chunks, not a single blob"
        )

    def test_uses_split_pattern(self, ragas_source):
        """Contexts must split by paragraph separator."""
        assert 'context.split("\\n\\n")' in ragas_source, (
            "Must split context by double-newline for RAGAS chunk granularity"
        )


class TestC7CloseSharedClient:
    """C7: main() must call close_shared_client() in finally."""

    def test_imports_close_shared_client(self, ragas_source):
        """close_shared_client must be imported."""
        assert "close_shared_client" in ragas_source

    def test_main_has_try_finally(self, ragas_source):
        """main() must wrap body in try/finally."""
        tree = ast.parse(ragas_source)
        main_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "main":
                main_func = node
                break
        assert main_func is not None, "main() function not found"
        # Check for Try node in main body
        has_try = any(isinstance(stmt, ast.Try) for stmt in main_func.body)
        assert has_try, "main() must contain a try/finally block"

    def test_finally_calls_close_shared_client(self, ragas_source):
        """finally block must call close_shared_client()."""
        tree = ast.parse(ragas_source)
        main_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "main":
                main_func = node
                break
        assert main_func is not None
        # Find Try node
        try_node = None
        for stmt in main_func.body:
            if isinstance(stmt, ast.Try):
                try_node = stmt
                break
        assert try_node is not None, "No try block found in main()"
        # Check finalbody has close_shared_client call
        found = False
        for stmt in try_node.finalbody:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Name) and func.id == "close_shared_client":
                    found = True
        assert found, "finally block must call close_shared_client()"


class TestM3DeadFilterObjects:
    """M3: example_combined_filters and example_practical_use_case must build filters."""

    def test_combined_filters_calls_build(self, filters_source):
        """example_combined_filters must call build_vector_search_filters."""
        # Find the function body text
        tree = ast.parse(filters_source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "example_combined_filters"
            ):
                # Check for a Call to build_vector_search_filters
                calls = [
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "build_vector_search_filters"
                ]
                assert len(calls) > 0, (
                    "example_combined_filters must call build_vector_search_filters()"
                )
                return
        pytest.fail("example_combined_filters function not found")

    def test_practical_use_case_calls_build(self, filters_source):
        """example_practical_use_case must call build_vector_search_filters."""
        tree = ast.parse(filters_source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "example_practical_use_case"
            ):
                calls = [
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "build_vector_search_filters"
                ]
                assert len(calls) > 0, (
                    "example_practical_use_case must call build_vector_search_filters()"
                )
                return
        pytest.fail("example_practical_use_case function not found")


class TestM4ModeComments:
    """M4: Search mode comments explaining naive, hybrid, mix semantics."""

    def test_modes_have_comments(self, ragas_source):
        """modes list must have explanatory comments nearby."""
        lines = ragas_source.split("\n")
        for i, line in enumerate(lines):
            if 'modes = ["naive", "hybrid", "mix"]' in line:
                # Check preceding lines (up to 5) for comments about each mode
                context = "\n".join(lines[max(0, i - 5) : i + 1])
                assert "naive" in context and "vector" in context.lower(), (
                    "Must explain naive mode (vector-only)"
                )
                assert "hybrid" in context and (
                    "local" in context.lower() or "global" in context.lower()
                ), "Must explain hybrid mode (local+global)"
                assert "mix" in context and (
                    "graph" in context.lower() or "entity" in context.lower()
                ), "Must explain mix mode (graph/entity)"
                return
        pytest.fail("modes list not found in ragas source")

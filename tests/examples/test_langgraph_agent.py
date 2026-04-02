"""Tests for the LangGraph agent example.

These tests verify the example script can be imported and basic structures work.
Full integration tests require API keys and MongoDB connection.
"""

import ast
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "examples"


class TestLangGraphAgentExample:
    """Test the LangGraph agent example components."""

    def test_example_parses(self) -> None:
        """Test that the example is syntactically valid without optional deps."""
        example_path = EXAMPLE_DIR / "09_langgraph_agent.py"
        source = example_path.read_text()
        tree = ast.parse(source, filename=str(example_path))

        assert tree.body, "Example should contain executable Python code"

    def test_example_file_exists(self) -> None:
        """Test that the example file exists."""
        example_path = EXAMPLE_DIR / "09_langgraph_agent.py"
        assert example_path.exists(), f"Example file not found at {example_path}"

    def test_example_has_docstring(self) -> None:
        """Test that the example has proper documentation."""
        example_path = EXAMPLE_DIR / "09_langgraph_agent.py"
        content = example_path.read_text()

        # Check for key documentation elements
        assert '"""' in content, "Example should have docstrings"
        assert "LangGraph" in content, "Should mention LangGraph"
        assert "HybridRAG" in content, "Should mention HybridRAG"
        assert "Prerequisites" in content, "Should have prerequisites section"
        assert "pip install" in content, "Should have installation instructions"

    def test_example_has_main_guard(self) -> None:
        """Test that the example has proper __main__ guard."""
        example_path = EXAMPLE_DIR / "09_langgraph_agent.py"
        content = example_path.read_text()

        assert 'if __name__ == "__main__"' in content, "Should have main guard"
        assert "asyncio.run" in content, "Should use asyncio.run for async main"

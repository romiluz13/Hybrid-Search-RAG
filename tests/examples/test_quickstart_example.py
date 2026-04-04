"""Smoke checks for the canonical quickstart example."""

import ast
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "examples"


def test_quickstart_parses() -> None:
    example_path = EXAMPLE_DIR / "01_quickstart.py"
    source = example_path.read_text()
    tree = ast.parse(source, filename=str(example_path))

    assert tree.body, "Quickstart example should contain executable Python code"


def test_quickstart_uses_supported_envs() -> None:
    content = (EXAMPLE_DIR / "01_quickstart.py").read_text()

    assert "VOYAGE_API_KEY" in content
    assert "OPENAI_API_KEY" in content
    assert "ANTHROPIC_API_KEY" not in content


def test_quickstart_uses_supported_modes() -> None:
    content = (EXAMPLE_DIR / "01_quickstart.py").read_text()

    assert 'mode="mix"' in content
    assert "query_with_answer" not in content
    assert "QueryParam(" not in content

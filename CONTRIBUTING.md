# Contributing to HybridRAG

Thank you for your interest in contributing! This guide will help you get started.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback

## Getting Started

### Development Setup

```bash
# Clone repository
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Write tests** for new functionality
5. **Run tests**: `pytest tests/ -v`
6. **Format code**: `black src/ tests/ && isort src/ tests/`
7. **Type check**: `mypy src/`
8. **Commit**: `git commit -m "Add feature: description"`
9. **Push**: `git push origin feature/your-feature-name`
10. **Create Pull Request**

## Code Style

### Python Style Guide

- Follow PEP 8
- Use **Black** formatter (88 char line length)
- Use **isort** for imports (Black profile)
- Type hints required for all functions
- Docstrings in Google style

### Example

```python
from typing import List, Dict, Optional
from hybridrag import Settings

async def process_documents(
    documents: List[str],
    settings: Optional[Settings] = None,
) -> Dict[str, any]:
    """
    Process a list of documents.

    Args:
        documents: List of document paths
        settings: Optional HybridRAG settings

    Returns:
        Dictionary with processing results

    Raises:
        ValueError: If documents list is empty
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    # Implementation
    ...
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files: `test_*.py`
- Name test functions: `test_*`
- Use pytest fixtures for setup

### Example Test

```python
import pytest
from hybridrag import create_hybridrag

@pytest.mark.asyncio
async def test_query_basic():
    rag = await create_hybridrag()
    result = await rag.query("test query", mode="naive")
    assert "answer" in result
    assert result["answer"] is not None
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_query.py -v

# Run with coverage
pytest tests/ -v --cov=src/hybridrag

# Run specific test
pytest tests/test_query.py::test_query_basic -v
```

## Project Structure

```
HybridRAG/
├── src/hybridrag/       # Main source code
│   ├── api/            # FastAPI endpoints
│   ├── cli/            # CLI interface
│   ├── config/         # Configuration
│   ├── core/           # Core functionality
│   ├── engine/         # RAG engine
│   ├── enhancements/   # Enhancements
│   ├── ingestion/      # Document ingestion
│   ├── integrations/   # External integrations
│   ├── memory/         # Conversation memory
│   └── ui/             # Chainlit UI
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Areas for Contribution

### High Priority

- **Documentation**: Improve docs, add examples
- **Tests**: Increase test coverage
- **Performance**: Optimize queries, caching
- **Error Handling**: Better error messages

### Features

- **New Query Modes**: Implement custom retrieval strategies
- **Integrations**: Add support for more LLM providers
- **UI Improvements**: Enhance Chainlit interface
- **CLI Features**: Add more CLI commands

### Bug Fixes

- Check [GitHub Issues](https://github.com/romiluz13/Hybrid-Search-RAG/issues)
- Fix bugs and add tests
- Document the fix

## Pull Request Process

1. **Update Documentation**: Update relevant docs
2. **Add Tests**: Ensure new code is tested
3. **Update CHANGELOG**: Document your changes
4. **Run Checks**: Ensure all tests pass
5. **Write Clear PR**: Describe what and why

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted (Black)
- [ ] Type hints added
```

## Commit Messages

Follow conventional commits:

```
feat: Add new query mode
fix: Fix MongoDB connection issue
docs: Update installation guide
perf: Optimize vector search
test: Add tests for query modes
```

## Questions?

- Open a [GitHub Issue](https://github.com/romiluz13/Hybrid-Search-RAG/issues)
- Check existing documentation
- Review existing code for examples

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing! 🎉

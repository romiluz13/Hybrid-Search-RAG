# Getting started

## Prerequisites

- Python 3.11 or higher
- Docker Desktop (for local MongoDB)
- API keys for production use: Voyage AI (embeddings) and one LLM provider (OpenAI, Anthropic, or Gemini)

## Installation

```bash
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG

# Create virtual environment and install all dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Or use the Makefile shortcut
make first-time-setup
```

## Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Required | Default | Purpose |
| -------- | -------- | ------- | ------- |
| `MONGODB_URI` | No | `mongodb://localhost:27018/?directConnection=true` | MongoDB connection string |
| `MONGODB_DATABASE` | No | `hybridrag` | Database name |
| `VOYAGE_API_KEY` | Yes (production) | — | Voyage AI embeddings |
| `LLM_PROVIDER` | No | `anthropic` | LLM provider: `anthropic`, `openai`, `gemini`, `grove` |
| `ANTHROPIC_API_KEY` | If provider=anthropic | — | Anthropic Claude API key |
| `OPENAI_API_KEY` | If provider=openai | — | OpenAI API key |
| `GEMINI_API_KEY` | If provider=gemini | — | Google Gemini API key |

See [Configuration](../reference/configuration.md) for the full settings reference.

## Running the demo

### No-keys demo (60 seconds)

```bash
make demo
```

This starts a local MongoDB (`mongodb/mongodb-atlas-local:preview` via Docker) and runs the real hybrid search pipeline against seeded data with sample vectors. No API keys needed.

### Full generative RAG demo

```bash
make demo-full
```

Requires `VOYAGE_API_KEY` and an LLM key in `.env`. Ingests real Voyage embeddings and generates an answer.

## Running the services

```bash
make run-api     # FastAPI server at http://localhost:8000
make run-ui      # Chainlit chat UI at http://localhost:8001
make run-cli     # Interactive CLI
```

## Python SDK usage

```python
import asyncio
from hybridrag import create_hybridrag

async def main():
    rag = await create_hybridrag()

    # Ingest documents
    await rag.ingest_files("./documents/")

    # Query
    result = await rag.query_with_sources(
        "How does hybrid search work?",
        mode="mix",
        top_k=5,
    )
    print(result["answer"])
    print(result["references"])

asyncio.run(main())
```

## CLI usage

```bash
hybridrag chat                          # Interactive chat
hybridrag ingest ./documents/           # Ingest files
hybridrag query "What is MongoDB?"      # One-shot query
hybridrag status                        # System status
```

## Testing

```bash
make test           # All tests (excludes integration)
make test-quick     # Fast unit tests only
make test-cov       # With coverage report
make test-integration  # MongoDB-backed integration tests
```

Tests use pytest with pytest-asyncio. See [Testing](../how-to-contribute/testing.md) for patterns and markers.

## Code quality

```bash
make lint           # Ruff linting
make format         # Black + isort formatting
make typecheck      # MyPy type checking
make ci             # Full CI suite (lint + typecheck + test)
```

## Build

```bash
make build          # Build distribution packages
make docker         # Build Docker image
```

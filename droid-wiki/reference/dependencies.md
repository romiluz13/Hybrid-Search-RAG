# Dependencies

`pyproject.toml` keeps the base library focused while optional extras add the API, UI, ingestion, evaluation, observability, agent, and contributor toolchains. The `all` extra installs every named optional group.

## Build system

| Package | Constraint | Why it is used |
| --- | --- | --- |
| setuptools | `>=68.0` | PEP 517 build backend and `src/` package discovery |
| wheel | unpinned | Wheel distribution support |

## Required runtime packages

### Core and validation

| Package | Constraint | Why it is used |
| --- | --- | --- |
| certifi | `>=2024.2.2` | Trusted CA bundle for MongoDB TLS, including macOS Python installs |
| numpy | `>=1.24.0` | Embedding arrays and numeric vector handling |
| python-dotenv | `>=1.0.0` | Load local `.env` configuration |
| pydantic | `>=2.0.0` | Request, response, and domain validation |
| pydantic-settings | `>=2.0.0` | Typed environment-backed settings |

### MongoDB

| Package | Constraint | Why it is used |
| --- | --- | --- |
| pymongo | `>=4.7.0,<5.0` | MongoDB driver, async client, aggregation, and Search index management |
| motor | `>=3.4.0,<4.0` | Async MongoDB compatibility used by engine integrations |

### Model providers

| Package | Constraint | Why it is used |
| --- | --- | --- |
| voyageai | `>=0.3.0` | Embeddings and reranking |
| anthropic | `>=0.39.0` | Claude generation |
| openai | `>=1.45.0` | OpenAI and OpenAI-compatible gateways; version supports `max_completion_tokens` |
| google-genai | `>=0.2.0` | Gemini generation |

### Async, resilience, and processing

| Package | Constraint | Why it is used |
| --- | --- | --- |
| tiktoken | `>=0.8.0` | Token counting and chunk budgets |
| tenacity | `>=9.0.0` | Retry policies around remote calls |
| aioboto3 | `>=13.2.0` | Async AWS service access |
| aiohttp | `>=3.11.9` | Async HTTP transport |
| httpx | `>=0.27.0` | HTTP client for API integrations and tests |
| json-repair | `>=0.54.0` | Recover structured data from imperfect model JSON |

## Optional groups

### `api`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| ascii-colors | `>=0.11.0` | Engine server console output |
| fastapi | `>=0.109.0` | HTTP API framework and OpenAPI generation |
| PyJWT | `>=2.8.0` | JWT authentication for the full engine server |
| python-multipart | `>=0.0.18` | Multipart document uploads and form handling |
| uvicorn | `>=0.27.0` | ASGI server |

### `ui`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| chainlit | `>=1.0.0` | Optional chat interface |
| pymupdf | `>=1.23.0` | PDF handling in UI/document workflows |

### `cli`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| rich | `>=13.0.0` | Styled terminal output |
| typer | `>=0.9.0` | CLI command definitions |

### `ingestion`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| docling | `>=2.0.0` | Structured document conversion |
| docling-core | `>=2.0.0` | Docling document primitives |
| transformers | `>=4.30.0` | Model-backed ingestion processing |
| tavily-python | `>=0.3.0` | Web extraction for URL and website ingestion |

### `observability`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| langfuse | `>=2.0.0` | Optional LLM tracing and usage observability |

### `evaluation`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| ragas | `>=0.2.0` | RAG quality evaluation |
| datasets | `>=2.14.0` | Evaluation datasets |
| langchain-openai | `>=0.2.0` | OpenAI adapter for evaluation workflows |

### `agent`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| langchain-core | `>=0.3.0` | Agent/message abstractions |
| langchain-anthropic | `>=0.3.0` | Anthropic integration for agents |
| langgraph | `>=0.2.0` | Stateful agent graphs |

### `dev`

| Package | Constraint | Why it is used |
| --- | --- | --- |
| pytest | `>=7.0.0` | Test runner |
| pytest-asyncio | `>=0.23.0` | Async test execution |
| pytest-cov | `>=4.0.0` | Coverage reporting |
| black | `>=24.0.0` | Code formatting |
| isort | `>=5.13.0` | Import ordering |
| mypy | `>=1.8.0` | Static type checks |
| ruff | `>=0.2.0` | Fast linting and import/style checks |

### `all`

`mongodb-hybridrag[api,ui,cli,ingestion,observability,evaluation,agent,dev]` is an umbrella extra. It adds no package directly; it selects all eight functional extras.

```bash
pip install -e .             # Required runtime packages
pip install -e ".[api]"      # Reference API
pip install -e ".[dev]"      # Contributor tools
pip install -e ".[all]"      # Complete environment
```

The manifest contains 17 required runtime requirements, 27 package entries across the eight functional optional groups, one umbrella group, and two build requirements. Broader dependency totals may include resolved transitive packages; inspect the installed environment or lock output when reproducing an exact deployment.

See [Tooling](../how-to-contribute/tooling.md) for commands and [Configuration](configuration.md) for provider selection.

# HybridRAG Examples

Practical examples demonstrating HybridRAG capabilities.

## Blessed Stack

The canonical reference path for these examples is:

- MongoDB Atlas 8.2+ or `mongodb/mongodb-atlas-local:preview` on `mongodb://localhost:27018/?directConnection=true`
- Voyage for embeddings and rerank
- OpenAI API or an OpenAI-compatible endpoint for LLM generation
- Python library + REST API + CLI

## Quick Start

```bash
# Install HybridRAG
pip install mongodb-hybridrag[all]

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run examples
python examples/01_quickstart.py
```

## Examples

### 01_quickstart.py
Basic HybridRAG usage:
- Initialize the RAG system
- Ingest documents
- Query the knowledge base

```bash
python examples/01_quickstart.py
```

### 02_hybrid_search.py
Advanced hybrid search features:
- Query type detection for reranking
- Filter builders (vector search + Atlas Search)
- Search configuration options
- Multi-field weighted search
- Per-pipeline score extraction

```bash
python examples/02_hybrid_search.py
```

### 03_prompts_usage.py
Prompts module demonstration:
- System prompts (full and compact)
- Custom domain-specific prompts
- Query-type detection
- Entity extraction prompts
- Memory and topic prompts

```bash
python examples/03_prompts_usage.py
```

### 04_mix_mode_search.py
Mix mode search capabilities:
- Graph traversal configuration
- Mix mode search configuration
- Result analysis with source breakdown
- Entity-only graph search patterns

```bash
python examples/04_mix_mode_search.py
```

### 09_langgraph_agent.py
**Building AI Agents on HybridRAG:**
- HybridRAG as a LangChain tool
- LangGraph agent with ReAct pattern
- Agent decides when to use knowledge base
- Multi-turn conversation with state management

```bash
pip install mongodb-hybridrag[agent]
python examples/09_langgraph_agent.py
```

This is the "user side" example - showing how to take HybridRAG and build
sophisticated agents on top of it.

## Prerequisites

### Required API Keys
- **MongoDB Atlas**: Connection string for your cluster
- **Voyage AI**: API key for embeddings and reranking

### Optional API Keys
- **OpenAI API**: Canonical public LLM path for the blessed stack
- **OpenAI-compatible endpoint**: Supported for live validation via `OPENAI_BASE_URL`

### MongoDB Atlas Setup
1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a database user
3. Whitelist your IP address
4. Get connection string (SRV format)
5. Create Atlas Search index (for hybrid search)

## Common Patterns

### Basic Query
```python
from hybridrag import create_hybridrag, QueryParam

rag = await create_hybridrag()
response = await rag.query("What is RAG?")
```

### Hybrid Search with Filters
```python
from hybridrag.enhancements import (
    VectorSearchFilterConfig,
    build_vector_search_filters,
)

config = VectorSearchFilterConfig(
    date_field="timestamp",
    start_date=datetime.now() - timedelta(days=7),
    equality_filters={"category": "tech"},
)
filters = build_vector_search_filters(config)
```

### Query-Type Aware Reranking
```python
from hybridrag import detect_query_type, select_rerank_instruction

query_type = detect_query_type("How do I install MongoDB?")
# Returns: QueryType.TOOLS

instruction = select_rerank_instruction(query_type)
# Returns: Tool-focused reranking instruction
```

### Custom Domain Prompts
```python
from hybridrag import create_system_prompt

prompt = create_system_prompt(
    domain="Legal Research",
    persona="Legal Analyst",
    entity_types=["case", "statute", "court"],
)
```

### HybridRAG as Agent Tool
```python
from langchain_core.tools import tool
from hybridrag import create_hybridrag

# Initialize HybridRAG once
rag = await create_hybridrag()

@tool
async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    result = await rag.query_with_sources(query, mode="mix")
    return result.get("answer", "No information found.")

# Use in LangGraph agent
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, [search_knowledge_base])
```

## Troubleshooting

### ImportError
Make sure you've installed with all dependencies:
```bash
pip install mongodb-hybridrag[all]
```

### MongoDB Connection Error
1. Check your connection string in `.env`
2. Verify IP whitelist in Atlas
3. Test connection: `make atlas-check`

### Missing API Keys
Set required environment variables:
```bash
export MONGODB_URI="mongodb+srv://..."
export VOYAGE_API_KEY="pa-..."
export OPENAI_API_KEY="sk-..."
# Optional for OpenAI-compatible gateways:
# export OPENAI_BASE_URL="https://your-endpoint/v1"
# export OPENAI_EXTRA_HEADERS='{"api-key":"..."}'
```

## More Resources

- [HybridRAG Documentation](https://github.com/romiluz13/HybridRAG)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Voyage AI Documentation](https://docs.voyageai.com/)

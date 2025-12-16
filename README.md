# HybridRAG: State-of-the-Art RAG with MongoDB

## Acknowledgements

This project builds upon the excellent work of the open-source community in GraphRAG and Retrieval-Augmented Generation. We extend our gratitude to the original researchers and developers who paved the way for efficient RAG systems.

This project enhances the RAG engine with:
- **MongoDB Atlas** for storage (Graph + Vector + Key-Value).
- **Voyage AI** for high-quality embeddings and reranking.
- **Multiple LLM Support**: Gemini, Claude, OpenAI.
- **Improved Caching**: Intelligent caching for LLM responses and embeddings.
- **Implicit Expansion**: Automatically discovering related entities.

This is a production-ready implementation of GraphRAG concepts, built on top of a highly optimized base engine.
- **Knowledge Graph Construction** - Automatic entity and relationship extraction
- **Entity Boosting** - Enhanced retrieval through entity-aware reranking
- **Implicit Semantic Expansion** - Find related concepts via vector similarity
- **Conversation Memory** - Multi-turn conversation support with MongoDB-backed sessions
- **Hybrid Search** - Combined vector and text search with MongoDB $rankFusion

## Quick Start

### Prerequisites

- Python 3.11+
- MongoDB Atlas cluster with Vector Search enabled
- Voyage AI API key
- LLM API key (Anthropic, OpenAI, or Google)

### Installation

```bash
# Install from PyPI
pip install mongodb-hybridrag

# Or install from source
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
pip install -e .
```

### Configuration

Create a `.env` file with your credentials:

```bash
# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net
MONGODB_DATABASE=hybridrag

# Voyage AI
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx

# LLM Provider (choose one)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxxxxxxx
# GEMINI_API_KEY=xxxxxxxxxxxxx

# Optional: Langfuse Observability
# LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxx
# LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxx
```

### Basic Usage

```python
import asyncio
from hybridrag import create_hybridrag, Settings

async def main():
    # Initialize HybridRAG
    settings = Settings(
        mongodb_database="my_database",
        llm_provider="anthropic",  # or "openai", "gemini"
    )
    rag = await create_hybridrag(settings)

    # Ingest documents
    await rag.ingest("path/to/document.pdf")

    # Query with conversation memory
    session_id = await rag.create_conversation_session()

    result = await rag.query_with_memory(
        query="What is this document about?",
        session_id=session_id,
        mode="mix",  # Combines knowledge graph and vector search
    )

    print(result["answer"])

asyncio.run(main())
```

## Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `mix` | Knowledge graph + vector search (recommended) | General queries |
| `local` | Entity-focused retrieval | Specific entity queries |
| `global` | Community summaries | High-level overview |
| `hybrid` | Local + global | Comprehensive answers |
| `naive` | Vector search only | Simple similarity search |

## API Server

Start the FastAPI server:

```bash
uvicorn src.hybridrag.api.main:app --reload
```

### Endpoints

- `POST /query` - Query the RAG system
- `POST /ingest` - Ingest documents
- `POST /sessions` - Create conversation session
- `GET /sessions/{id}/history` - Get conversation history
- `GET /health` - Health check

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HybridRAG                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Voyage    │  │   Claude/   │  │    MongoDB Atlas    │  │
│  │  Embeddings │  │  GPT/Gemini │  │  (Vector + Graph)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   Enhancements                          ││
│  │  • Entity Boosting  • Implicit Expansion  • Reranking   ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Conversation Memory                     ││
│  │           MongoDB-backed session storage                ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Configuration Options

```python
from hybridrag import Settings

settings = Settings(
    # MongoDB
    mongodb_database="hybridrag",

    # Embedding
    embedding_model="voyage-3-large",
    embedding_dimensions=1024,

    # Reranking
    rerank_model="rerank-2.5",
    rerank_top_k=10,

    # LLM
    llm_provider="anthropic",  # "openai", "gemini"
    llm_model="claude-sonnet-4-20250514",

    # Query defaults
    default_query_mode="mix",
    chunk_top_k=10,
    entity_top_k=60,
)
```

## Development

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy src/

# Format code
black src/ tests/
isort src/ tests/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

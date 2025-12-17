<div align="center">

```
██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗ ██████╗  █████╗  ██████╗
██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██╔══██╗██╔════╝
███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║██████╔╝███████║██║  ███╗
██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║██╔══██╗██╔══██║██║   ██║
██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
```

# The Atomic RAG Boilerplate

**Stop syncing 4 databases. Store vectors, graphs, and docs in one ACID-compliant MongoDB document.**

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)](https://www.mongodb.com/atlas)
[![Voyage AI](https://img.shields.io/badge/Voyage_AI-Embeddings-purple.svg)](https://www.voyageai.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-hybrid-search-works) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 The Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│  THE FRAGMENTED WAY                                                      │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Pinecone │  │  Neo4j   │  │  Redis   │  │ Postgres │                 │
│  │ Vectors  │  │  Graph   │  │  Cache   │  │ Metadata │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │             │             │             │                        │
│       └─────────────┴─────────────┴─────────────┘                        │
│                         │                                                │
│                    SYNC HELL 😱                                          │
│         If one write fails, your RAG returns                             │
│         vectors for deleted text                                         │
└─────────────────────────────────────────────────────────────────────────┘

                              VS

┌─────────────────────────────────────────────────────────────────────────┐
│  THE HYBRIDRAG WAY                                                       │
│                                                                          │
│                    ┌─────────────────────┐                               │
│                    │   MongoDB Atlas     │                               │
│                    │  ┌───┐ ┌───┐ ┌───┐  │                               │
│                    │  │ V │ │ G │ │ K │  │                               │
│                    │  └───┘ └───┘ └───┘  │                               │
│                    │  Vector Graph  KV   │                               │
│                    └─────────────────────┘                               │
│                              │                                           │
│                    ONE DOCUMENT = ATOMIC ✅                              │
│              All or nothing. Never inconsistent.                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  🔄 ATOMIC UPDATES         Vector + metadata + graph in one transaction │
│  🔍 HYBRID SEARCH          Vector + Graph + Keyword with RRF fusion     │
│  🧠 KNOWLEDGE GRAPH        Automatic entity & relationship extraction   │
│  💬 SELF-COMPACTING MEMORY Conversations auto-summarize, never lost     │
│  🚀 ENTITY BOOSTING        Knowledge graph enhances vector reranking    │
│  📊 RAGAS EVALUATION       Built-in RAG quality metrics                 │
│  🔌 MULTI-LLM              Gemini, Claude, OpenAI - switch anytime      │
│  📈 LANGFUSE TRACING       Production observability built-in            │
│  🎨 CHAINLIT UI            Beautiful web chat interface                 │
│  ⚡ VOYAGE AI              State-of-the-art embeddings + reranking      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 How Hybrid Search Works

HybridRAG doesn't just do vector search. It combines **three retrieval methods** using **Reciprocal Rank Fusion (RRF)**:

```
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   VECTOR    │     │    GRAPH    │     │   KEYWORD   │
  │   SEARCH    │     │   SEARCH    │     │   SEARCH    │
  │             │     │             │     │             │
  │  Semantic   │     │  Entity     │     │   Text      │
  │  Similarity │     │  Relations  │     │  Matching   │
  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │                 │
                    │   RRF FUSION    │
                    │                 │
                    │  RRF(d) = Σ 1   │
                    │         ─────   │
                    │         k + r   │
                    │                 │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  FINAL RANKED   │
                    │    RESULTS      │
                    └─────────────────┘
```

**Why RRF?** Documents appearing high in multiple search results get boosted. A result ranked #1 in vectors and #3 in graph beats a result ranked #1 in only one method.

---

## 🚀 Quick Start

### Installation

```bash
# Full installation with all features
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
pip install -e ".[all]"
```

### Configuration

```bash
# Create .env file
cat > .env << EOF
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DATABASE=hybridrag
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
EOF
```

### Launch Web UI

```bash
chainlit run src/hybridrag/ui/chat.py
```

Then open `http://localhost:8000` - drag & drop files to ingest, ask questions!

---

## 📖 Usage

### Python SDK

```python
import asyncio
from hybridrag import create_hybridrag

async def main():
    # Initialize
    rag = await create_hybridrag()

    # Ingest documents
    await rag.ingest("path/to/documents/")

    # Query with conversation memory
    session_id = await rag.create_conversation_session()

    result = await rag.query_with_memory(
        query="What are the key findings?",
        session_id=session_id,
        mode="mix",  # Vector + Graph + Keyword
    )

    print(result["answer"])

asyncio.run(main())
```

### Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `mix` | KG + Vector + Keyword (recommended) | General queries |
| `local` | Entity-focused retrieval | Specific entities |
| `global` | Community summaries | High-level overview |
| `hybrid` | Local + Global combined | Comprehensive answers |
| `naive` | Vector search only | Simple similarity |

### CLI Interface

```bash
hybridrag  # Launch interactive CLI

# Commands:
# > ingest path/to/file.pdf
# > What is this document about?
# > /mode mix
# > /status
# > exit
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HybridRAG                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │   Voyage AI    │  │  Claude/GPT/   │  │      MongoDB Atlas         │ │
│  │   Embeddings   │  │    Gemini      │  │                            │ │
│  │   + Reranking  │  │                │  │  ┌──────┐ ┌──────┐ ┌────┐  │ │
│  └────────────────┘  └────────────────┘  │  │Vector│ │Graph │ │ KV │  │ │
│                                          │  └──────┘ └──────┘ └────┘  │ │
│                                          └────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         ENHANCEMENTS                                 ││
│  │  Entity Boosting │ Implicit Expansion │ Self-Compacting Memory      ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        INTERFACES                                    ││
│  │        Chainlit UI  │  Rich CLI  │  REST API  │  Python SDK         ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Why Not Postgres?

| Task | Postgres + pgvector | HybridRAG |
|------|---------------------|-----------|
| Add metadata field | `ALTER TABLE` + backfill + reindex | Just add it |
| Change embedding model | Rewrite entire table (MVCC bloat) | Bulk update, no rewrite |
| Hybrid search | Manual result merging in app code | Single aggregation pipeline |
| Filter vectors by metadata | Separate index, query planner struggles | Compound index, native |
| Time to first query | Hours (extensions, schema, indexes) | 30 minutes (Atlas free tier) |

---

## 🔧 Configuration

```python
from hybridrag import Settings

settings = Settings(
    # MongoDB
    mongodb_database="hybridrag",

    # Embeddings
    embedding_model="voyage-3-large",
    embedding_dimensions=1024,

    # Reranking
    rerank_model="rerank-2.5",
    rerank_top_k=10,

    # LLM
    llm_provider="anthropic",  # or "openai", "gemini"
    llm_model="claude-sonnet-4-20250514",

    # Memory
    memory_max_tokens=32000,  # Self-compaction threshold
)
```

---

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Options](docs/configuration.md)
- [Query Modes Explained](docs/query-modes.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ && isort src/
```

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Vector Search + Graph Search + Keyword Search = HYBRID      ║
║                                                               ║
║   Reciprocal Rank Fusion merges them all into one result      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

**Made with ❤️ for the RAG community**

[⬆ Back to Top](#)

</div>

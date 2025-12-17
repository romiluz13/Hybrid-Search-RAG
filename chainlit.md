# Welcome to HybridRAG

**Intelligent Knowledge Base Search** powered by:
- **MongoDB Atlas** - Vector + Graph Storage
- **Voyage AI** - State-of-the-art Embeddings & Reranking
- **Multi-Provider LLM** - Anthropic, OpenAI, Gemini, and more

## Getting Started

1. **Upload Documents** - Drag & drop PDF, TXT, or Markdown files
2. **Ask Questions** - Query your documents using natural language
3. **Explore Connections** - Leverage knowledge graph relationships

## Commands

- `/mode <local|global|hybrid|naive|mix>` - Switch query modes
- `/status` - View system configuration
- `/help` - Show available commands

## Query Modes

| Mode | Description |
|------|-------------|
| `mix` | Combined KG + Vector search (recommended) |
| `local` | Entity-focused graph neighbors |
| `global` | Community summaries |
| `hybrid` | Local + Global combined |
| `naive` | Direct vector search |

---

Upload a document to get started!

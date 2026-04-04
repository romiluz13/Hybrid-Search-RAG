# API Reference

Complete Python SDK reference for HybridRAG.

## Quick Start

```python
import asyncio
from hybridrag import create_hybridrag

async def main():
    # Initialize HybridRAG (auto-initializes by default)
    rag = await create_hybridrag()

    # Ingest documents from folder (uses Docling processor)
    results = await rag.ingest_files("path/to/documents/")

    # Or insert raw text directly
    await rag.insert(["Document 1 content...", "Document 2 content..."])

    # Create conversation session
    session_id = await rag.create_conversation_session()

    # Query with memory
    result = await rag.query_with_memory(
        query="What are the key findings?",
        session_id=session_id,
        mode="mix",
    )

    print(result["answer"])

asyncio.run(main())
```

## Core Classes

### `HybridRAG`

Main class for interacting with HybridRAG.

```python
from hybridrag import HybridRAG, Settings, get_settings

# Initialize with default settings
rag = HybridRAG()
await rag.initialize()

# Initialize with custom settings
settings = Settings(
    mongodb_database="my_database",
    voyage_embedding_model="voyage-3-large",
    llm_provider="openai",
)
rag = HybridRAG(settings=settings)
await rag.initialize()

# Or use factory function (auto-initializes)
rag = await create_hybridrag(settings=settings)
```

#### Methods

##### `insert(documents: str | Sequence[str], ids: Sequence[str] | None = None, file_paths: Sequence[str] | None = None) -> None`

Insert raw text documents into the RAG system.

```python
# Insert single document
await rag.insert("Document content here...")

# Insert multiple documents
await rag.insert([
    "Document 1 content...",
    "Document 2 content...",
])

# Insert with IDs and file paths
await rag.insert(
    documents=["Content 1", "Content 2"],
    ids=["doc1", "doc2"],
    file_paths=["file1.pdf", "file2.pdf"],
)
```

**Parameters:**
- `documents` (str | Sequence[str]): Single document or list of documents
- `ids` (Sequence[str] | None): Optional document IDs
- `file_paths` (Sequence[str] | None): Optional file paths for metadata

##### `ingest_files(folder_path: str | Path, config: IngestionConfig | None = None, progress_callback: Callable[[int, int], None] | None = None) -> list[IngestionResult]`

Ingest documents from a folder using Docling document processor.

```python
from hybridrag import IngestionConfig, ChunkingConfig

# Ingest all files from folder
results = await rag.ingest_files("./documents/")

# Ingest with custom config
config = IngestionConfig(
    chunking=ChunkingConfig(
        max_tokens=512,
        chunk_size=1000,
        chunk_overlap=200,
    ),
    enable_audio_transcription=True,
)
results = await rag.ingest_files("./documents/", config=config)

# Check results
for r in results:
    if r.success:
        print(f"✓ {r.title}: {r.chunks_created} chunks")
    else:
        print(f"✗ {r.title}: {r.errors}")
```

**Parameters:**
- `folder_path` (str | Path): Path to folder containing documents
- `config` (IngestionConfig | None): Optional ingestion configuration
- `progress_callback` (Callable[[int, int], None] | None): Optional progress callback

**Returns:** List of `IngestionResult` objects

##### `ingest_file(file_path: str | Path, config: IngestionConfig | None = None) -> IngestionResult`

Ingest a single file using Docling document processor.

```python
result = await rag.ingest_file("./report.pdf")
if result.success:
    print(f"Ingested {result.chunks_created} chunks")
```

**Parameters:**
- `file_path` (str | Path): Path to file to ingest
- `config` (IngestionConfig | None): Optional ingestion configuration

**Returns:** `IngestionResult` object

##### `ingest_url(url: str, config: IngestionConfig | None = None) -> IngestionResult`

Ingest content from a single URL using Tavily Extract API.

**Requires:** `TAVILY_API_KEY` environment variable to be set.

```python
# Extract content from a single URL
result = await rag.ingest_url("https://docs.mongodb.com/atlas/vector-search/overview/")

if result.success:
    print(f"Ingested {result.chunks_created} chunks from {result.title}")
    print(f"Source: {result.source}")
else:
    print(f"Errors: {result.errors}")
```

**Parameters:**
- `url` (str): URL to extract content from (must be valid http/https URL)
- `config` (IngestionConfig | None): Optional ingestion configuration

**Returns:** `IngestionResult` object

**Features:**
- Uses Tavily Extract API for high-quality content extraction
- Returns RAG-optimized markdown content
- Automatically chunks, embeds, and stores content
- Extracts knowledge graph entities and relationships
- Logs to Langfuse if enabled

**Error Handling:**
- Returns `IngestionResult` with errors if:
  - Tavily API key not configured
  - Invalid URL format
  - Rate limit exceeded
  - Request timeout
  - Access forbidden

##### `ingest_website(url: str, max_pages: int = 10, max_depth: int = 2, config: IngestionConfig | None = None, progress_callback: Callable[[int, int], None] | None = None) -> list[IngestionResult]`

Ingest content from a website by crawling multiple pages using Tavily Crawl API.

**Requires:** `TAVILY_API_KEY` environment variable to be set.

```python
# Crawl website (default: 10 pages, depth 2)
results = await rag.ingest_website("https://docs.mongodb.com/atlas/vector-search/")

# Crawl with custom limits
results = await rag.ingest_website(
    "https://docs.mongodb.com/atlas/vector-search/",
    max_pages=5,
    max_depth=1,
)

# With progress callback
def progress(current: int, total: int) -> None:
    print(f"Processing page {current}/{total}...")

results = await rag.ingest_website(
    "https://docs.mongodb.com/atlas/vector-search/",
    max_pages=10,
    progress_callback=progress,
)

# Check results
successful = sum(1 for r in results if r.success)
total_chunks = sum(r.chunks_created for r in results)
print(f"Ingested {successful}/{len(results)} pages, {total_chunks} total chunks")
```

**Parameters:**
- `url` (str): Base URL to start crawling from (must be valid http/https URL)
- `max_pages` (int): Maximum number of pages to crawl (default: 10)
- `max_depth` (int): Maximum crawl depth (default: 2)
- `config` (IngestionConfig | None): Optional ingestion configuration
- `progress_callback` (Callable[[int, int], None] | None): Optional callback for progress updates

**Returns:** List of `IngestionResult` objects, one per crawled page

**Features:**
- Uses Tavily Crawl API for website crawling
- Returns RAG-optimized markdown content for each page
- Automatically chunks, embeds, and stores all pages
- Extracts knowledge graph entities and relationships
- Logs to Langfuse if enabled
- Supports progress tracking via callback

**Error Handling:**
- Returns list with single `IngestionResult` containing errors if:
  - Tavily API key not configured
  - Invalid URL format
  - Rate limit exceeded
  - Request timeout
  - Access forbidden
  - No pages extracted

##### `query(query: str, mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None, top_k: int | None = None, rerank_top_k: int | None = None, enable_rerank: bool | None = None, only_context: bool = False, system_prompt: str | None = None, stream: bool = False) -> str | AsyncIterator[str]`

Query without conversation memory.

```python
# Simple query
answer = await rag.query("What is this document about?")

# Query with options
answer = await rag.query(
    query="What is this document about?",
    mode="mix",
    top_k=60,
    rerank_top_k=10,
    enable_rerank=True,
)

# Get only context (no LLM generation)
context = await rag.query(
    query="What is this document about?",
    only_context=True,
)

# Stream answer chunks from the blessed stack
stream = await rag.query(
    query="What should a production RAG starter expose?",
    mode="mix",
    stream=True,
)
async for chunk in stream:
    print(chunk, end="")
```

**Parameters:**
- `query` (str): Search query
- `mode` (Literal): Query mode (default: from settings)
- `top_k` (int | None): Number of results to retrieve (default: 60)
- `rerank_top_k` (int | None): Number of results after reranking (default: 10)
- `enable_rerank` (bool | None): Whether to enable reranking (default: True)
- `only_context` (bool): If True, return only context without LLM response
- `system_prompt` (str | None): Optional system prompt for LLM
- `stream` (bool): If True, return an async iterator of answer chunks

**Returns:** Generated response string, context string, or an async iterator when `stream=True`

##### `query_with_sources(query: str, mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None, top_k: int | None = None) -> dict[str, Any]`

Query and return both response and source context.

```python
result = await rag.query_with_sources(
    query="What is this document about?",
    mode="mix",
    top_k=60,
)
```

**Returns:**
```python
{
    "answer": "The document discusses...",
    "context": "Retrieved context text...",
    "query": "What is this document about?",
    "mode": "mix",
    "references": [{"reference_id": "ref-1", "file_path": "inline://doc-1"}],
    "metadata": {"query_mode": "mix"},
}
```

##### `stream_query(query: str, mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None, top_k: int | None = None, rerank_top_k: int | None = None, enable_rerank: bool | None = None, include_context: bool = False, include_references: bool = True) -> dict[str, Any]`

Canonical streaming helper for the backend-first reference surface.

```python
result = await rag.stream_query(
    query="What should a production RAG starter expose?",
    mode="mix",
    include_context=True,
    include_references=True,
)

print(result["references"])
async for chunk in result["response_iterator"]:
    print(chunk, end="")
```

##### `query_with_memory(query: str, session_id: str, mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None, top_k: int | None = None, max_history_messages: int = 10) -> dict[str, Any]`

Query with conversation memory - enables multi-turn conversations.

```python
session_id = await rag.create_conversation_session()

result = await rag.query_with_memory(
    query="What are the key findings?",
    session_id=session_id,
    mode="mix",
    top_k=60,
    max_history_messages=10,
)
```

**Parameters:**
- `query` (str): User query
- `session_id` (str): Conversation session ID (created if doesn't exist)
- `mode` (Literal): Query mode (default: from settings)
- `top_k` (int | None): Number of results to retrieve (default: 60)
- `max_history_messages` (int): Max history messages to include (default: 10)

**Returns:**
```python
{
    "answer": "Based on the research...",
    "context": "Retrieved context text...",
    "query": "What are the key findings?",
    "session_id": "...",
    "mode": "mix",
    "history_used": 3,
}
```

##### `create_conversation_session(session_id: str | None = None, metadata: dict[str, Any] | None = None) -> str`

Create a new conversation session.

```python
# Create anonymous session (auto-generated ID)
session_id = await rag.create_conversation_session()

# Create session with custom ID
session_id = await rag.create_conversation_session(session_id="user123")

# Create session with metadata
session_id = await rag.create_conversation_session(
    session_id="user123",
    metadata={"user_id": "user123", "book": "Alice in Wonderland"},
)
```

**Returns:** Session ID string

##### `get_conversation_history(session_id: str, limit: int | None = None) -> list[dict[str, Any]]`

Get conversation history for a session.

```python
# Get all history
history = await rag.get_conversation_history(session_id)

# Get last 10 messages
history = await rag.get_conversation_history(session_id, limit=10)

# Returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

##### `clear_conversation(session_id: str) -> None`

Clear conversation history for a session (keeps the session).

```python
await rag.clear_conversation(session_id)
```

##### `get_status() -> dict[str, Any]`

Get system status information.

```python
status = await rag.get_status()
# Returns: {
#     "initialized": True,
#     "mongodb_database": "hybridrag",
#     "llm_provider": "anthropic",
#     "llm_model": "claude-sonnet-4-20250514",
#     "embedding_provider": "voyage",
#     "embedding_model": "voyage-3-large",
#     "rerank_model": "rerank-2.5",
#     "enhancements": {"implicit_expansion": True, "entity_boosting": True},
# }
```

##### `get_knowledge_base_stats() -> dict[str, Any]`

Get knowledge base statistics including documents, entities, and relationships.

```python
stats = await rag.get_knowledge_base_stats()
# Returns: {
#     "documents": {"total": 100, "by_status": {...}},
#     "entities": 200,
#     "relationships": 150,
#     "chunks": 5000,
#     "recent_documents": [...],
# }
```

##### `delete_document(doc_id: str) -> None`

Delete a document from the RAG system.

```python
await rag.delete_document("doc_id_here")
```

### `Settings`

Configuration class for HybridRAG (uses pydantic-settings, loads from environment variables).

```python
from hybridrag import Settings

# Settings are loaded from environment variables by default
# Or create with explicit values:
settings = Settings(
    # MongoDB (required)
    mongodb_uri=SecretStr("mongodb+srv://..."),
    mongodb_database="hybridrag",
    mongodb_workspace="default",

    # Voyage AI (required for embeddings)
    voyage_api_key=SecretStr("pa-..."),
    voyage_embedding_model="voyage-3-large",
    voyage_context_model="voyage-context-3",
    voyage_rerank_model="rerank-2.5",

    # Blessed-stack LLM provider
    llm_provider="openai",
    openai_api_key=SecretStr("sk-..."),
    openai_model="gpt-5.4",

    # Query defaults
    default_query_mode="mix",
    default_top_k=60,
    default_rerank_top_k=10,
    enable_rerank=True,

    # Enhancements
    enable_implicit_expansion=True,
    implicit_expansion_threshold=0.75,
    implicit_expansion_max=10,
    enable_entity_boosting=True,
    entity_boost_weight=0.2,

    # Embedding settings
    embedding_dim=1024,
    max_token_size=4096,
    embedding_batch_size=128,

    # Context limits
    max_token_for_text_unit=4000,
    max_token_for_local_context=4000,
    max_token_for_global_context=4000,

    # Observability (optional)
    langfuse_public_key="pk-lf-...",
    langfuse_secret_key=SecretStr("sk-lf-..."),
    langfuse_host="https://cloud.langfuse.com",
)
```

See [Configuration Guide](configuration.md) for all options.

## Factory Functions

### `create_hybridrag(settings: Settings = None) -> HybridRAG`

Convenience function to create HybridRAG instance.

```python
from hybridrag import create_hybridrag

rag = await create_hybridrag()
```

## Query Modes

See [Query Modes Guide](query-modes.md) for detailed explanation.

| Mode | Description |
|------|-------------|
| `mix` | KG + Vector + Keyword (recommended) |
| `local` | Entity-focused retrieval |
| `global` | Community summaries |
| `hybrid` | Local + Global combined |
| `naive` | Vector search only |
| `bypass` | Skip retrieval, direct LLM |

## Advanced Usage

### Custom Embedding Function

```python
from hybridrag import HybridRAG
from voyageai import Client

async def custom_embedding(texts: list[str]) -> list[list[float]]:
    client = Client()
    result = client.embed(texts, model="voyage-3-large")
    return result.embeddings

rag = await HybridRAG.create()
rag.embedding_func = custom_embedding
```

### Custom LLM Function

```python
from hybridrag import HybridRAG
from anthropic import AsyncAnthropic

async def custom_llm(messages: list[dict]) -> str:
    client = AsyncAnthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=messages,
    )
    return response.content[0].text

rag = await HybridRAG.create()
rag.llm_func = custom_llm
```

### Batch Queries

```python
queries = [
    "What is the main topic?",
    "Who are the key authors?",
    "What are the conclusions?",
]

results = await asyncio.gather(*[
    rag.query_with_memory(q, session_id, mode="mix")
    for q in queries
])
```

### Streaming Responses

```python
async for chunk in rag.query_stream(
    query="Explain the methodology",
    session_id=session_id,
    mode="mix",
):
    print(chunk, end="", flush=True)
```

## Error Handling

```python
from hybridrag import HybridRAGError, MongoDBConnectionError

try:
    rag = await HybridRAG.create()
    result = await rag.query("...")
except MongoDBConnectionError as e:
    print(f"MongoDB error: {e}")
except HybridRAGError as e:
    print(f"HybridRAG error: {e}")
```

## Type Hints

All functions include type hints for IDE support:

```python
from hybridrag import HybridRAG
from typing import Dict, List

async def process_query(rag: HybridRAG, query: str) -> Dict[str, any]:
    result = await rag.query(query)
    return result
```

## Examples

### Complete Example

```python
import asyncio
from hybridrag import create_hybridrag

async def main():
    # Initialize (auto-initializes by default)
    rag = await create_hybridrag()

    # Ingest documents from folder
    print("Ingesting documents...")
    results = await rag.ingest_files("research_papers/")
    print(f"Ingested {len(results)} documents")

    # Create session
    session_id = await rag.create_conversation_session()

    # Query loop
    while True:
        query = input("\nQuery (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = await rag.query_with_memory(
            query=query,
            session_id=session_id,
            mode="mix",
        )

        print(f"\nAnswer: {result['answer']}")
        print(f"\nContext length: {len(result.get('context', ''))} chars")
        print(f"History used: {result.get('history_used', 0)} messages")

asyncio.run(main())
```

### Multi-Session Example

```python
async def handle_user_query(user_id: str, query: str):
    rag = await create_hybridrag()

    # Create or reuse user session
    session_id = await rag.create_conversation_session(
        session_id=user_id,
        metadata={"user_id": user_id},
    )

    result = await rag.query_with_memory(
        query=query,
        session_id=session_id,
        mode="mix",
    )

    return result["answer"]
```

## Next Steps

- [Installation Guide](installation.md) - Set up HybridRAG
- [Configuration Guide](configuration.md) - Configure settings
- [Query Modes](query-modes.md) - Understand query modes
- [Deployment Guide](deployment.md) - Deploy to production

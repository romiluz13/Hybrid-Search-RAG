# Query Modes Explained

HybridRAG supports multiple query modes, each optimized for different use cases.

## Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    QUERY MODES                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  mix      → KG + Vector + Keyword (RECOMMENDED)              │
│  local    → Entity-focused retrieval                        │
│  global   → Community summaries                              │
│  hybrid   → Local + Global combined                          │
│  naive    → Vector + keyword fusion, without KG              │
│  bypass   → Skip retrieval, direct LLM                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Mode: `mix` (Recommended)

**Best for**: General queries, most use cases

Combines knowledge graph traversal, vector search, and keyword search using RRF (Reciprocal Rank Fusion).

```python
result = await rag.query_with_memory(
    query="What are the key findings in the research paper?",
    session_id=session_id,
    mode="mix",
)
```

**How it works:**

1. **Vector Search**: Semantic similarity search using embeddings
2. **Keyword Search**: Text matching using MongoDB text indexes
3. **Knowledge Graph**: Entity traversal via `$graphLookup`
4. **RRF Fusion**: Combines results using reciprocal rank fusion
5. **Reranking**: Voyage AI reranks top results
6. **Entity Boosting**: Chunks with relevant entities get boosted

**Use when:**

- You want the best overall results
- Query involves both concepts and specific entities
- You need comprehensive retrieval

## Mode: `local`

**Best for**: Entity-specific queries, fact lookups

Focuses on knowledge graph entities and their relationships.

```python
result = await rag.query_with_memory(
    query="What is the relationship between Alice and the Queen?",
    session_id=session_id,
    mode="local",
)
```

**How it works:**

1. **Entity Extraction**: Extracts entities from query
2. **Graph Traversal**: Finds related entities via relationships
3. **Entity Boosting**: Retrieves chunks mentioning these entities
4. **Vector Search**: Semantic search for context around entities

**Use when:**

- Query mentions specific entities (people, places, concepts)
- You need to understand relationships
- Fact-based questions

## Mode: `global`

**Best for**: High-level overviews, summaries

Uses community summaries from the knowledge graph.

```python
result = await rag.query_with_memory(
    query="Give me an overview of the document",
    session_id=session_id,
    mode="global",
)
```

**How it works:**

1. **Community Detection**: Finds entity communities in knowledge graph
2. **Summary Retrieval**: Gets pre-computed community summaries
3. **Vector Search**: Semantic search for relevant summaries

**Use when:**

- You need high-level understanding
- Query is broad/conceptual
- You want summarized information

## Mode: `hybrid`

**Best for**: Comprehensive answers combining local and global

Combines entity-focused (`local`) and summary-focused (`global`) retrieval.

```python
result = await rag.query_with_memory(
    query="Explain the main concepts and their relationships",
    session_id=session_id,
    mode="hybrid",
)
```

**How it works:**

1. Runs both `local` and `global` modes
2. Merges results with deduplication
3. Reranks combined results

**Use when:**

- You need both specific facts and overview
- Query requires comprehensive answer
- You want best of both worlds

## Mode: `naive`

**Best for**: Document-focused retrieval, metadata filtering, testing

Combines vector and keyword chunk retrieval without knowledge-graph context. Score fusion is the default; callers can explicitly select rank fusion.

```python
result = await rag.query_with_memory(
    query="Find similar documents",
    session_id=session_id,
    mode="naive",
)
```

**How it works:**

1. **Vector Search**: Semantic similarity using embeddings
2. **Keyword Search**: Exact and lexical matching over chunk content
3. **Fusion**: Rank fusion by default, optional score fusion on supported servers
4. **No KG Context**: Entity and relationship traversal are skipped
5. **Optional Filtering/Reranking**: Public metadata filters constrain both retrieval branches

**Use when:**

- Metadata filters must bound every retrieved chunk
- You want document retrieval without graph context
- Comparing vector/keyword retrieval strategies

## Mode: `bypass`

**Best for**: Testing, debugging, direct LLM queries

Skips retrieval entirely and queries LLM directly.

```python
result = await rag.query_with_memory(
    query="What is 2+2?",
    session_id=session_id,
    mode="bypass",
)
```

**How it works:**

1. **No Retrieval**: Skips all RAG steps
2. **Direct LLM**: Sends query directly to LLM
3. **Memory**: Still uses conversation memory

**Use when:**

- Testing LLM responses
- General knowledge questions (no RAG needed)
- Debugging LLM integration

## Comparison Table

| Mode | Vector | Keyword | KG | Rerank | Best For |
| ------ | -------- | --------- | ---- | ---- | ---------- |
| `mix` | ✅ | ✅ | ✅ | ✅ | General queries |
| `local` | ✅ | ❌ | ✅ | ✅ | Entity queries |
| `global` | ✅ | ❌ | ✅ | ✅ | Overview queries |
| `hybrid` | ✅ | ❌ | ✅ | ✅ | Comprehensive |
| `naive` | ✅ | ✅ | ❌ | ✅ | Filtered document retrieval |
| `bypass` | ❌ | ❌ | ❌ | ❌ | Testing/debugging |

## Performance Characteristics

| Mode | Speed | Quality | Use Cases |
| ------ | ------- | --------- | ----------- |
| `mix` | Medium | Highest | Production |
| `local` | Fast | High | Entity queries |
| `global` | Fast | Medium | Summaries |
| `hybrid` | Slow | Highest | Comprehensive |
| `naive` | Fastest | Medium | Simple search |
| `bypass` | Fastest | Variable | Testing |

## Choosing the Right Mode

**Start with `mix`** - It's the default and works best for most cases.

**Switch to `local`** if:

- Query mentions specific entities
- You need relationship information
- Results are too broad

**Switch to `global`** if:

- Query is very high-level
- You want summarized information
- Entity details aren't important

**Use `hybrid`** if:

- You need comprehensive answers
- Query requires both facts and overview
- Quality is more important than speed

**Use `naive`** if:

- You're testing embedding quality
- You want simple similarity
- Knowledge graph isn't needed

**Use `bypass`** if:

- Testing LLM integration
- General knowledge questions
- No RAG needed

## Examples

### Example 1: Research Paper Query

```python
# Best: mix mode
result = await rag.query_with_memory(
    query="What methodology did the researchers use?",
    session_id=session_id,
    mode="mix",  # Combines all retrieval methods
)
```

### Example 2: Entity Relationship Query

```python
# Best: local mode
result = await rag.query_with_memory(
    query="How is Alice related to the White Rabbit?",
    session_id=session_id,
    mode="local",  # Focuses on entity relationships
)
```

### Example 3: High-Level Overview

```python
# Best: global mode
result = await rag.query_with_memory(
    query="What is this document about?",
    session_id=session_id,
    mode="global",  # Uses community summaries
)
```

### Example 4: Comprehensive Answer

```python
# Best: hybrid mode
result = await rag.query_with_memory(
    query="Explain the main concepts and how they relate",
    session_id=session_id,
    mode="hybrid",  # Combines local + global
)
```

## Advanced: Custom Mode Configuration

You can customize retrieval parameters per mode:

```python
result = await rag.query_with_memory(
    query="...",
    session_id=session_id,
    mode="mix",
    # Override defaults
    vector_search_top_k=30,  # More vector results
    keyword_search_top_k=20,  # More keyword results
    rerank_top_k=15,  # Rerank top 15
)
```

## Next Steps

- [Installation Guide](installation.md) - Set up HybridRAG
- [Configuration Guide](configuration.md) - Configure settings
- [API Reference](api.md) - Use the Python SDK

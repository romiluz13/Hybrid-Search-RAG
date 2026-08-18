# HybridRAG Cookbook: MongoDB AI/RAG Best Practices

The definitive guide to building state-of-the-art AI applications with MongoDB Atlas.

## Overview

This cookbook showcases MongoDB's cutting-edge AI capabilities for building production-ready RAG (Retrieval-Augmented Generation) applications. Each recipe demonstrates best practices validated against MongoDB's official documentation and real-world implementations.

## Quick Navigation

| Recipe | Description | MongoDB Features |
|--------|-------------|-----------------|
| [01-hybrid-search](./01-hybrid-search.md) | Native score and rank fusion | `$scoreFusion`, `$rankFusion`, `$vectorSearch`, `$search` |
| [02-lexical-prefilters](./02-lexical-prefilters.md) | Lexical vector prefiltering | `$search.vectorSearch`, fuzzy, phrase, wildcard |
| [03-conversation-memory](./03-conversation-memory.md) | Multi-turn conversation with MongoDB | Document storage, session management |
| [04-vector-search-optimization](./04-vector-search-optimization.md) | Vector search best practices | Index tuning, numCandidates, quantization |
| [05-knowledge-graph](./05-knowledge-graph.md) | Graph-enhanced RAG | `$graphLookup`, entity relationships |
| [06-filtering-strategies](./06-filtering-strategies.md) | Three filter systems explained | Vector, Atlas, Lexical prefilters |
| [07-agent-memory-patterns](./07-agent-memory-patterns.md) | AI agent memory with MongoDB | Session storage, context management |
| [08-production-deployment](./08-production-deployment.md) | Production checklist | Indexes, scaling, monitoring |

## Core MongoDB AI Features

### 1. Native Hybrid Search

MongoDB Atlas provides native hybrid search combining:
- **Vector Search**: Semantic similarity using embeddings
- **Full-Text Search**: Keyword matching with BM25 scoring
- **Score Fusion**: Normalized, explicitly weighted score combination
- **Rank Fusion**: Reciprocal Rank Fusion as an explicit alternative

```python
# $rankFusion combines vector and text search natively
pipeline = [
    {
        "$rankFusion": {
            "input": {
                "pipelines": {
                    "vector": [{"$vectorSearch": {...}}],
                    "text": [{"$search": {...}}]
                }
            },
            "combination": {"weights": {"vector": 0.6, "text": 0.4}},
            "scoreDetails": True
        }
    }
]
```

### 2. Lexical Prefilters

Use Atlas Search operators as prefilters for vector search:

```python
# $search.vectorSearch with lexical prefilters
{
    "$search": {
        "index": "vector_index",
        "vectorSearch": {
            "queryVector": query_embedding,
            "path": "vector",
            "numCandidates": 200,
            "limit": 10,
            "filter": {
                "compound": {
                    "filter": [
                        {"text": {"path": "category", "query": "technology", "fuzzy": {"maxEdits": 2}}},
                        {"range": {"path": "timestamp", "gte": "2024-01-01"}}
                    ]
                }
            }
        }
    }
}
```

### 3. Three Filter Systems

MongoDB provides three distinct filtering approaches:

| System | Stage | Operators | Use Case |
|--------|-------|-----------|----------|
| **Vector Search Filters** | `$vectorSearch` | MQL ($eq, $gte, $in) | Simple prefiltering |
| **Atlas Search Filters** | `$search` | Atlas (range, equals) | Text search filtering |
| **Lexical Prefilters** | `$search.vectorSearch` | Atlas (text, fuzzy, phrase, wildcard, geo) | Advanced vector prefiltering |

### 4. Graph-Enhanced RAG

Combine knowledge graphs with vector search:

```python
# $graphLookup for entity relationships
{
    "$graphLookup": {
        "from": "entities",
        "startWith": "$source_entity",
        "connectFromField": "related_entities",
        "connectToField": "name",
        "as": "entity_graph",
        "maxDepth": 2
    }
}
```

## Architecture Patterns

### Recommended Stack

```
┌─────────────────────────────────────────────────────┐
│                   Your AI Application                │
├─────────────────────────────────────────────────────┤
│  HybridRAG Engine                                    │
│  ├── Voyage AI Embeddings (voyage-4-large)          │
│  ├── Voyage AI Reranking (rerank-2.5)               │
│  └── LLM (Claude, GPT, Gemini)                      │
├─────────────────────────────────────────────────────┤
│  MongoDB Atlas                                       │
│  ├── Vector Search (semantic similarity)            │
│  ├── Atlas Search (full-text, fuzzy)               │
│  ├── $rankFusion (hybrid search)                   │
│  ├── $graphLookup (knowledge graphs)               │
│  └── Document Storage (conversations, metadata)    │
└─────────────────────────────────────────────────────┘
```

### Index Configuration

Required indexes for full functionality:

```javascript
// 1. Vector Search Index (for $vectorSearch)
{
  "name": "vector_knn_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [{
      "type": "vector",
      "path": "vector",
      "numDimensions": 1024,  // Voyage AI dimension
      "similarity": "cosine"
    }]
  }
}

// 2. Atlas Search Index (for $search and lexical prefilters)
{
  "name": "default",
  "type": "search",
  "definition": {
    "mappings": {
      "dynamic": true,
      "fields": {
        "content": {"type": "string", "analyzer": "lucene.standard"},
        "vector": {"type": "vector", "numDimensions": 1024, "similarity": "cosine"}
      }
    }
  }
}

// 3. Text Search Index (for full-text search)
{
  "name": "text_search_index",
  "type": "search",
  "definition": {
    "mappings": {
      "fields": {
        "content": {"type": "string", "analyzer": "lucene.standard"}
      }
    }
  }
}
```

## Best Practices

### 1. numCandidates Calculation

Always use 10-20x the limit for good recall:

```python
def calculate_num_candidates(top_k: int, multiplier: int = 20) -> int:
    """MongoDB best practice: numCandidates = 10-20x limit."""
    return top_k * multiplier

# Example: For top_k=10, use numCandidates=200
```

### 2. Score Details for Debugging

Always enable `scoreDetails` in $rankFusion:

```python
{
    "$rankFusion": {
        ...
        "scoreDetails": True  # Critical for debugging
    }
},
{
    "$addFields": {
        "hybrid_score": {"$meta": "score"},
        "score_details": {"$meta": "scoreDetails"}
    }
}
```

### 3. Fail-Closed Capability Handling

Keep the requested evidence boundary and fusion algorithm intact. Capability
errors are explicit; callers decide whether to retry with a different strategy:

```python
from hybridrag.engine.exceptions import RetrievalCapabilityError

try:
    return await rag.query_data(query, mode="naive", fusion_strategy="score")
except RetrievalCapabilityError:
    # Surface the deployment capability problem. Never silently change the
    # fusion algorithm or discard one of the evidence branches.
    raise
```

## Feature Availability

HybridRAG does not maintain a numeric version or Atlas-tier matrix. It executes
the requested native capability and returns a typed capability error when the
connected deployment does not support or has not enabled it. This keeps the
library current as MongoDB deployment capabilities evolve.

## Getting Started

1. **Install HybridRAG**:
   ```bash
   pip install hybridrag
   ```

2. **Configure Environment**:
   ```bash
   export MONGODB_URI="mongodb+srv://..."
   export VOYAGE_API_KEY="pa-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. **Initialize and Query**:
   ```python
   from hybridrag import HybridRAG

   rag = HybridRAG()
   await rag.initialize()

   # Ingest documents
   await rag.ingest_files("./docs")

   # Query with hybrid search
   response = await rag.query("What are the key features?", mode="mix")
   ```

## References

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [MongoDB $rankFusion Documentation](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/)
- [Lexical Prefilters Blog Post](https://www.mongodb.com/company/blog/product-release-announcements/semantic-power-lexical-precision-advanced-filtering-for-vector-search)
- [Voyage AI Documentation](https://docs.voyageai.com/)

---

**Last Updated**: January 2026
**MongoDB Version**: 8.2
**HybridRAG Version**: 0.1.0

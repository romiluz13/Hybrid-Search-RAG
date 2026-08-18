# Recipe 06: Three Filter Systems Explained

Master MongoDB's filtering capabilities for vector search, text search, and hybrid search.

## Overview

MongoDB provides **three distinct filtering systems**, each with different operators and use cases. Understanding when to use each is critical for optimal performance.

## Filter Systems Comparison

| System | Stage | Operator Syntax | Supports |
|--------|-------|-----------------|----------|
| **Vector Search Filters** | `$vectorSearch` | MQL ($eq, $in, $gte) | Basic metadata filtering |
| **Atlas Search Filters** | `$search` compound | Atlas (range, equals) | Text search with filters |
| **Lexical Prefilters** | `$search.vectorSearch` | Atlas (text, fuzzy, geo) | Advanced vector prefiltering |

## 1. Vector Search Filters

### When to Use

- Simple metadata filtering with `$vectorSearch`
- Exact value matching
- Numeric range queries
- Array membership checks

### Supported Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"category": {"$eq": "docs"}}` |
| `$ne` | Not equals | `{"status": {"$ne": "draft"}}` |
| `$gt`, `$gte` | Greater than | `{"score": {"$gte": 0.5}}` |
| `$lt`, `$lte` | Less than | `{"date": {"$lt": "2024-01-01"}}` |
| `$in` | In array | `{"type": {"$in": ["a", "b"]}}` |
| `$nin` | Not in array | `{"status": {"$nin": ["deleted"]}}` |
| `$and`, `$or` | Logical | Combine conditions |

### Implementation

```python
from hybridrag.enhancements.filters import (
    VectorSearchFilterConfig,
    build_vector_search_filters
)

config = VectorSearchFilterConfig(
    # Simple equality
    equality_filters={
        "metadata.category": "documentation",
        "metadata.language": "en"
    },
    # Array membership
    in_filters={
        "metadata.tags": ["mongodb", "vector-search"]
    },
    # Numeric/date ranges
    range_filters={
        "metadata.timestamp": {
            "$gte": "2024-01-01T00:00:00Z",
            "$lt": "2024-12-31T23:59:59Z"
        },
        "metadata.score": {
            "$gte": 0.5
        }
    },
    # Timestamp field (special handling)
    timestamp_field="metadata.created_at",
    timestamp_start="2024-01-01",
    timestamp_end="2024-06-30"
)

filters = build_vector_search_filters(config)

# Use in $vectorSearch
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_knn_index",
            "path": "vector",
            "queryVector": query_embedding,
            "numCandidates": 200,
            "limit": 10,
            "filter": filters  # MQL operators
        }
    }
]
```

### Generated Filter Structure

```javascript
// Result from build_vector_search_filters()
{
  "$and": [
    {"metadata.category": {"$eq": "documentation"}},
    {"metadata.language": {"$eq": "en"}},
    {"metadata.tags": {"$in": ["mongodb", "vector-search"]}},
    {"metadata.timestamp": {"$gte": "2024-01-01T00:00:00Z", "$lt": "2024-12-31T23:59:59Z"}},
    {"metadata.score": {"$gte": 0.5}},
    {"metadata.created_at": {"$gte": ISODate("2024-01-01"), "$lte": ISODate("2024-06-30")}}
  ]
}
```

### Index Requirements

```javascript
// Vector search index with filter fields
{
  "name": "vector_knn_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {"type": "vector", "path": "vector", "numDimensions": 1024, "similarity": "cosine"},
      {"type": "filter", "path": "metadata.category"},
      {"type": "filter", "path": "metadata.language"},
      {"type": "filter", "path": "metadata.tags"},
      {"type": "filter", "path": "metadata.timestamp"},
      {"type": "filter", "path": "metadata.score"}
    ]
  }
}
```

## 2. Atlas Search Filters

### When to Use

- Full-text search with `$search`
- Text search combined with metadata filters
- Compound queries with must/should/filter

### Supported Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `equals` | Exact match | `{"path": "x", "value": "y"}` |
| `range` | Range query | `{"path": "x", "gte": 0, "lte": 10}` |
| `exists` | Field exists | `{"path": "x"}` |
| `near` | Geo near | `{"path": "x", "origin": {...}}` |

### Implementation

```python
from hybridrag.enhancements.filters import (
    AtlasSearchFilterConfig,
    build_atlas_search_filters
)

config = AtlasSearchFilterConfig(
    # Equality filters
    equality_filters={
        "metadata.source": "official_docs",
        "metadata.status": "published"
    },
    # Range filters (Atlas syntax: gte, gt, lte, lt - no $)
    range_filters={
        "metadata.timestamp": {
            "gte": "2024-01-01T00:00:00Z",
            "lt": "2025-01-01T00:00:00Z"
        },
        "metadata.version": {
            "gte": 8.0
        }
    }
)

filters = build_atlas_search_filters(config)

# Use in $search compound query
pipeline = [
    {
        "$search": {
            "index": "text_search_index",
            "compound": {
                "must": [
                    {"text": {"query": "vector search", "path": "content"}}
                ],
                "filter": filters  # Atlas operators
            }
        }
    }
]
```

### Generated Filter Structure

```javascript
// Result from build_atlas_search_filters()
[
  {"equals": {"path": "metadata.source", "value": "official_docs"}},
  {"equals": {"path": "metadata.status", "value": "published"}},
  {"range": {"path": "metadata.timestamp", "gte": "2024-01-01T00:00:00Z", "lt": "2025-01-01T00:00:00Z"}},
  {"range": {"path": "metadata.version", "gte": 8.0}}
]
```

### Index Requirements

```javascript
// Atlas Search index with filterable fields
{
  "name": "text_search_index",
  "type": "search",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "content": {"type": "string", "analyzer": "lucene.standard"},
        "metadata": {
          "type": "document",
          "fields": {
            "source": {"type": "token"},
            "status": {"type": "token"},
            "timestamp": {"type": "date"},
            "version": {"type": "number"}
          }
        }
      }
    }
  }
}
```

## 3. Lexical Prefilters (MongoDB 8.2+)

### When to Use

- Vector search with advanced text filtering
- Fuzzy matching before vector similarity
- Phrase or wildcard patterns
- Geospatial constraints on vector search

### Supported Operators

| Operator | Description | Use Case |
|----------|-------------|----------|
| `text` | Text search | Keyword matching |
| `fuzzy` | Typo-tolerant | User input with errors |
| `phrase` | Exact phrase | Multi-word terms |
| `wildcard` | Pattern match | File patterns, codes |
| `geoWithin` | Geo filter | Location-based |
| `queryString` | Lucene syntax | Power users |
| `range` | Range filter | Dates, numbers |
| `equals` | Exact match | Exact values |

### Implementation

```python
from hybridrag.enhancements.filters import (
    LexicalPrefilterConfig,
    TextFilter,
    FuzzyFilter,
    PhraseFilter,
    WildcardFilter,
    GeoFilter,
    QueryStringFilter,
    build_lexical_prefilters
)

config = LexicalPrefilterConfig(
    # Text with optional fuzzy
    text_filters=[
        TextFilter(
            path="content",
            query="mongodb",
            fuzzy={"maxEdits": 1}
        )
    ],

    # Fuzzy matching
    fuzzy_filters=[
        FuzzyFilter(
            path="title",
            query="authetication",  # Typo
            maxEdits=2,
            prefixLength=3
        )
    ],

    # Exact phrase
    phrase_filters=[
        PhraseFilter(
            path="content",
            query="vector search index",
            slop=1  # Allow 1 word gap
        )
    ],

    # Wildcard patterns
    wildcard_filters=[
        WildcardFilter(
            path="filename",
            query="config*.json"
        )
    ],

    # Geospatial
    geo_filters=[
        GeoFilter(
            path="location",
            geometry={
                "type": "Circle",
                "coordinates": [-73.99, 40.75],
                "radius": 1000
            }
        )
    ],

    # Lucene query string
    query_string_filter=QueryStringFilter(
        defaultPath="content",
        query='mongodb AND "vector search" -deprecated'
    ),

    # Range filters
    range_filters={
        "metadata.date": {"gte": "2024-01-01", "lt": "2025-01-01"}
    },

    # Equality
    equality_filters={
        "metadata.language": "en"
    }
)

filters = build_lexical_prefilters(config)

# Use in $search.vectorSearch
pipeline = [
    {
        "$search": {
            "index": "default",
            "vectorSearch": {
                "queryVector": query_embedding,
                "path": "vector",
                "numCandidates": 200,
                "limit": 10,
                "filter": filters  # Atlas Search operators as prefilters
            }
        }
    }
]
```

### Generated Filter Structure

```javascript
// Result from build_lexical_prefilters()
{
  "compound": {
    "filter": [
      {"text": {"path": "content", "query": "mongodb", "fuzzy": {"maxEdits": 1}}},
      {"text": {"path": "title", "query": "authetication", "fuzzy": {"maxEdits": 2, "prefixLength": 3, "maxExpansions": 50}}},
      {"phrase": {"path": "content", "query": "vector search index", "slop": 1}},
      {"wildcard": {"path": "filename", "query": "config*.json"}},
      {"geoWithin": {"path": "location", "geometry": {...}}},
      {"queryString": {"defaultPath": "content", "query": "mongodb AND \"vector search\" -deprecated"}},
      {"range": {"path": "metadata.date", "gte": "2024-01-01", "lt": "2025-01-01"}},
      {"equals": {"path": "metadata.language", "value": "en"}}
    ]
  }
}
```

### Index Requirements

```javascript
// Atlas Search index with vector field (for lexical prefilters)
{
  "name": "default",
  "type": "search",
  "definition": {
    "mappings": {
      "dynamic": true,
      "fields": {
        "vector": {"type": "vector", "numDimensions": 1024, "similarity": "cosine"},
        "content": {"type": "string", "analyzer": "lucene.standard"},
        "title": {"type": "string", "analyzer": "lucene.standard"},
        "filename": {"type": "string"},
        "location": {"type": "geo"},
        "metadata": {
          "type": "document",
          "fields": {
            "date": {"type": "date"},
            "language": {"type": "token"}
          }
        }
      }
    }
  }
}
```

## Decision Matrix

### Which Filter System to Use?

| Scenario | Recommended System | Reason |
|----------|-------------------|--------|
| Simple metadata + vector search | Vector Search Filters | Native, performant |
| Text search + metadata | Atlas Search Filters | Compound queries |
| Fuzzy + vector search | Lexical Prefilters | Advanced text analysis |
| Geo + vector search | Lexical Prefilters | Only option for geo |
| Hybrid search + filters | Both | Vector and Atlas filters |
| Pattern matching | Lexical Prefilters | Wildcard support |
| Multi-language typos | Lexical Prefilters | Fuzzy with control |

### Performance Considerations

| System | Pre-filter Speed | Filter Expressiveness | Index Size |
|--------|-----------------|----------------------|------------|
| Vector Search | Fast | Limited | Small |
| Atlas Search | Medium | High | Medium |
| Lexical Prefilters | Fast | Highest | Medium |

## Using Multiple Filter Systems Together

### Hybrid Search with Both Filters

```python
from hybridrag.enhancements import (
    hybrid_search_with_rank_fusion,
    MongoDBHybridSearchConfig,
)
from hybridrag.enhancements.filters import (
    VectorSearchFilterConfig,
    AtlasSearchFilterConfig,
    LexicalPrefilterConfig,
)

# Vector search uses VectorSearchFilterConfig OR LexicalPrefilterConfig
vector_config = VectorSearchFilterConfig(
    equality_filters={"metadata.category": "docs"}
)

# OR for MongoDB 8.2+:
lexical_config = LexicalPrefilterConfig(
    fuzzy_filters=[{"path": "title", "query": "mongodb", "maxEdits": 2}]
)

# Atlas search uses AtlasSearchFilterConfig
atlas_config = AtlasSearchFilterConfig(
    equality_filters={"metadata.source": "official"},
    range_filters={"metadata.date": {"gte": "2024-01-01"}}
)

# Hybrid search with appropriate filters
results = await hybrid_search_with_rank_fusion(
    collection=chunks_collection,
    query_text="vector search",
    query_vector=query_embedding,
    top_k=10,
    config=MongoDBHybridSearchConfig(use_lexical_prefilters=True),
    vector_filter_config=vector_config,    # For vector pipeline
    atlas_filter_config=atlas_config,      # For text pipeline
    lexical_filter_config=lexical_config,  # For lexical prefilters
)
```

## Common Mistakes

### 1. Wrong Operator Syntax

```python
# WRONG: Using MQL operators in Atlas Search
atlas_config = AtlasSearchFilterConfig(
    range_filters={"date": {"$gte": "2024-01-01"}}  # $gte won't work!
)

# CORRECT: Use Atlas syntax (no $ prefix)
atlas_config = AtlasSearchFilterConfig(
    range_filters={"date": {"gte": "2024-01-01"}}  # Correct
)
```

### 2. Missing Index Fields

```python
# WRONG: Filter on field not in vector index
{
    "$vectorSearch": {
        "filter": {"unindexed_field": {"$eq": "value"}}  # Will fail!
    }
}

# CORRECT: Add filter field to index definition
{
    "fields": [
        {"type": "vector", "path": "vector", ...},
        {"type": "filter", "path": "unindexed_field"}  # Add this
    ]
}
```

### 3. Handling Lexical-Prefilter Capability Errors

```python
# Execute the requested capability. HybridRAG raises a typed capability error
# if the deployment does not support it; it never silently substitutes a
# different filter language or evidence boundary.
await vector_search_with_lexical_prefilters(...)
```

## Summary

| Task | Use |
|------|-----|
| Basic vector search filtering | `VectorSearchFilterConfig` |
| Text search with filters | `AtlasSearchFilterConfig` |
| Advanced vector prefiltering | `LexicalPrefilterConfig` |
| Fuzzy, geo, wildcard | `LexicalPrefilterConfig` |
| Hybrid search | Combine appropriately |

---

**Previous**: [Recipe 05: Knowledge Graph](./05-knowledge-graph.md)
**Next**: [Recipe 07: Agent Memory Patterns](./07-agent-memory-patterns.md)

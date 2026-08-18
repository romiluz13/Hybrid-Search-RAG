# Recipe 02: Lexical Prefilters for Vector Search

Unlock advanced text and geospatial filtering with MongoDB's newest vector search capability.

## Overview

Lexical Prefilters enable Atlas Search operators as prefilters for vector search. They support applications that need:

- Fuzzy text matching before vector search
- Phrase and wildcard filtering
- Geospatial constraints
- Complex boolean logic with Lucene syntax

## The Three Filter Systems

MongoDB now provides **three distinct filtering approaches**:

| System | Stage | Operators | When to Use |
|--------|-------|-----------|-------------|
| **Vector Search Filters** | `$vectorSearch` | MQL ($eq, $gte, $in, $and) | Simple metadata filtering |
| **Atlas Search Filters** | `$search` compound | Atlas (range, equals) | Text search with filters |
| **Lexical Prefilters** | `$search.vectorSearch` | Atlas (text, fuzzy, phrase, wildcard, geo) | Advanced vector prefiltering |

## Why Lexical Prefilters Matter

### Standard vector metadata filters

```python
# Standard $vectorSearch only supports basic MQL operators
{
    "$vectorSearch": {
        "filter": {
            "category": {"$eq": "technology"},  # Limited to exact match
            "date": {"$gte": "2024-01-01"}       # No fuzzy, phrase, or geo
        }
    }
}
```

### Lexical vector prefilters

```python
# $search.vectorSearch supports full Atlas Search operators
{
    "$search": {
        "index": "vector_index",
        "vectorSearch": {
            "queryVector": [...],
            "path": "vector",
            "filter": {
                "compound": {
                    "filter": [
                        {
                            "text": {
                                "path": "category",
                                "query": "tehnology",  # Typo!
                                "fuzzy": {"maxEdits": 2}  # Still matches "technology"
                            }
                        },
                        {
                            "geoWithin": {
                                "path": "location",
                                "geometry": {...}  # Geospatial filtering
                            }
                        }
                    ]
                }
            }
        }
    }
}
```

## Supported Filter Types

### 1. Text Filter (with optional fuzzy)

```python
from hybridrag.enhancements.filters import TextFilter, LexicalPrefilterConfig

config = LexicalPrefilterConfig(
    text_filters=[
        TextFilter(
            path="content",
            query="authentication",
            fuzzy={"maxEdits": 2, "prefixLength": 3}  # Optional
        )
    ]
)
```

### 2. Fuzzy Filter

Typo-tolerant matching for user queries:

```python
from hybridrag.enhancements.filters import FuzzyFilter

config = LexicalPrefilterConfig(
    fuzzy_filters=[
        FuzzyFilter(
            path="title",
            query="authetication",  # Typo
            maxEdits=2,             # Up to 2 character changes
            prefixLength=3,         # First 3 chars must match
            maxExpansions=50        # Max term expansions
        )
    ]
)
```

### 3. Phrase Filter

Exact phrase matching with optional word distance (slop):

```python
from hybridrag.enhancements.filters import PhraseFilter

config = LexicalPrefilterConfig(
    phrase_filters=[
        PhraseFilter(
            path="content",
            query="machine learning model",
            slop=2  # Words can be up to 2 positions apart
        )
    ]
)
```

### 4. Wildcard Filter

Pattern matching with `*` and `?`:

```python
from hybridrag.enhancements.filters import WildcardFilter

config = LexicalPrefilterConfig(
    wildcard_filters=[
        WildcardFilter(
            path="filename",
            query="config*.json",  # Matches config.json, config-dev.json
            allowAnalyzedField=False
        )
    ]
)
```

### 5. Geo Filter

Geospatial filtering for location-based searches:

```python
from hybridrag.enhancements.filters import GeoFilter

config = LexicalPrefilterConfig(
    geo_filters=[
        GeoFilter(
            path="location",
            relation="within",
            geometry={
                "type": "Polygon",
                "coordinates": [[
                    [-73.99, 40.75],
                    [-73.98, 40.75],
                    [-73.98, 40.76],
                    [-73.99, 40.76],
                    [-73.99, 40.75]
                ]]
            }
        )
    ]
)
```

### 6. QueryString Filter

Full Lucene query syntax for power users:

```python
from hybridrag.enhancements.filters import QueryStringFilter

config = LexicalPrefilterConfig(
    query_string_filter=QueryStringFilter(
        defaultPath="content",
        query='(mongodb OR atlas) AND "vector search" -deprecated'
        # Lucene syntax: OR, AND, NOT (-), phrases, grouping
    )
)
```

### 7. Range Filter

Numeric and date range filtering:

```python
config = LexicalPrefilterConfig(
    range_filters={
        "metadata.timestamp": {
            "gte": "2024-01-01T00:00:00Z",
            "lt": "2024-12-31T23:59:59Z"
        },
        "metadata.score": {
            "gte": 0.8,
            "lte": 1.0
        }
    }
)
```

### 8. Equality Filter

Simple value matching:

```python
config = LexicalPrefilterConfig(
    equality_filters={
        "metadata.source": "official_docs",
        "metadata.language": "en"
    }
)
```

## Complete Implementation

### Building the $search.vectorSearch Stage

```python
from hybridrag.enhancements.filters import (
    LexicalPrefilterConfig,
    build_lexical_prefilters,
    build_search_vector_search_stage,
)

# Configure all filters
config = LexicalPrefilterConfig(
    text_filters=[
        {"path": "category", "query": "technology"}
    ],
    fuzzy_filters=[
        {"path": "title", "query": "authetication", "maxEdits": 2}
    ],
    range_filters={
        "metadata.timestamp": {"gte": "2024-01-01"}
    },
    equality_filters={
        "metadata.language": "en"
    }
)

# Build the complete $search stage
search_stage = build_search_vector_search_stage(
    index_name="default",  # Must be an Atlas Search index with vector field
    query_vector=query_embedding,
    vector_path="vector",
    limit=10,
    num_candidates=200,
    filter_config=config
)

# Result:
# {
#     "$search": {
#         "index": "default",
#         "vectorSearch": {
#             "queryVector": [...],
#             "path": "vector",
#             "numCandidates": 200,
#             "limit": 10,
#             "filter": {
#                 "compound": {
#                     "filter": [
#                         {"text": {"path": "category", "query": "technology"}},
#                         {"text": {"path": "title", "query": "authetication", "fuzzy": {...}}},
#                         {"range": {"path": "metadata.timestamp", "gte": "2024-01-01"}},
#                         {"equals": {"path": "metadata.language", "value": "en"}}
#                     ]
#                 }
#             }
#         }
#     }
# }
```

### Using with HybridRAG

```python
from hybridrag.enhancements import (
    vector_search_with_lexical_prefilters,
    MongoDBHybridSearchConfig,
    LexicalPrefilterConfig,
)

# Configure search
config = MongoDBHybridSearchConfig(
    use_lexical_prefilters=True,
    lexical_prefilter_index="default"
)

# Configure filters
filter_config = LexicalPrefilterConfig(
    fuzzy_filters=[
        {"path": "content", "query": user_query, "maxEdits": 2}
    ],
    equality_filters={
        "metadata.category": "documentation"
    }
)

# Execute search
results = await vector_search_with_lexical_prefilters(
    collection=chunks_collection,
    query_vector=query_embedding,
    top_k=10,
    config=config,
    lexical_filter_config=filter_config
)
```

### Hybrid Search with Lexical Prefilters

```python
from hybridrag.enhancements import hybrid_search_with_rank_fusion

# Lexical prefilters work within $rankFusion
results = await hybrid_search_with_rank_fusion(
    collection=chunks_collection,
    query_text=user_query,
    query_vector=query_embedding,
    top_k=10,
    config=MongoDBHybridSearchConfig(use_lexical_prefilters=True),
    lexical_filter_config=LexicalPrefilterConfig(
        fuzzy_filters=[{"path": "title", "query": user_query, "maxEdits": 2}]
    )
)
```

## Index Requirements

Lexical prefilters require an Atlas Search index with a `vector` field type:

```javascript
{
  "name": "default",
  "type": "search",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        // Vector field for similarity search
        "vector": {
          "type": "vector",
          "numDimensions": 1024,
          "similarity": "cosine"
        },
        // Text fields for lexical filtering
        "content": {
          "type": "string",
          "analyzer": "lucene.standard"
        },
        "title": {
          "type": "string",
          "analyzer": "lucene.standard"
        },
        "category": {
          "type": "string"
        },
        // Geo field for geospatial filtering
        "location": {
          "type": "geo"
        },
        // Date field for range filtering
        "metadata.timestamp": {
          "type": "date"
        }
      }
    }
  }
}
```

## Migration from knnBeta

> **Important**: The `knnVector` field type and `$search.knnBeta` operator are deprecated. Migrate to the new `vector` field type and `$search.vectorSearch` operator.

### Before (Deprecated)

```javascript
// OLD: knnVector field type
{
  "mappings": {
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1024,
        "similarity": "cosine"
      }
    }
  }
}

// OLD: $search.knnBeta operator
{
  "$search": {
    "knnBeta": {
      "vector": [...],
      "path": "embedding",
      "k": 10
    }
  }
}
```

### After (Current)

```javascript
// NEW: vector field type
{
  "mappings": {
    "fields": {
      "embedding": {
        "type": "vector",
        "numDimensions": 1024,
        "similarity": "cosine"
      }
    }
  }
}

// NEW: $search.vectorSearch operator
{
  "$search": {
    "vectorSearch": {
      "queryVector": [...],
      "path": "embedding",
      "numCandidates": 200,
      "limit": 10
    }
  }
}
```

## Benefits of Lexical Prefilters

| Benefit | Description |
|---------|-------------|
| **Better Performance** | Narrow candidates BEFORE vector search |
| **Higher Precision** | Hard filters enforced before ranking |
| **Lower Cost** | Fewer vector comparisons = less compute |
| **Advanced Filtering** | Fuzzy, phrase, wildcard, geo support |
| **Complex Logic** | Boolean queries with Lucene syntax |

## Use Cases

### 1. Typo-Tolerant RAG

```python
# User types "configration" instead of "configuration"
config = LexicalPrefilterConfig(
    fuzzy_filters=[
        {"path": "content", "query": "configration", "maxEdits": 2}
    ]
)
# Still finds documents about "configuration"
```

### 2. Location-Based Search

```python
# Find nearby restaurants semantically similar to query
config = LexicalPrefilterConfig(
    geo_filters=[
        {
            "path": "location",
            "geometry": {
                "type": "Point",
                "coordinates": [-73.99, 40.75]
            },
            "relation": "near"
        }
    ]
)
```

### 3. Time-Bounded RAG

```python
# Only search recent documents
config = LexicalPrefilterConfig(
    range_filters={
        "metadata.created_at": {
            "gte": (datetime.now() - timedelta(days=30)).isoformat()
        }
    }
)
```

### 4. Category + Fuzzy Search

```python
# Find in specific category with fuzzy query
config = LexicalPrefilterConfig(
    equality_filters={"category": "api-docs"},
    fuzzy_filters=[
        {"path": "content", "query": user_query, "maxEdits": 2}
    ]
)
```

## Failure behavior

The low-level builder raises when a filter cannot be compiled or executed. It does not retry without lexical predicates or replace the requested operation with ordinary vector search.

## References

- [MongoDB Lexical Prefilters Blog](https://www.mongodb.com/company/blog/product-release-announcements/semantic-power-lexical-precision-advanced-filtering-for-vector-search)
- [vectorSearch Operator Documentation](https://www.mongodb.com/docs/atlas/atlas-search/operators-collectors/vectorSearch/)
- [Vector Search Index Definition](https://www.mongodb.com/docs/atlas/atlas-search/field-types/vector-type/)

---

**Previous**: [Recipe 01: Hybrid Search](./01-hybrid-search.md)
**Next**: [Recipe 03: Conversation Memory](./03-conversation-memory.md)

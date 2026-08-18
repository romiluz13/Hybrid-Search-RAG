# ADR 0006: MongoDB 8.2 Lexical Prefilters

**Status:** Superseded by [ADR 0008](0008-latest-first-search-capabilities.md). Historical version and fallback statements below are not current product policy.

**Status:** Accepted
**Date:** 2026-01-21
**Context:** MongoDB 8.2 (November 2025) introduced `$search.vectorSearch` operator

## Context

MongoDB 8.2 introduced a NEW vector search syntax `$search.vectorSearch` that enables Atlas Search operators (text, fuzzy, phrase, wildcard, geoWithin) as prefilters for vector similarity search.

### Before (MongoDB 8.0)
```javascript
// $vectorSearch with MQL prefilters only
{
  "$vectorSearch": {
    "queryVector": [...],
    "path": "vector",
    "filter": {
      "category": {"$eq": "tech"},  // Only MQL operators
      "timestamp": {"$gte": ISODate("2024-01-01")}
    }
  }
}
```

### After (MongoDB 8.2)
```javascript
// $search.vectorSearch with Atlas Search prefilters
{
  "$search": {
    "index": "vector_index",
    "vectorSearch": {
      "queryVector": [...],
      "path": "vector",
      "filter": {  // Atlas Search operators!
        "compound": {
          "filter": [
            {"text": {"path": "category", "query": "technology"}},
            {"fuzzy": {"path": "title", "query": "mongoDB", "maxEdits": 2}},
            {"range": {"path": "timestamp", "gte": "2024-01-01"}}
          ]
        }
      }
    }
  }
}
```

## Decision

Implement a **third filter system** in HybridRAG to support MongoDB 8.2 lexical prefilters:

1. **LexicalPrefilterConfig** - Configuration dataclass for Atlas Search operators
2. **build_lexical_prefilters()** - Builder function for `$search.vectorSearch` filters
3. **vector_search_with_lexical_prefilters()** - Vector search function using new syntax
4. **Graceful fallback** - Falls back to `$vectorSearch` if `$search.vectorSearch` unavailable

## Three Filter Systems

| System | Operators | Used In | MongoDB Version |
|--------|-----------|---------|-----------------|
| VectorSearchFilterConfig | MQL ($eq, $gte, $in) | `$vectorSearch` | 8.0+ |
| AtlasSearchFilterConfig | Atlas (range, equals) | `$search` compound | All |
| **LexicalPrefilterConfig** | **Atlas (text, fuzzy, phrase, wildcard, geo)** | **`$search.vectorSearch`** | **8.2+** |

## Benefits

1. **Complex text matching** - Fuzzy matching, phrase matching, wildcards
2. **Geospatial filtering** - Filter by location before vector search
3. **Lucene queries** - Support queryString operator for complex boolean logic
4. **Performance** - Narrow candidates BEFORE ANN, improving both speed and precision
5. **Backward compatible** - Defaults to `use_lexical_prefilters=False`

## Implementation

### Core Files Created
- `src/hybridrag/enhancements/filters/lexical_prefilters.py` - Config and builder
- `tests/enhancements/test_lexical_prefilters.py` - 14 unit tests

### Integrated Into
- `MongoDBHybridSearchConfig` - Added `use_lexical_prefilters` flag
- `hybrid_search_with_rank_fusion()` - Supports lexical prefilters in vector pipeline
- `MixModeConfig` - Added `default_lexical_prefilter` option
- `mix_mode_search()` - Passes lexical filters to hybrid search

### Exports
```python
from hybridrag import (
    LexicalPrefilterConfig,
    build_lexical_prefilters,
    build_search_vector_search_stage,
    vector_search_with_lexical_prefilters,
    TextFilter,
    FuzzyFilter,
    PhraseFilter,
    WildcardFilter,
    GeoFilter,
    QueryStringFilter,
)
```

## Usage Example

```python
from hybridrag import (
    HybridRAG,
    LexicalPrefilterConfig,
    MongoDBHybridSearchConfig,
)

# Enable lexical prefilters
hybrid_config = MongoDBHybridSearchConfig(
    use_lexical_prefilters=True,
    lexical_prefilter_index="vector_index",
)

# Create prefilter config
lexical_filter = LexicalPrefilterConfig(
    text_filters=[{"path": "category", "query": "technology"}],
    fuzzy_filters=[{"path": "title", "query": "mongoDB", "maxEdits": 2}],
    range_filters={"timestamp": {"gte": "2024-01-01"}},
)

# Use in RAG query
rag = await HybridRAG.from_config(hybrid_config=hybrid_config)
results = await rag.query(
    "How does MongoDB handle vector search?",
    lexical_filter_config=lexical_filter,
)
```

## Consequences

### Positive
- Enables advanced text filtering not possible with MQL operators
- Maintains backward compatibility (opt-in feature)
- Graceful fallback to legacy syntax if MongoDB < 8.2
- Better performance for narrow, filtered searches

### Negative
- Adds complexity with third filter system
- Requires MongoDB 8.2+ for full functionality
- Developers must understand three different filter syntaxes

## References
- [MongoDB Atlas Search vectorSearch Operator](https://www.mongodb.com/docs/atlas/atlas-search/operators-collectors/vectorSearch/)
- MongoDB 8.2 Release Notes (November 2025)
- ADR 0005: Filter Builder Systems

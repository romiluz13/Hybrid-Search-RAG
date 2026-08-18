# Recipe 01: Native Hybrid Search

Use HybridRAG's canonical vector-and-text retrieval. Score fusion is the default; rank fusion is explicit.

## Overview

Hybrid search addresses a fundamental limitation: neither vector search nor keyword search alone provides optimal results for all queries. By combining both approaches with Reciprocal Rank Fusion (RRF), you get the best of both worlds.

## The Problem

| Query Type | Vector Search | Text Search | Hybrid |
|------------|--------------|-------------|--------|
| "smart thermostat" | Finds semantic matches | Exact match | Both |
| "it's too warm at home" | Understands intent | No matches | Vector helps |
| "themrostat" (typo) | No match | Fuzzy finds it | Text helps |
| Technical acronyms | May miss context | Exact match | Text helps |

## Native fusion

MongoDB provides native score and rank fusion. HybridRAG executes the requested strategy without substituting another algorithm after an error. `$rankFusion`:
1. Runs vector and text search pipelines in parallel
2. Applies Reciprocal Rank Fusion algorithm
3. Returns unified, relevance-ranked results

### The RRF Formula

```
score(d) = Σ (1 / (k + rank_i(d)))
```

Where:
- `k` = 60 (default constant, configurable)
- `rank_i(d)` = position of document d in result list i

## Implementation

### Basic Hybrid Search

```python
results = await rag.query_data(
    "How do I configure authentication?",
    mode="naive",
    fusion_strategy="score",  # default; use "rank" for reciprocal-rank fusion
)
```

The lower-level enhancement helpers are pipeline utilities. Application retrieval should use `HybridRAG.query()`, `query_data()`, or `query_with_sources()` so filters, tenant constraints, cache identity, and typed errors share one contract.

### Full Pipeline Example

```python
# The complete $rankFusion pipeline
pipeline = [
    {
        "$rankFusion": {
            "input": {
                "pipelines": {
                    # Pipeline 1: Vector Search (semantic similarity)
                    "vector": [
                        {
                            "$vectorSearch": {
                                "index": "vector_knn_index",
                                "path": "vector",
                                "queryVector": query_embedding,
                                "numCandidates": 200,  # 20x limit for good recall
                                "limit": 20,
                            }
                        }
                    ],
                    # Pipeline 2: Full-Text Search (keyword matching)
                    "text": [
                        {
                            "$search": {
                                "index": "text_search_index",
                                "compound": {
                                    "must": [{
                                        "text": {
                                            "query": query_text,
                                            "path": "content",
                                            "fuzzy": {
                                                "maxEdits": 2,
                                                "prefixLength": 3
                                            }
                                        }
                                    }]
                                }
                            }
                        },
                        {"$limit": 20}
                    ]
                }
            },
            # Configurable weights for each pipeline
            "combination": {
                "weights": {
                    "vector": 0.6,
                    "text": 0.4
                }
            },
            # CRITICAL: Enable for debugging
            "scoreDetails": True
        }
    },
    # Extract scores for analysis
    {
        "$addFields": {
            "hybrid_score": {"$meta": "score"},
            "score_details": {"$meta": "scoreDetails"}
        }
    },
    {"$limit": 10},
    {"$project": {"vector": 0}}  # Exclude large vector field
]

results = await collection.aggregate(pipeline).to_list(length=1000)
```

### Extracting Per-Pipeline Scores

```python
def extract_pipeline_score(score_details: dict, pipeline_name: str) -> float:
    """Extract individual pipeline contribution from scoreDetails."""
    if not score_details or "details" not in score_details:
        return 0.0

    for detail in score_details.get("details", []):
        if detail.get("inputPipelineName") == pipeline_name:
            return detail.get("value", 0.0)

    return 0.0

# Usage
for result in results:
    vector_score = extract_pipeline_score(result["score_details"], "vector")
    text_score = extract_pipeline_score(result["score_details"], "text")
    print(f"Total: {result['hybrid_score']:.4f} (vector: {vector_score:.4f}, text: {text_score:.4f})")
```

## Alternative: $scoreFusion

For weighted score combination instead of rank fusion:

```python
pipeline = [
    {
        "$scoreFusion": {
            "input": {
                "pipelines": {...},
                "normalization": "sigmoid"  # Required inside input
            },
            "combination": {
                "weights": {
                    "vector": 0.6,
                    "text": 0.4
                }
            },
            "scoreDetails": True
        }
    }
]
```

### When to Use Each

| Use Case | Algorithm | Reasoning |
|----------|-----------|-----------|
| General RAG | $rankFusion | Rank-based, position-aware |
| Similar score ranges | $scoreFusion | Direct score combination |

## Multi-Field Weighted Text Search

Search across multiple fields with different importance:

```python
config = MongoDBHybridSearchConfig(
    text_search_path_weights={
        "content": 10,      # Main content - highest weight
        "title": 5,         # Title - medium weight
        "metadata.tags": 3, # Tags - lower weight
    }
)

# Generates compound query with score boosting
{
    "compound": {
        "should": [
            {
                "text": {
                    "query": query_text,
                    "path": "content",
                    "fuzzy": {"maxEdits": 2, "prefixLength": 3},
                    "score": {"boost": {"value": 10}}
                }
            },
            {
                "text": {
                    "query": query_text,
                    "path": "title",
                    "fuzzy": {"maxEdits": 2, "prefixLength": 3},
                    "score": {"boost": {"value": 5}}
                }
            }
        ],
        "minimumShouldMatch": 1
    }
}
```

## Index Requirements

### Vector Search Index

```javascript
{
  "name": "vector_knn_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [{
      "type": "vector",
      "path": "vector",
      "numDimensions": 1024,
      "similarity": "cosine"
    }]
  }
}
```

### Text Search Index

```javascript
{
  "name": "text_search_index",
  "type": "search",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "content": {
          "type": "string",
          "analyzer": "lucene.standard"
        }
      }
    }
  }
}
```

## Best Practices

### 1. numCandidates Sizing

```python
# Always use 10-20x the limit for good recall
NUM_CANDIDATES_MULTIPLIER = 20

def calculate_num_candidates(top_k: int) -> int:
    return top_k * NUM_CANDIDATES_MULTIPLIER

# For top_k=10, use numCandidates=200
```

### 2. Weight Tuning

| Query Type | Vector Weight | Text Weight |
|------------|--------------|-------------|
| Natural language questions | 0.7 | 0.3 |
| Technical documentation | 0.5 | 0.5 |
| Code search | 0.3 | 0.7 |
| Proper nouns/names | 0.4 | 0.6 |

### 3. Error Handling

```python
from hybridrag.engine.exceptions import (
    RetrievalCapabilityError,
    RetrievalExecutionError,
)

try:
    results = await rag.query_data(
        query_text,
        mode="naive",
        fusion_strategy="score",
    )
except RetrievalCapabilityError:
    # The requested operation is unavailable. Change deployment capability or
    # make an explicit product decision; do not silently change algorithms.
    raise
except RetrievalExecutionError:
    # The selected retrieval path failed and returned no substituted evidence.
    raise
```

## Performance Considerations

| Factor | Recommendation |
|--------|---------------|
| numCandidates | 10-20x limit |
| Pipeline limit | 2x final limit (overfetch) |
| Index warming | Pre-query on startup |
| Result caching | Cache common queries |

## HybridRAG Integration

```python
from hybridrag import HybridRAG

rag = HybridRAG()
await rag.initialize()

# Uses hybrid search by default with mode="mix"
response = await rag.query(
    "What authentication methods are supported?",
    mode="mix"  # Combines vector + text + knowledge graph
)
```

## References

- [MongoDB $rankFusion Documentation](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/)
- [Hybrid Search Tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/hybrid-search/)
- [MongoDB Blog: Harness the Power of $rankFusion](https://www.mongodb.com/company/blog/technical/harness-power-atlas-search-vector-search-with-rankfusion)

---

**Next**: [Recipe 02: Lexical Prefilters for Vector Search](./02-lexical-prefilters.md)

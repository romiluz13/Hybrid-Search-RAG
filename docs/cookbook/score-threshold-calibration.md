# Score Threshold Calibration Guide

> **MongoDB Skill Rule:** `[Rule: query-get-scores]` - Document how to interpret and calibrate vector search scores.

## How Vector Search Scores Work

MongoDB Atlas Vector Search returns similarity scores via `$meta: "vectorSearchScore"`. The score semantics depend on your similarity function:

| Similarity | Score Range | Meaning |
|-----------|------------|---------|
| `cosine` | 0.0 - 1.0 | 1.0 = identical, 0.0 = orthogonal |
| `euclidean` | 0.0 - 1.0 | Normalized: 1.0 = identical |
| `dotProduct` | -inf to +inf | Higher = more similar |

HybridRAG uses **cosine** similarity by default (configured in `MongoVectorDBStorage`).

## Retrieving Scores

```python
# In aggregation pipeline, after $vectorSearch:
{"$addFields": {"score": {"$meta": "vectorSearchScore"}}}

# For $rankFusion results:
{"$addFields": {"score": {"$meta": "rankFusionScore"}}}

# For $scoreFusion results:
{"$addFields": {"score": {"$meta": "scoreFusionScore"}}}
```

## Recommended Threshold Ranges

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| High-precision retrieval | 0.85+ | Only very similar results |
| General Q&A / RAG | 0.70 - 0.85 | Good balance of recall and precision |
| Exploratory search | 0.50 - 0.70 | Cast a wider net |
| Deduplication | 0.95+ | Near-identical content |

HybridRAG's default threshold is configured via `cosine_better_than_threshold` in the global config.

## How to Calibrate

### Step 1: Collect Score Distributions

Run representative queries and collect the score distribution:

```python
from hybridrag import HybridRAG

rag = HybridRAG()
await rag.initialize()

# Query with only_context=True to see raw retrieval
context = await rag.query(
    "your representative query",
    only_context=True,
    top_k=50,  # Retrieve more to see the full distribution
)
```

### Step 2: Analyze the Distribution

Look at the score histogram across your query set:
- **Top results** should have scores > 0.80
- **Relevant but tangential** results: 0.60 - 0.80
- **Irrelevant results**: < 0.50

### Step 3: Set Your Threshold

```python
# In your HybridRAG configuration or global_config:
"vector_db_storage_cls_kwargs": {
    "cosine_better_than_threshold": 0.75  # Adjust based on your distribution
}
```

### Step 4: Monitor and Adjust

After deployment, monitor:
- **Recall**: Are relevant documents being missed? Lower the threshold.
- **Precision**: Are irrelevant documents included? Raise the threshold.
- **Score drift**: If your embedding model changes, recalibrate.

## Score-Based Filtering Example

```python
# Custom post-processing with score tiers
results = await vector_storage.query(query, top_k=20)

high_confidence = [r for r in results if r["score"] >= 0.85]
medium_confidence = [r for r in results if 0.70 <= r["score"] < 0.85]
low_confidence = [r for r in results if r["score"] < 0.70]
```

## Important Notes

- Scores are **not probabilities** - they are similarity measures
- Scores are **not comparable across different embedding models**
- After changing your embedding model, you must recalibrate thresholds
- Quantized indexes may have slightly different score distributions than full-precision

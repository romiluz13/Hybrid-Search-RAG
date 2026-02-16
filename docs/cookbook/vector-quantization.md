# Vector Quantization Guide

> **MongoDB Skill Rule:** `[Rule: index-quantization]` - Enable quantization at 100K+ vectors for RAM savings.

## When to Enable Quantization

| Dataset Size | Recommendation | RAM Reduction |
|-------------|----------------|---------------|
| < 100K vectors | No quantization needed | - |
| 100K - 1M vectors | Scalar quantization | ~3.75x |
| > 1M vectors | Binary quantization | ~24x |

## How It Works

Quantization reduces the memory footprint of vector indexes by compressing
float32 vectors into lower-precision representations:

- **Scalar (int8)**: Each float32 dimension becomes int8, reducing memory ~3.75x
- **Binary**: Each dimension becomes a single bit, reducing memory ~24x

The tradeoff is a small reduction in recall accuracy (typically < 2% for scalar).

## Scalar Quantization (Recommended for Most Cases)

Update your vector index definition to enable scalar quantization:

```javascript
db.collection.createSearchIndex(
  "vector_index",
  "vectorSearch",
  {
    fields: [
      {
        type: "vector",
        path: "vector",
        numDimensions: 1024,
        similarity: "cosine",
        quantization: "scalar"  // Add this
      },
      {
        type: "filter",
        path: "created_at"
      }
    ]
  }
)
```

## Binary Quantization (Maximum Compression)

For very large datasets (1M+ vectors) where RAM is the bottleneck:

```javascript
{
  type: "vector",
  path: "vector",
  numDimensions: 1024,
  similarity: "cosine",
  quantization: "binary"
}
```

## HybridRAG Integration

HybridRAG currently uses full-precision vectors. To enable quantization:

1. **Update the vector index** (not the application code) - quantization is an
   index-level setting that does not require code changes.

2. **Recreate the index** with the quantization parameter. Atlas will rebuild
   the index in the background.

3. **Monitor recall** - Run your benchmark queries before and after to verify
   acceptable accuracy:

```python
# Run benchmark before and after quantization
hybridrag benchmark --mode mix --top-k 10
```

## When NOT to Quantize

- Small datasets (< 100K vectors): No meaningful RAM savings
- High-precision requirements (deduplication, near-exact matching)
- Binary quantization with low-dimensional embeddings (< 256 dims)

## RAM Estimation

| Vectors | Dimensions | Full Precision | Scalar | Binary |
|---------|-----------|---------------|--------|--------|
| 100K | 1024 | ~400 MB | ~107 MB | ~17 MB |
| 1M | 1024 | ~4 GB | ~1.1 GB | ~170 MB |
| 10M | 1024 | ~40 GB | ~11 GB | ~1.7 GB |

## References

- [MongoDB Atlas Vector Search Quantization](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/)
- MongoDB Skill Rule: `perf-quantization-scale`
- MongoDB Skill Rule: `index-quantization`

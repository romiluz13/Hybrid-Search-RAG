# Enhanced Search Features

HybridRAG provides advanced search capabilities that go beyond simple vector similarity.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   SEARCH HIERARCHY                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Level 1: Vector Only (naive mode)                          │
│    └─ Semantic similarity via embeddings                    │
│                                                              │
│  Level 2: Hybrid (vector + keyword)                         │
│    └─ RRF fusion of vector + text search                    │
│                                                              │
│  Level 3: Mix Mode (vector + keyword + graph)               │
│    └─ Full KG traversal + entity boosting                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Graph Search

Graph search uses MongoDB's `$graphLookup` aggregation stage to traverse the knowledge graph.

### How It Works

1. **Entity Extraction**: Entities are extracted from the query
2. **Graph Traversal**: `$graphLookup` traverses entity relationships
3. **Chunk Retrieval**: Chunks mentioning expanded entities are fetched
4. **Score Boosting**: Entity overlap boosts result scores

### Configuration

```python
from hybridrag.enhancements import GraphTraversalConfig, graph_traversal

config = GraphTraversalConfig(
    edges_collection="kg_edges",      # Collection storing edges
    chunks_collection="text_chunks",   # Collection storing chunks
    entities_collection="kg_entities", # Collection storing entities
    source_field="source_node_id",     # Field for source entity
    target_field="target_node_id",     # Field for target entity
    relation_field="relation_type",    # Field for relationship type
    weight_field="weight",             # Field for edge weight
    max_depth=2,                       # Max traversal depth
    max_nodes=50,                      # Max nodes to visit
)

# Execute graph traversal
result = await graph_traversal(db, "MongoDB", config=config)
print(f"Found {len(result.entities)} entities")
print(f"Traversed {len(result.edges)} edges")
```

### Graph Traversal Result

```python
@dataclass
class GraphTraversalResult:
    """Result of graph traversal."""

    entities: list[str]           # Discovered entity names
    edges: list[GraphEdge]        # Traversed edges
    entity_scores: dict[str, float]  # Entity relevance scores
```

### Entity Expansion

Expand entities via graph relationships:

```python
from hybridrag.enhancements import expand_entities_via_graph

# Start with query entities
query_entities = ["MongoDB", "vector search"]

# Expand via graph
expanded_entities, edges = await expand_entities_via_graph(
    db=db,
    query_entities=query_entities,
    config=config,
)

print(f"Expanded from {len(query_entities)} to {len(expanded_entities)} entities")
```

### Get Chunks for Entities

Retrieve chunks mentioning specific entities:

```python
from hybridrag.enhancements import get_chunks_for_entities

chunks = await get_chunks_for_entities(
    db=db,
    entity_names=["MongoDB", "Atlas", "vector search"],
    limit=20,
    config=config,
)

for chunk in chunks:
    print(f"Chunk: {chunk['content'][:100]}...")
```

## Mix Mode Search

Mix mode combines all search modalities for comprehensive retrieval.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MIX MODE SEARCH                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │    HYBRID     │  │    GRAPH      │  │    ENTITY     │    │
│  │ (native fusion)│  │  ($graphLookup)│  │   BOOSTING   │    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘    │
│          │                  │                  │             │
│          └──────────────────┼──────────────────┘             │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │ MERGE + DEDUP   │                       │
│                    │   + Dedup       │                       │
│                    └────────┬────────┘                       │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │ ENTITY BOOSTING │                       │
│                    └────────┬────────┘                       │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │   TOP-K RESULTS │                       │
│                    └─────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Basic Usage

```python
results = await rag.query_data(
    "How does MongoDB handle vector search?",
    mode="mix",
    fusion_strategy="score",
)
```

The public path fails explicitly if any requested native or graph branch fails. Low-level helpers do not provide a semantic fallback contract.

### Configuration

```python
from hybridrag.enhancements import (
    MixModeConfig,
    MongoDBHybridSearchConfig,
    GraphTraversalConfig,
)

# Configure hybrid search
hybrid_config = MongoDBHybridSearchConfig(
    vector_weight=0.6,       # Weight for vector search
    text_weight=0.4,         # Weight for keyword search
    fuzzy_max_edits=2,       # Fuzzy matching edits
    cosine_threshold=0.3,    # Minimum cosine similarity
)

# Configure graph traversal
graph_config = GraphTraversalConfig(
    max_depth=2,
    max_nodes=50,
)

# Configure mix mode
config = MixModeConfig(
    hybrid_config=hybrid_config,
    graph_config=graph_config,
    enable_graph_traversal=True,   # Enable/disable graph
    enable_entity_boosting=True,   # Enable/disable entity boost
    enable_reranking=True,         # Enable/disable reranking
    entity_boost_weight=0.2,       # How much entity overlap affects score
    entity_only_weight=0.5,        # Score for entity-only results
)

results = await mix_mode_search(
    db=db,
    query=query,
    query_vector=embedding,
    config=config,
)
```

### MixModeSearcher Class

For repeated searches with the same configuration:

```python
from hybridrag.enhancements import create_mix_mode_searcher

# Create a configured searcher
searcher = await create_mix_mode_searcher(
    db=db,
    workspace="",
    vector_weight=0.6,
    text_weight=0.4,
    entity_boost_weight=0.2,
    enable_graph=True,
)

# Execute searches
results = await searcher.search(
    query="...",
    query_vector=embedding,
    top_k=10,
    query_entities=["entity1", "entity2"],
)

# Graph-only search (no hybrid)
graph_results = await searcher.search_with_graph_only(
    query_entities=["entity1", "entity2"],
    top_k=10,
)
```

### Result Structure

```python
class MixModeSearchResult(BaseModel):
    """Extended search result with mix mode metadata."""

    chunk_id: str                    # MongoDB ObjectId as string
    document_id: str                 # Parent document ObjectId
    content: str                     # Chunk text content
    score: float                     # Combined relevance score
    metadata: dict[str, Any]         # Chunk metadata

    # Search source breakdown
    search_type: str                 # "mix_mode", "entity_only", etc.
    source_scores: dict[str, float]  # Per-source scores: {vector, text, entity}

    # Graph metadata
    graph_entities: list[str]        # Related entities from graph
    entity_boost: float              # Boost applied from entity overlap

    # Document metadata
    document_title: str              # Title from document lookup
    document_source: str             # Source path from document
```

## Entity Boosting

Entity boosting improves relevance by considering knowledge graph entities.

### How It Works

1. **Entity Extraction**: Extract entities from query and chunks
2. **Overlap Calculation**: Compute entity overlap between query and chunk
3. **Score Boost**: Add boost proportional to overlap

### Configuration

```python
from hybridrag.enhancements import EntityBoostingReranker

reranker = EntityBoostingReranker(
    entity_weight=0.3,        # Weight for entity overlap
    semantic_weight=0.7,      # Weight for semantic similarity
    min_entity_overlap=0.1,   # Minimum overlap to apply boost
)

boosted_results = await reranker.rerank(
    query="...",
    results=initial_results,
    query_entities=["entity1", "entity2"],
)
```

## Filter Builders

Pre-filter results before vector search for better performance.

### Vector Search Filters

```python
from hybridrag.enhancements import (
    VectorSearchFilterConfig,
    build_vector_search_filters,
)
from datetime import datetime, timedelta

config = VectorSearchFilterConfig(
    timestamp_field="timestamp",
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    equality_filters={"category": "tech"},
    in_filters={"tags": ["mongodb", "rag"]},
)

filters = build_vector_search_filters(config)
# Returns: {"$and": [...]}
```

### Atlas Search Filters

```python
from hybridrag.enhancements import (
    AtlasSearchFilterConfig,
    build_atlas_search_filters,
)

config = AtlasSearchFilterConfig(
    timestamp_field="timestamp",
    start_date=datetime.now() - timedelta(days=7),
    equality_filters={"author": "John"},
)

clauses = build_atlas_search_filters(config)
# Returns list of Atlas Search filter clauses
```

## Performance Considerations

### When to Use Each Mode

| Scenario | Recommended Mode |
|----------|------------------|
| General queries | `mix` |
| Entity-specific | `local` + graph |
| High-level overview | `global` |
| Speed critical | `naive` |
| Maximum quality | `mix` + reranking |

### Optimization Tips

1. **Index your entities**: Ensure indexes on entity name fields
2. **Limit graph depth**: Start with `max_depth=2`, increase if needed
3. **Pre-extract entities**: Cache entity extraction for repeated queries
4. **Use workspace isolation**: Prefix collections for multi-tenant

### Example Indexes

```javascript
// Entity collection index
db.kg_entities.createIndex({ name: 1 })
db.kg_entities.createIndex({ workspace: 1, name: 1 })

// Edge collection indexes
db.kg_edges.createIndex({ source_node_id: 1 })
db.kg_edges.createIndex({ target_node_id: 1 })
db.kg_edges.createIndex({ source_node_id: 1, target_node_id: 1 })

// Chunk collection index for entity lookup
db.text_chunks.createIndex({ "metadata.entities": 1 })
```

## Next Steps

- [Query Modes Explained](query-modes.md) - Understanding all query modes
- [Configuration Guide](configuration.md) - All configuration options
- [API Reference](api.md) - Complete API documentation

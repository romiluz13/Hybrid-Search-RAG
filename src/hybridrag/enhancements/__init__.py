"""
HybridRAG Enhancements.

Advanced RAG capabilities:
- Entity Boosting: Structural relevance signal from KG entities
- Implicit Expansion: Find semantically related entities without explicit graph edges
- MongoDB Hybrid Search: Native score/rank fusion with fail-closed execution
- Graph Search: Knowledge graph traversal via $graphLookup
- Mix Mode Search: Combines hybrid search + graph traversal + entity boosting
- Filter Builders: Vector Search and Atlas Search prefiltering
- Query Optimizer: Automatic parameter optimization based on query analysis
"""

from .entity_boosting import EntityBoostingReranker, create_boosted_rerank_func
from .filters import (
    AtlasSearchFilterConfig,
    FuzzyFilter,
    GeoFilter,
    LexicalPrefilterConfig,
    PhraseFilter,
    QueryStringFilter,
    TextFilter,
    VectorSearchFilterConfig,
    WildcardFilter,
    build_atlas_search_filters,
    build_lexical_prefilters,
    build_search_vector_search_stage,
    build_vector_search_filters,
)
from .graph_search import (
    GraphEdge,
    GraphTraversalConfig,
    GraphTraversalResult,
    build_graph_lookup_pipeline,
    expand_entities_via_graph,
    get_chunks_for_entities,
    graph_traversal,
    normalize_entity_name,
)
from .implicit_expansion import ImplicitExpander
from .mix_mode_search import (
    MixModeConfig,
    MixModeSearcher,
    MixModeSearchResult,
    create_mix_mode_searcher,
    mix_mode_search,
)
from .mongodb_hybrid_search import (
    DEFAULT_RRF_CONSTANT,
    NUM_CANDIDATES_MULTIPLIER,
    MongoDBHybridSearchConfig,
    MongoDBHybridSearcher,
    SearchResult,
    calculate_num_candidates,
    create_hybrid_searcher,
    extract_pipeline_score,
    hybrid_search_with_rank_fusion,
    hybrid_search_with_score_fusion,
    manual_hybrid_search_with_rrf,
    multi_field_text_search,
    reciprocal_rank_fusion,
    text_only_search,
    vector_only_search,
    vector_search_with_lexical_prefilters,
)
from .query_optimizer import OptimizedQueryParams, QueryOptimizer

__all__ = [
    # Entity Boosting
    "EntityBoostingReranker",
    "create_boosted_rerank_func",
    # Implicit Expansion
    "ImplicitExpander",
    # Query Optimizer
    "QueryOptimizer",
    "OptimizedQueryParams",
    # Graph Search
    "GraphEdge",
    "GraphTraversalConfig",
    "GraphTraversalResult",
    "build_graph_lookup_pipeline",
    "graph_traversal",
    "expand_entities_via_graph",
    "get_chunks_for_entities",
    "normalize_entity_name",
    # Mix Mode Search
    "MixModeConfig",
    "MixModeSearcher",
    "MixModeSearchResult",
    "mix_mode_search",
    "create_mix_mode_searcher",
    # Config
    "MongoDBHybridSearchConfig",
    "VectorSearchFilterConfig",
    "AtlasSearchFilterConfig",
    "LexicalPrefilterConfig",
    # Searcher
    "MongoDBHybridSearcher",
    "create_hybrid_searcher",
    # Search functions
    "hybrid_search_with_rank_fusion",
    "hybrid_search_with_score_fusion",
    "manual_hybrid_search_with_rrf",
    "multi_field_text_search",
    "text_only_search",
    "vector_only_search",
    "vector_search_with_lexical_prefilters",
    # Utilities
    "reciprocal_rank_fusion",
    "SearchResult",
    "calculate_num_candidates",
    "extract_pipeline_score",
    # Constants
    "DEFAULT_RRF_CONSTANT",
    "NUM_CANDIDATES_MULTIPLIER",
    # Filter builders
    "build_vector_search_filters",
    "build_atlas_search_filters",
    "build_lexical_prefilters",
    "build_search_vector_search_stage",
    # Filter types
    "TextFilter",
    "FuzzyFilter",
    "PhraseFilter",
    "WildcardFilter",
    "GeoFilter",
    "QueryStringFilter",
]

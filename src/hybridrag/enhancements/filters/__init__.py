"""
MongoDB Filter Builders for HybridRAG.

This module provides three distinct filter builder systems:
1. Vector Search Filters - Use standard MongoDB operators ($gte, $lte, $eq)
2. Atlas Search Filters - Use Atlas Search operators (range, equals)
3. Lexical Prefilters - Use Atlas Search operators in $search.vectorSearch (MongoDB 8.2+)

CRITICAL: These use DIFFERENT syntaxes for the same logical filters!
"""

from .atlas_search_filters import (
    AtlasSearchFilterConfig,
    build_atlas_search_filters,
    build_compound_search_stage,
)
from .lexical_prefilters import (
    FuzzyFilter,
    GeoFilter,
    LexicalPrefilterConfig,
    PhraseFilter,
    QueryStringFilter,
    TextFilter,
    WildcardFilter,
    build_lexical_prefilters,
    build_search_vector_search_stage,
)
from .vector_search_filters import (
    FilterConfig,
    FilterPredicate,
    RetrievalSecurityContext,
    VectorSearchFilterConfig,
    build_vector_search_filters,
    build_vector_search_stage,
    compile_filter_to_atlas,
    compile_filter_to_mql,
    compile_retrieval_filter_to_atlas,
    compile_retrieval_filter_to_mql,
    validate_filter_config_for_mappings,
)

__all__ = [
    # Backend-neutral public filters
    "FilterConfig",
    "FilterPredicate",
    "RetrievalSecurityContext",
    "compile_filter_to_atlas",
    "compile_filter_to_mql",
    "compile_retrieval_filter_to_atlas",
    "compile_retrieval_filter_to_mql",
    "validate_filter_config_for_mappings",
    # Vector Search Filters (MQL operators)
    "build_vector_search_filters",
    "build_vector_search_stage",
    "VectorSearchFilterConfig",
    # Atlas Search Filters (Atlas operators)
    "build_atlas_search_filters",
    "build_compound_search_stage",
    "AtlasSearchFilterConfig",
    # Lexical Prefilters (MongoDB 8.2+ $search.vectorSearch)
    "build_lexical_prefilters",
    "build_search_vector_search_stage",
    "LexicalPrefilterConfig",
    "TextFilter",
    "FuzzyFilter",
    "PhraseFilter",
    "WildcardFilter",
    "GeoFilter",
    "QueryStringFilter",
]

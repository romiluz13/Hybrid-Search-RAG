"""
Search presets for common use cases.

Provides predefined search configurations to optimize for:
- Balanced search (default)
- Semantic-focused search
- Keyword-focused search
- Comprehensive search (with graph)
"""

from dataclasses import dataclass


@dataclass
class SearchPreset:
    """
    Search configuration preset.

    Attributes:
        name: Preset name
        vector_weight: Weight for vector search (0.0-1.0)
        text_weight: Weight for text search (0.0-1.0)
        entity_weight: Optional weight for entity/graph search
        top_k: Number of results to return
        graph_traversal: Enable graph traversal
        description: Human-readable description
        rank_fusion_weights: Weights dict for $rankFusion combination (auto-generated)
    """

    name: str
    vector_weight: float
    text_weight: float
    entity_weight: float | None = None
    top_k: int = 10
    graph_traversal: bool = False
    description: str = ""

    def __post_init__(self):
        """Validate weights and generate rank_fusion_weights."""
        total = self.vector_weight + self.text_weight
        # [L17] Use 'is not None' instead of truthiness to handle entity_weight=0.0
        if self.entity_weight is not None:
            total += self.entity_weight

        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Weights must sum to 1.0, got {total} "
                f"(vector={self.vector_weight}, text={self.text_weight}, "
                f"entity={self.entity_weight})"
            )

        # Generate $rankFusion combination weights from preset
        self.rank_fusion_weights: dict[str, float] = {
            "vector": self.vector_weight,
            "text": self.text_weight,
        }


# Predefined presets
PRESETS = {
    "balanced": SearchPreset(
        name="balanced",
        vector_weight=0.5,
        text_weight=0.5,
        top_k=10,
        description="Balanced vector + keyword search. Good default for most use cases.",
    ),
    "semantic": SearchPreset(
        name="semantic",
        vector_weight=0.8,
        text_weight=0.2,
        top_k=15,
        description="Semantic-focused search. Best for conceptual queries and paraphrasing.",
    ),
    "keyword": SearchPreset(
        name="keyword",
        vector_weight=0.2,
        text_weight=0.8,
        top_k=10,
        description="Keyword-focused search. Best for exact term matching and technical terms.",
    ),
    "comprehensive": SearchPreset(
        name="comprehensive",
        vector_weight=0.5,
        text_weight=0.3,
        entity_weight=0.2,
        graph_traversal=True,
        top_k=20,
        description="Full hybrid + graph search. Best for complex queries requiring entity relationships.",
    ),
    "fast": SearchPreset(
        name="fast",
        vector_weight=0.6,
        text_weight=0.4,
        top_k=5,
        description="Optimized for speed. Fewer results, good recall.",
    ),
    "precise": SearchPreset(
        name="precise",
        vector_weight=0.4,
        text_weight=0.6,
        top_k=5,
        description="Precision-focused. Best for specific, well-defined queries.",
    ),
    "exploratory": SearchPreset(
        name="exploratory",
        vector_weight=0.7,
        text_weight=0.3,
        top_k=20,
        description="Broad search for exploration. More results, diverse perspectives.",
    ),
}


def get_preset(name: str) -> SearchPreset:
    """
    Get a preset by name.

    Args:
        name: Preset name (e.g., "balanced", "semantic", "keyword")

    Returns:
        SearchPreset configuration

    Raises:
        KeyError: If preset not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Preset '{name}' not found. Available: {available}")

    return PRESETS[name]


def list_presets() -> dict[str, str]:
    """
    List all available presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {name: preset.description for name, preset in PRESETS.items()}


# [L18] Set of built-in preset names that cannot be overwritten
_BUILTIN_PRESETS: frozenset[str] = frozenset(PRESETS.keys())


def register_preset(preset: SearchPreset) -> None:
    """
    Register a custom preset.

    Built-in presets (balanced, semantic, keyword, comprehensive, fast,
    precise, exploratory) cannot be overwritten.

    Args:
        preset: SearchPreset instance to register

    Raises:
        ValueError: If attempting to overwrite a built-in preset

    Example:
        >>> custom = SearchPreset(
        ...     name="custom",
        ...     vector_weight=0.7,
        ...     text_weight=0.3,
        ...     top_k=15,
        ...     description="My custom preset"
        ... )
        >>> register_preset(custom)
    """
    if preset.name in _BUILTIN_PRESETS:
        raise ValueError(
            f"Cannot overwrite built-in preset '{preset.name}'. "
            f"Built-in presets: {', '.join(sorted(_BUILTIN_PRESETS))}"
        )
    PRESETS[preset.name] = preset


# Usage examples in docstrings
"""
Usage Examples:

1. Basic usage with preset name:
    >>> from hybridrag import create_hybridrag
    >>> from hybridrag.presets import get_preset
    >>>
    >>> rag = await create_hybridrag()
    >>> preset = get_preset("semantic")
    >>> results = await rag.query(
    ...     query="explain vector embeddings",
    ...     mode="hybrid",
    ...     vector_weight=preset.vector_weight,
    ...     text_weight=preset.text_weight,
    ...     top_k=preset.top_k,
    ... )

2. List available presets:
    >>> from hybridrag.presets import list_presets
    >>> presets = list_presets()
    >>> for name, desc in presets.items():
    ...     print(f"{name}: {desc}")

3. Create custom preset:
    >>> from hybridrag.presets import SearchPreset, register_preset
    >>>
    >>> custom = SearchPreset(
    ...     name="technical_docs",
    ...     vector_weight=0.3,
    ...     text_weight=0.7,
    ...     top_k=10,
    ...     description="Optimized for technical documentation with exact terminology"
    ... )
    >>> register_preset(custom)

4. Access preset properties:
    >>> from hybridrag.presets import get_preset
    >>> preset = get_preset("balanced")
    >>> print(f"Vector weight: {preset.vector_weight}")
    >>> print(f"Text weight: {preset.text_weight}")
    >>> print(f"Top K: {preset.top_k}")
"""

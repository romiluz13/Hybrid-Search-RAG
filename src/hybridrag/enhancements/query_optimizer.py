"""
Query Optimizer

Automatically optimizes query parameters based on query analysis.

Features:
- Query length analysis
- Entity density detection
- Query type classification
- Automatic parameter tuning
"""

import re
from dataclasses import dataclass

from ..prompts import QueryType, detect_query_type


@dataclass
class OptimizedQueryParams:
    """
    Optimized query parameters.

    Attributes:
        vector_weight: Optimized vector search weight
        text_weight: Optimized text search weight
        top_k: Optimized number of results
        mode: Recommended search mode
        enable_reranking: Whether to enable reranking
        enable_graph: Whether to enable graph traversal
        reasoning: Explanation of optimization decisions
    """

    vector_weight: float
    text_weight: float
    top_k: int
    mode: str = "hybrid"
    enable_reranking: bool = False
    enable_graph: bool = False
    reasoning: str = ""


class QueryOptimizer:
    """
    Automatically optimize query parameters based on query analysis.

    The optimizer analyzes:
    - Query length (short vs long)
    - Query complexity (simple vs complex)
    - Entity density (number of proper nouns, technical terms)
    - Query type (general, summary, tools, troubleshooting)

    And recommends:
    - Optimal vector/text weights
    - Appropriate top_k
    - Whether to use reranking
    - Whether to enable graph traversal
    """

    def __init__(
        self,
        default_vector_weight: float = 0.6,
        default_text_weight: float = 0.4,
        default_top_k: int = 10,
    ):
        """
        Initialize optimizer with defaults.

        Args:
            default_vector_weight: Default vector search weight
            default_text_weight: Default text search weight
            default_top_k: Default number of results
        """
        self.default_vector_weight = default_vector_weight
        self.default_text_weight = default_text_weight
        self.default_top_k = default_top_k

    def optimize(self, query: str) -> OptimizedQueryParams:
        """
        Optimize query parameters based on query analysis.

        Args:
            query: User query text

        Returns:
            OptimizedQueryParams with recommended configuration
        """
        reasoning_parts = []

        # Analyze query
        query_length = len(query.split())
        query_type = detect_query_type(query)
        entity_density = self._calculate_entity_density(query)
        has_technical_terms = self._has_technical_terms(query)

        # Initialize with defaults
        vector_weight = self.default_vector_weight
        text_weight = self.default_text_weight
        top_k = self.default_top_k
        enable_reranking = False
        enable_graph = False

        # Optimization rules

        # Rule 1: Short queries → Increase vector weight (semantic matching)
        if query_length <= 3:
            vector_weight = 0.7
            text_weight = 0.3
            reasoning_parts.append(
                f"Short query ({query_length} words) → increased vector weight for semantic matching"
            )

        # Rule 2: Long queries → Increase text weight (keyword matching)
        elif query_length >= 15:
            vector_weight = 0.4
            text_weight = 0.6
            reasoning_parts.append(
                f"Long query ({query_length} words) → increased text weight for keyword matching"
            )

        # Rule 3: High entity density → Enable graph + increase text weight
        if entity_density > 0.3:
            enable_graph = True
            text_weight = min(text_weight + 0.1, 0.7)
            vector_weight = 1.0 - text_weight
            reasoning_parts.append(
                f"High entity density ({entity_density:.2f}) → enabled graph traversal"
            )

        # Rule 4: Technical terms → Favor keyword matching
        if has_technical_terms:
            text_weight = min(text_weight + 0.1, 0.7)
            vector_weight = 1.0 - text_weight
            reasoning_parts.append(
                "Technical terms detected → increased text weight for exact matching"
            )

        # Rule 5: Query type optimization
        if query_type == QueryType.SUMMARY:
            # Summaries benefit from semantic search
            vector_weight = 0.7
            text_weight = 0.3
            top_k = 15  # More context for summarization
            reasoning_parts.append(
                "Summary query → semantic focus + increased top_k for context"
            )

        elif query_type == QueryType.TOOLS:
            # How-to queries benefit from keyword matching
            text_weight = 0.7
            vector_weight = 0.3
            reasoning_parts.append(
                "How-to query → keyword focus for step-by-step content"
            )

        elif query_type == QueryType.TROUBLESHOOTING:
            # Troubleshooting benefits from exact error matching
            text_weight = 0.6
            vector_weight = 0.4
            top_k = 15  # More results for debugging
            reasoning_parts.append(
                "Troubleshooting query → keyword focus + increased top_k"
            )

        # Rule 6: Enable reranking for complex queries
        if query_length > 10 or entity_density > 0.2:
            enable_reranking = True
            reasoning_parts.append("Complex query → enabled reranking for quality")

        # Ensure weights sum to 1.0
        total = vector_weight + text_weight
        vector_weight = vector_weight / total
        text_weight = text_weight / total

        # [L9] Validate weight normalization invariant
        assert abs(vector_weight + text_weight - 1.0) < 1e-6, (
            f"Weights must sum to 1.0 after normalization, got {vector_weight + text_weight}"
        )

        reasoning = (
            " | ".join(reasoning_parts) if reasoning_parts else "Default configuration"
        )

        return OptimizedQueryParams(
            vector_weight=vector_weight,
            text_weight=text_weight,
            top_k=top_k,
            mode="hybrid",
            enable_reranking=enable_reranking,
            enable_graph=enable_graph,
            reasoning=reasoning,
        )

    def _calculate_entity_density(self, query: str) -> float:
        """
        Calculate entity density (ratio of capitalized words).

        Args:
            query: Query text

        Returns:
            Entity density (0.0-1.0)
        """
        words = query.split()
        if not words:
            return 0.0

        # Count capitalized words (excluding first word)
        capitalized = sum(1 for word in words[1:] if word and word[0].isupper())

        return capitalized / len(words) if len(words) > 1 else 0.0

    def _has_technical_terms(self, query: str) -> bool:
        """
        Check if query contains technical terms.

        Args:
            query: Query text

        Returns:
            True if technical terms detected
        """
        # Common technical indicators
        technical_patterns = [
            r"\bAPI\b",
            r"\bSDK\b",
            r"\bHTTP\b",
            r"\bJSON\b",
            r"\bXML\b",
            r"\bSQL\b",
            r"\$\w+",  # MongoDB operators like $match, $lookup
            r"\b\w+\(\)",  # Function syntax
            r"\b\w+\.\w+",  # Dotted notation
            r"\bv?\d+\.\d+",  # Version numbers
        ]

        query_upper = query.upper()

        # Check for technical patterns
        for pattern in technical_patterns:
            if re.search(pattern, query):
                return True

        # Check for common technical keywords
        technical_keywords = [
            "INDEX",
            "QUERY",
            "DATABASE",
            "COLLECTION",
            "DOCUMENT",
            "PIPELINE",
            "AGGREGATION",
            "SCHEMA",
            "VECTOR",
            "EMBEDDING",
            "CLUSTER",
        ]

        return any(keyword in query_upper for keyword in technical_keywords)


# Usage example
"""
Usage Example:

    from hybridrag.enhancements import QueryOptimizer

    optimizer = QueryOptimizer()

    # Optimize query
    params = optimizer.optimize("What is MongoDB Atlas?")

    print(f"Vector weight: {params.vector_weight}")
    print(f"Text weight: {params.text_weight}")
    print(f"Top K: {params.top_k}")
    print(f"Enable reranking: {params.enable_reranking}")
    print(f"Enable graph: {params.enable_graph}")
    print(f"Reasoning: {params.reasoning}")

    # Use optimized parameters
    results = await rag.query(
        query="What is MongoDB Atlas?",
        mode=params.mode,
        vector_weight=params.vector_weight,
        text_weight=params.text_weight,
        top_k=params.top_k,
    )
"""

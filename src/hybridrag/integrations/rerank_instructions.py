"""
Intelligent instruction generation for instruction-following reranking.

Follows Voyage AI's official patterns from MongoDB blog post:
https://www.mongodb.com/developer/products/atlas/instruction-following-reranking/

Instructions use hierarchical prioritization patterns:
- "Prioritize X, followed by Y, then Z, and finally W"
- "Prioritize X and Y" (for multiple criteria)
- Keep instructions concise and action-oriented
"""

from typing import Literal


def get_rerank_instructions(
    query_mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"],
    custom_instructions: str | None = None,
    enable_smart_defaults: bool = True,
) -> str | None:
    """
    Generate intelligent reranking instructions based on query mode.

    Instructions follow Voyage AI's official patterns:
    - Short and action-oriented
    - Use "Prioritize X, then Y" hierarchical patterns
    - Keep under 150 characters for best results

    Args:
        query_mode: The query mode being used
        custom_instructions: Custom instructions to override defaults
        enable_smart_defaults: Whether to use intelligent defaults (default: True)

    Returns:
        Instructions string or None if disabled
    """
    # Custom instructions always take precedence
    if custom_instructions:
        return custom_instructions.strip() if custom_instructions.strip() else None

    # If smart defaults disabled, return None
    if not enable_smart_defaults:
        return None

    # Mode-specific instructions following Voyage AI patterns
    # Based on MongoDB blog scenarios: source prioritization, query type, temporal/importance
    instructions_map = {
        "mix": (
            "Prioritize documents with specific entities and facts, followed by "
            "comprehensive content, then general explanations."
        ),
        "local": (
            "Prioritize documents containing specific entities and facts mentioned "
            "in the query, followed by documents with direct relationships."
        ),
        "global": (
            "Prioritize high-level summaries and overviews, followed by comprehensive "
            "explanations and authoritative sources."
        ),
        "hybrid": (
            "Prioritize documents balancing specific details with broader context, "
            "followed by content connecting entities to themes."
        ),
        "naive": (
            "Prioritize documents with highest semantic similarity to the query."
        ),
        "bypass": None,  # No reranking for bypass mode
    }

    return instructions_map.get(query_mode)

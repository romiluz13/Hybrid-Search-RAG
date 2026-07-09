"""
Reranking Prompt Templates.

Query-type specific reranking instructions for cross-encoder rerankers
(e.g., Voyage Rerank, Cohere Rerank, BGE Reranker).

The instructions guide the reranker to prioritize different aspects based on query intent:
- General: Balanced relevance across topics
- Summary: Comprehensive overview and context
- Tools: Practical how-to and implementation details
- Troubleshooting: Problem diagnosis and solutions

Usage:
    from hybridrag.prompts import detect_query_type, select_rerank_instruction

    query = "How do I fix the authentication error?"
    query_type = detect_query_type(query)
    instruction = select_rerank_instruction(query_type)

    # Use with reranker
    reranker.rerank(query, documents, instruction=instruction)
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final


class QueryType(StrEnum):
    """Query type classification for reranking instruction selection."""

    GENERAL = "general"
    SUMMARY = "summary"
    TOOLS = "tools"
    TROUBLESHOOTING = "troubleshooting"


# Query type detection patterns
QUERY_TYPE_PATTERNS: Final[dict[QueryType, list[str]]] = {
    QueryType.SUMMARY: [
        r"(?:what|tell me about|explain|describe|overview|summary|summarize)",
        r"(?:who is|what is|where is|when did)",
        r"(?:background|history|context)",
    ],
    QueryType.TOOLS: [
        r"(?:how (?:do|can|to|should)|steps to|guide|tutorial|example)",
        r"(?:implement|configure|setup|install|deploy|integrate)",
        r"(?:code|script|command|api|sdk|library)",
        r"(?:best practice|pattern|approach|method)",
    ],
    QueryType.TROUBLESHOOTING: [
        r"(?:error|issue|problem|bug|fix|debug|troubleshoot)",
        r"(?:not working|doesn't work|failed|failing|broken)",
        r"(?:why (?:is|does|did|won't|can't))",
        r"(?:help|stuck|can't figure out)",
    ],
}


RERANK_INSTRUCTION_GENERAL: Final[str] = """Prioritize documents that:
1. Directly answer the user's question with specific, factual information
2. Contain the key concepts and entities mentioned in the query
3. Provide authoritative and comprehensive coverage of the topic
4. Are from recent or canonical sources when relevance is equal
"""

RERANK_INSTRUCTION_SUMMARY: Final[str] = """Prioritize documents that:
1. Provide comprehensive overview and background information
2. Define key concepts and explain their significance
3. Cover the historical context or evolution of the topic
4. Offer multiple perspectives or viewpoints when available
5. Include foundational knowledge that helps understand the topic
"""

RERANK_INSTRUCTION_TOOLS: Final[str] = """Prioritize documents that:
1. Contain step-by-step instructions, code examples, or implementation guides
2. Show practical usage patterns, configurations, or API calls
3. Include working code snippets or command-line examples
4. Describe prerequisites, setup requirements, and dependencies
5. Address common implementation patterns and best practices
"""

RERANK_INSTRUCTION_TROUBLESHOOTING: Final[str] = """Prioritize documents that:
1. Address the specific error, issue, or symptom mentioned
2. Provide diagnostic steps or debugging approaches
3. Offer solutions with clear fix instructions
4. Explain root causes and prevention strategies
5. Include workarounds when direct solutions aren't available
"""

# Map query types to instructions
_INSTRUCTION_MAP: Final[dict[QueryType, str]] = {
    QueryType.GENERAL: RERANK_INSTRUCTION_GENERAL,
    QueryType.SUMMARY: RERANK_INSTRUCTION_SUMMARY,
    QueryType.TOOLS: RERANK_INSTRUCTION_TOOLS,
    QueryType.TROUBLESHOOTING: RERANK_INSTRUCTION_TROUBLESHOOTING,
}


def detect_query_type(query: str) -> QueryType:
    """
    Detect the query type based on pattern matching.

    Uses regex patterns to classify queries into one of four types:
    - SUMMARY: Questions about what, who, overview, background
    - TOOLS: How-to questions, implementation, code examples
    - TROUBLESHOOTING: Error fixing, debugging, problem solving
    - GENERAL: Default fallback for other queries

    Args:
        query: The user's search query

    Returns:
        QueryType enum value

    Examples:
        >>> detect_query_type("What is machine learning?")
        <QueryType.SUMMARY: 'summary'>

        >>> detect_query_type("How do I implement authentication?")
        <QueryType.TOOLS: 'tools'>

        >>> detect_query_type("Error: connection refused")
        <QueryType.TROUBLESHOOTING: 'troubleshooting'>
    """
    query_lower = query.lower()

    # Check troubleshooting first (highest priority for error-related queries)
    for pattern in QUERY_TYPE_PATTERNS[QueryType.TROUBLESHOOTING]:
        if re.search(pattern, query_lower):
            return QueryType.TROUBLESHOOTING

    # Check tools/how-to patterns
    for pattern in QUERY_TYPE_PATTERNS[QueryType.TOOLS]:
        if re.search(pattern, query_lower):
            return QueryType.TOOLS

    # Check summary/informational patterns
    for pattern in QUERY_TYPE_PATTERNS[QueryType.SUMMARY]:
        if re.search(pattern, query_lower):
            return QueryType.SUMMARY

    # Default to general
    return QueryType.GENERAL


def select_rerank_instruction(query_type: QueryType | str) -> str:
    """
    Select the appropriate reranking instruction for a query type.

    Args:
        query_type: QueryType enum or string ("general", "summary", "tools", "troubleshooting")

    Returns:
        Reranking instruction string

    Examples:
        >>> select_rerank_instruction(QueryType.TOOLS)
        'Prioritize documents that:\\n1. Contain step-by-step...'

        >>> select_rerank_instruction("troubleshooting")
        'Prioritize documents that:\\n1. Address the specific error...'
    """
    if isinstance(query_type, str):
        query_type = QueryType(query_type.lower())

    return _INSTRUCTION_MAP.get(query_type, RERANK_INSTRUCTION_GENERAL)


def get_rerank_instruction_for_query(query: str) -> str:
    """
    Convenience function to detect query type and return appropriate instruction.

    Args:
        query: The user's search query

    Returns:
        Reranking instruction string

    Example:
        >>> instruction = get_rerank_instruction_for_query("How do I fix SSL error?")
        >>> # Returns RERANK_INSTRUCTION_TROUBLESHOOTING
    """
    query_type = detect_query_type(query)
    return select_rerank_instruction(query_type)

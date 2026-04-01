#!/usr/bin/env python3
"""
HybridRAG Hybrid Search Example
===============================

This example demonstrates advanced hybrid search capabilities:
1. Native MongoDB $rankFusion (M10+ tiers)
2. Manual RRF fusion (all tiers)
3. Filter builders for prefiltering
4. Query-type specific reranking

Prerequisites:
    pip install mongodb-hybridrag[all]

Run:
    python examples/02_hybrid_search.py
"""

import asyncio

# Add src to path for development
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybridrag.enhancements import (
    AtlasSearchFilterConfig,
    # Config
    MongoDBHybridSearchConfig,
    VectorSearchFilterConfig,
    build_atlas_search_filters,
    # Search functions
    build_vector_search_filters,
    # Utilities
    calculate_num_candidates,
    extract_pipeline_score,
)
from hybridrag.prompts import QueryType, detect_query_type, select_rerank_instruction


async def demo_query_type_detection() -> None:
    """Demonstrate query type detection for reranking."""
    print("\n" + "=" * 60)
    print("1. Query Type Detection")
    print("=" * 60)

    test_queries = [
        ("What is machine learning?", QueryType.SUMMARY),
        ("How do I configure SSL?", QueryType.TOOLS),
        ("Error: connection refused", QueryType.TROUBLESHOOTING),
        ("Find recent papers on transformers", QueryType.GENERAL),
    ]

    for query, expected in test_queries:
        detected = detect_query_type(query)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} '{query}'")
        print(f"      Detected: {detected.value}, Expected: {expected.value}")

        # Get reranking instruction
        instruction = select_rerank_instruction(detected)
        print(f"      Instruction: {instruction[:50]}...")


async def demo_filter_builders() -> None:
    """Demonstrate filter builder usage."""
    print("\n" + "=" * 60)
    print("2. Filter Builders")
    print("=" * 60)

    # Example: Filter messages from last 7 days by specific sender
    now = datetime.now()
    week_ago = now - timedelta(days=7)

    # Vector Search Filter (standard MongoDB operators)
    vector_config = VectorSearchFilterConfig(
        timestamp_field="timestamp",
        start_date=week_ago,
        end_date=now,
        equality_filters={"senderName": "John"},
        in_filters={"category": ["tech", "science"]},
    )

    vector_filter = build_vector_search_filters(vector_config)
    print("\n  Vector Search Filter (MongoDB operators):")
    print(f"    {vector_filter}")

    # Atlas Search Filter (Atlas-specific operators)
    atlas_config = AtlasSearchFilterConfig(
        timestamp_field="timestamp",
        start_date=week_ago,
        end_date=now,
        equality_filters={"senderName": "John"},
        in_filters={"category": ["tech", "science"]},
    )

    atlas_clauses = build_atlas_search_filters(atlas_config)
    print("\n  Atlas Search Filter (Atlas operators):")
    for clause in atlas_clauses:
        print(f"    {clause}")


async def demo_search_config() -> None:
    """Demonstrate search configuration options."""
    print("\n" + "=" * 60)
    print("3. Search Configuration")
    print("=" * 60)

    # Default config
    default_config = MongoDBHybridSearchConfig()
    print("\n  Default Configuration:")
    print(f"    Vector weight: {default_config.vector_weight}")
    print(f"    Text weight: {default_config.text_weight}")
    print(f"    Fuzzy max edits: {default_config.fuzzy_max_edits}")
    print(f"    Cosine threshold: {default_config.cosine_threshold}")

    # Custom config for precision-focused search
    precision_config = MongoDBHybridSearchConfig(
        vector_weight=0.7,  # Favor semantic similarity
        text_weight=0.3,  # Lower keyword influence
        fuzzy_max_edits=1,  # Stricter fuzzy matching
        cosine_threshold=0.5,  # Higher threshold
    )
    print("\n  Precision Configuration:")
    print(f"    Vector weight: {precision_config.vector_weight}")
    print(f"    Text weight: {precision_config.text_weight}")
    print(f"    Cosine threshold: {precision_config.cosine_threshold}")

    # Dynamic numCandidates calculation
    for top_k in [5, 10, 20]:
        num_candidates = calculate_num_candidates(top_k)
        print(f"\n  top_k={top_k} -> numCandidates={num_candidates} (20x multiplier)")


async def demo_multi_field_search() -> None:
    """Demonstrate multi-field weighted search."""
    print("\n" + "=" * 60)
    print("4. Multi-Field Weighted Search")
    print("=" * 60)

    # Configure field weights for WhatsApp-style messages
    field_weights = {
        "content": {"weight": 1.0},  # Main content (highest)
        "senderName": {"weight": 0.5},  # Sender name (medium)
        "topics": {"weight": 0.3},  # Topics array (lower)
    }

    print("\n  Field weights configuration:")
    for field, config in field_weights.items():
        print(f"    {field}: {config['weight']}")

    print("\n  This enables weighted search across multiple text fields")
    print("  with Atlas Search's score boosting.")


async def demo_score_extraction() -> None:
    """Demonstrate per-pipeline score extraction."""
    print("\n" + "=" * 60)
    print("5. Per-Pipeline Score Extraction")
    print("=" * 60)

    # Simulated scoreDetails from $rankFusion
    score_details = {
        "value": 0.85,
        "details": [
            {"inputPipelineName": "vector", "value": 0.92},
            {"inputPipelineName": "text", "value": 0.78},
        ],
    }

    print("\n  Simulated scoreDetails from $rankFusion:")
    print(f"    Combined score: {score_details['value']}")

    # Extract individual pipeline scores
    vector_score = extract_pipeline_score(score_details, "vector")
    text_score = extract_pipeline_score(score_details, "text")

    print("\n  Extracted scores:")
    print(f"    Vector pipeline: {vector_score}")
    print(f"    Text pipeline: {text_score}")

    # Calculate contribution
    total = vector_score + text_score
    if total > 0:
        print("\n  Contribution analysis:")
        print(f"    Vector: {vector_score / total * 100:.1f}%")
        print(f"    Text: {text_score / total * 100:.1f}%")


async def main() -> None:
    """Run all hybrid search demos."""
    print("\n" + "=" * 60)
    print("HybridRAG Hybrid Search Examples")
    print("=" * 60)

    await demo_query_type_detection()
    await demo_filter_builders()
    await demo_search_config()
    await demo_multi_field_search()
    await demo_score_extraction()

    print("\n" + "=" * 60)
    print("✓ All hybrid search demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

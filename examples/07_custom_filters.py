"""
Example 07: Custom Filters

Demonstrates:
- Vector search prefiltering (MongoDB standard syntax)
- Atlas Search filtering (Atlas-specific syntax)
- Date range filtering
- Metadata filtering
- Combining filters

This example only builds filter objects and shows where to attach them in
MongoDB search pipelines. It does not require a live database connection.
"""

import asyncio
from datetime import UTC, datetime, timedelta


async def example_vector_search_filters():
    """Vector search with prefiltering (standard MongoDB syntax)."""
    from hybridrag.enhancements import (
        VectorSearchFilterConfig,
        build_vector_search_filters,
    )

    print("=" * 60)
    print("Example 1: Vector Search Filters (Standard MongoDB)")
    print("=" * 60)

    # Filter by category
    filter_config = VectorSearchFilterConfig(
        equality_filters={"metadata.category": "features"}
    )

    filters = build_vector_search_filters(filter_config)

    print("\nFilter: Only 'features' category")
    print(f"MongoDB filter syntax: {filters}\n")

    # Note: This would be used in the vector search pipeline
    # For this example, we'll show the filter structure
    print("Example usage:")
    print("""
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "filter": filters,  # ← Applied here
                "limit": 10
            }
        }
    ]
    """)


async def example_atlas_search_filters():
    """Atlas Search with compound filters (Atlas-specific syntax)."""
    from hybridrag.enhancements import (
        AtlasSearchFilterConfig,
        build_atlas_search_filters,
    )

    print("=" * 60)
    print("Example 2: Atlas Search Filters (Atlas-Specific)")
    print("=" * 60)

    # Filter by multiple fields
    filter_config = AtlasSearchFilterConfig(
        equality_filters={
            "metadata.source": "mongodb_docs",
            "metadata.language": "en",
        }
    )

    filters = build_atlas_search_filters(filter_config)

    print("\nFilter: source='mongodb_docs' AND language='en'")
    print(f"Atlas Search filter syntax: {filters}\n")

    print("Example usage:")
    print("""
    pipeline = [
        {
            "$search": {
                "compound": {
                    "must": [
                        {"text": {"query": query, "path": "content"}}
                    ],
                    "filter": filters  # ← Applied here
                }
            }
        }
    ]
    """)


async def example_date_range_filters():
    """Filter by date range."""
    from hybridrag.enhancements import VectorSearchFilterConfig

    print("=" * 60)
    print("Example 3: Date Range Filters")
    print("=" * 60)

    # Filter documents from last 14 days
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=14)

    # Vector search date range (standard MongoDB $gte, $lte)
    filter_config = VectorSearchFilterConfig(
        range_filters={
            "metadata.timestamp": {
                "$gte": start_date,
                "$lte": end_date,
            }
        }
    )

    print("\nFilter: Last 14 days")
    print(f"Start: {start_date.isoformat()}")
    print(f"End: {end_date.isoformat()}\n")

    # Atlas Search date range (different syntax)
    from hybridrag.enhancements import AtlasSearchFilterConfig

    atlas_filter = AtlasSearchFilterConfig(
        range_filters={
            "metadata.timestamp": {
                "gte": start_date,
                "lte": end_date,
            }
        }
    )

    print("Vector Search syntax (MongoDB):")
    print(f"  {filter_config.range_filters}")
    print("\nAtlas Search syntax (Atlas):")
    print(f"  {atlas_filter.range_filters}")


async def example_in_filters():
    """Filter with multiple allowed values."""
    from hybridrag.enhancements import VectorSearchFilterConfig

    print("\n" + "=" * 60)
    print("Example 4: IN Filters (Multiple Values)")
    print("=" * 60)

    # Filter by multiple categories
    filter_config = VectorSearchFilterConfig(
        in_filters={"metadata.category": ["features", "platform"]}
    )

    print("\nFilter: category IN ['features', 'platform']")
    print(f"MongoDB syntax: {filter_config.in_filters}\n")

    print("Example usage:")
    print("""
    # This will match documents with category='features' OR category='platform'
    filters = build_vector_search_filters(filter_config)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "filter": filters,
                "limit": 10,
            }
        }
    ]
    """)


async def example_combined_filters():
    """Combine multiple filter types."""
    from datetime import datetime, timedelta

    from hybridrag.enhancements import VectorSearchFilterConfig

    print("=" * 60)
    print("Example 5: Combined Filters")
    print("=" * 60)

    from hybridrag.enhancements import build_vector_search_filters

    # Combine equality, range, and IN filters
    combined_config = VectorSearchFilterConfig(
        equality_filters={"metadata.source": "mongodb_docs"},
        range_filters={
            "metadata.timestamp": {"$gte": datetime.now(UTC) - timedelta(days=30)}
        },
        in_filters={"metadata.category": ["features", "platform"]},
    )
    combined_filters = build_vector_search_filters(combined_config)

    print("\nCombined filter:")
    print("  - source = 'mongodb_docs'")
    print("  - timestamp >= 30 days ago")
    print("  - category IN ['features', 'platform']")
    print(f"  MongoDB filter: {combined_filters}\n")

    print("All conditions must match (AND logic)\n")


async def example_practical_use_case():
    """Practical example: Search with filters."""
    from hybridrag.enhancements import VectorSearchFilterConfig

    print("=" * 60)
    print("Example 6: Practical Use Case")
    print("=" * 60)

    from hybridrag.enhancements import build_vector_search_filters

    query = "How does search work?"

    # Search only in 'features' category
    practical_config = VectorSearchFilterConfig(
        equality_filters={"metadata.category": "features"}
    )
    practical_filters = build_vector_search_filters(practical_config)

    print(f"\nQuery: {query}")
    print("Filter: category='features'")
    print(f"  MongoDB filter: {practical_filters}\n")

    print("This would search only in documents with category='features',")
    print("excluding documents about pricing, platform, etc.\n")

    print("Benefits:")
    print("  ✓ Faster search (fewer documents to scan)")
    print("  ✓ More relevant results")
    print("  ✓ Reduced noise from unrelated content")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HybridRAG Example 07: Custom Filters")
    print("=" * 60)

    try:
        # Run examples
        await example_vector_search_filters()
        await example_atlas_search_filters()
        await example_date_range_filters()
        await example_in_filters()
        await example_combined_filters()
        await example_practical_use_case()

        print("\n" + "=" * 60)
        print("All examples complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  - Vector search uses MongoDB standard syntax ($eq, $gte, $in)")
        print("  - Atlas Search uses Atlas-specific syntax (equals, range)")
        print("  - Both filter systems are type-safe and builder-based")
        print("  - Filters improve performance and relevance")
    finally:
        print("\nExample finished without touching a live database.")


if __name__ == "__main__":
    asyncio.run(main())

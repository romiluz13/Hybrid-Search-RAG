"""
Example 07: Custom Filters

Demonstrates:
- Vector search prefiltering (MongoDB standard syntax)
- Atlas Search filtering (Atlas-specific syntax)
- Date range filtering
- Metadata filtering
- Combining filters

Prerequisites:
- MONGODB_URI in .env
- VOYAGE_API_KEY in .env
- Documents with metadata fields
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
if not os.getenv("MONGODB_URI"):
    raise ValueError("MONGODB_URI not set in environment")
if not os.getenv("VOYAGE_API_KEY"):
    raise ValueError("VOYAGE_API_KEY not set in environment")


async def setup_test_data():
    """Insert sample documents with metadata."""
    from hybridrag import create_hybridrag

    rag = await create_hybridrag()

    # Sample documents with different metadata
    documents = [
        {
            "content": "MongoDB Atlas is a cloud database service for modern applications.",
            "metadata": {
                "source": "mongodb_docs",
                "category": "platform",
                "topic": "atlas",
                "language": "en",
                "timestamp": datetime.now(UTC) - timedelta(days=1),
            },
        },
        {
            "content": "Vector search in MongoDB Atlas enables semantic similarity searches.",
            "metadata": {
                "source": "mongodb_docs",
                "category": "features",
                "topic": "vector_search",
                "language": "en",
                "timestamp": datetime.now(UTC) - timedelta(days=7),
            },
        },
        {
            "content": "Atlas Search provides full-text search with fuzzy matching.",
            "metadata": {
                "source": "mongodb_docs",
                "category": "features",
                "topic": "atlas_search",
                "language": "en",
                "timestamp": datetime.now(UTC) - timedelta(days=30),
            },
        },
        {
            "content": "MongoDB pricing varies by cluster tier and storage usage.",
            "metadata": {
                "source": "mongodb_docs",
                "category": "pricing",
                "topic": "costs",
                "language": "en",
                "timestamp": datetime.now(UTC) - timedelta(days=60),
            },
        },
    ]

    print("Setting up test data...")
    for doc in documents:
        await rag.insert(doc["content"], metadata=doc["metadata"])

    print(f"✓ Inserted {len(documents)} test documents\n")
    return rag


async def example_vector_search_filters():
    """Vector search with prefiltering (standard MongoDB syntax)."""
    from hybridrag.enhancements import (
        VectorSearchFilterConfig,
        build_vector_search_filters,
    )

    print("=" * 60)
    print("Example 1: Vector Search Filters (Standard MongoDB)")
    print("=" * 60)

    await setup_test_data()

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
    results = await rag.query_with_filters(
        query="mongodb search",
        filter_config=filter_config,
        mode="vector",
    )
    """)


async def example_combined_filters():
    """Combine multiple filter types."""
    from datetime import datetime, timedelta

    from hybridrag.enhancements import VectorSearchFilterConfig

    print("=" * 60)
    print("Example 5: Combined Filters")
    print("=" * 60)

    # Combine equality, range, and IN filters
    VectorSearchFilterConfig(
        equality_filters={"metadata.source": "mongodb_docs"},
        range_filters={
            "metadata.timestamp": {"$gte": datetime.now(UTC) - timedelta(days=30)}
        },
        in_filters={"metadata.category": ["features", "platform"]},
    )

    print("\nCombined filter:")
    print("  - source = 'mongodb_docs'")
    print("  - timestamp >= 30 days ago")
    print("  - category IN ['features', 'platform']\n")

    print("All conditions must match (AND logic)\n")


async def example_practical_use_case():
    """Practical example: Search with filters."""
    from hybridrag.enhancements import VectorSearchFilterConfig

    print("=" * 60)
    print("Example 6: Practical Use Case")
    print("=" * 60)

    await setup_test_data()

    query = "How does search work?"

    # Search only in 'features' category
    VectorSearchFilterConfig(equality_filters={"metadata.category": "features"})

    print(f"\nQuery: {query}")
    print("Filter: category='features'\n")

    print("This would search only in documents with category='features',")
    print("excluding documents about pricing, platform, etc.\n")

    print("Benefits:")
    print("  ✓ Faster search (fewer documents to scan)")
    print("  ✓ More relevant results")
    print("  ✓ Reduced noise from unrelated content")


async def main():
    """Run all examples."""
    from hybridrag.core.mongodb_client import close_shared_client

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
        close_shared_client()


if __name__ == "__main__":
    asyncio.run(main())

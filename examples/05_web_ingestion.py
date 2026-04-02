"""
Example 05: Web Content Ingestion

Demonstrates:
- Ingesting content from URLs
- Website crawling with max_pages limit
- Progress tracking
- Error handling for web scraping

Prerequisites:
- MONGODB_URI in .env
- VOYAGE_API_KEY in .env
- tavily-python installed (pip install tavily-python)
"""

import asyncio
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
if not os.getenv("MONGODB_URI"):
    raise ValueError("MONGODB_URI not set in environment")
if not os.getenv("VOYAGE_API_KEY"):
    raise ValueError("VOYAGE_API_KEY not set in environment")


async def example_single_url():
    """Ingest content from a single URL."""
    from hybridrag import create_hybridrag

    print("=" * 60)
    print("Example 1: Single URL Ingestion")
    print("=" * 60)

    rag = await create_hybridrag()

    # Ingest MongoDB documentation page
    url = (
        "https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/"
    )

    print(f"\nIngesting URL: {url}")
    print("This may take a few seconds...\n")

    result = await rag.ingest_url(url)

    if result.success:
        print(f"✓ Successfully ingested: {result.title}")
        print(f"  Chunks created: {result.chunks_created}")
        print(f"  Document ID: {result.document_id}")
    else:
        print(f"✗ Failed to ingest: {result.errors}")

    # Query the ingested content
    print("\nQuerying ingested content...")
    answer = await rag.query(
        query="What is Atlas Vector Search?",
        mode="mix",
        top_k=5,
    )

    print("\nAnswer:")
    print(answer)


async def example_website_crawl():
    """Crawl and ingest multiple pages from a website."""
    from hybridrag import create_hybridrag

    print("\n" + "=" * 60)
    print("Example 2: Website Crawling")
    print("=" * 60)

    rag = await create_hybridrag()

    # Crawl MongoDB Atlas documentation
    base_url = "https://www.mongodb.com/docs/atlas/atlas-vector-search/"
    max_pages = 5  # Limit to 5 pages for this example

    print(f"\nCrawling website: {base_url}")
    print(f"Max pages: {max_pages}")
    print("This may take a minute...\n")

    def progress_callback(current: int, total: int):
        """Track crawling progress."""
        print(f"  Processing page {current}/{total}...")

    results = await rag.ingest_website(
        url=base_url,
        max_pages=max_pages,
        progress_callback=progress_callback,
    )

    # Summary
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_created for r in results)

    print("\n✓ Crawl complete:")
    print(f"  Pages ingested: {successful}/{len(results)}")
    print(f"  Total chunks: {total_chunks}")

    if successful < len(results):
        failed = [r.title for r in results if not r.success]
        print(f"\n  Failed pages: {', '.join(failed)}")

    # Query across all crawled pages
    print("\nQuerying across all pages...")
    answer = await rag.query(
        query="How do I create a vector search index in Atlas?",
        mode="mix",
        top_k=10,
    )

    print("\nAnswer:")
    print(answer)


async def example_batch_urls():
    """Ingest multiple specific URLs in parallel."""
    from hybridrag import create_hybridrag

    print("\n" + "=" * 60)
    print("Example 3: Batch URL Ingestion")
    print("=" * 60)

    rag = await create_hybridrag()

    # List of specific URLs to ingest
    urls = [
        "https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/",
        "https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/",
        "https://www.mongodb.com/docs/atlas/atlas-search/",
    ]

    print(f"\nIngesting {len(urls)} URLs in parallel...")
    print("This may take a few seconds...\n")

    # Ingest all URLs concurrently
    tasks = [rag.ingest_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"✗ Failed: {urls[i]}")
            print(f"  Error: {result}")
        elif result.success:
            print(f"✓ Success: {result.title}")
            print(f"  Chunks: {result.chunks_created}")
            successful += 1
        else:
            print(f"✗ Failed: {urls[i]}")
            print(f"  Errors: {result.errors}")

    print(f"\n✓ Batch ingestion complete: {successful}/{len(urls)} successful")


async def example_error_handling():
    """Demonstrate error handling for web ingestion."""
    from hybridrag import create_hybridrag

    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    rag = await create_hybridrag()

    # Test cases
    test_cases = [
        ("Valid URL", "https://www.mongodb.com/docs/atlas/"),
        ("Invalid URL", "https://this-url-definitely-does-not-exist-12345.com"),
        (
            "Non-HTML content",
            "https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/robots.txt",
        ),
    ]

    print("\nTesting various scenarios...\n")

    for description, url in test_cases:
        print(f"{description}: {url}")

        try:
            result = await rag.ingest_url(url)

            if result.success:
                print(f"  ✓ Success - {result.chunks_created} chunks")
            else:
                print(f"  ✗ Failed - {result.errors}")
        except Exception as e:
            print(f"  ✗ Exception - {type(e).__name__}: {e}")

        print()


async def main():
    """Run all examples."""
    from hybridrag.core.mongodb_client import close_shared_client

    print("\n" + "=" * 60)
    print("HybridRAG Example 05: Web Content Ingestion")
    print("=" * 60)

    try:
        # Run examples
        await example_single_url()
        await example_website_crawl()
        await example_batch_urls()
        await example_error_handling()

        print("\n" + "=" * 60)
        print("All examples complete!")
        print("=" * 60)
    finally:
        close_shared_client()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
HybridRAG Quickstart Example
============================

This example demonstrates the basic usage of HybridRAG:
1. Initialize the RAG system
2. Ingest documents
3. Query the knowledge base

Prerequisites:
    pip install mongodb-hybridrag[all]

    # Configure .env with:
    # MONGODB_URI=mongodb+srv://...
    # VOYAGE_API_KEY=pa-...
    # ANTHROPIC_API_KEY=sk-ant-... (or other LLM)

Run:
    python examples/01_quickstart.py
"""

import asyncio

# Add src to path for development
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybridrag import create_hybridrag


async def main() -> None:
    """Basic HybridRAG usage example."""
    from hybridrag.core.mongodb_client import close_shared_client

    print("=" * 60)
    print("HybridRAG Quickstart")
    print("=" * 60)

    # 1. Initialize HybridRAG
    print("\n1. Initializing HybridRAG...")
    rag = await create_hybridrag()
    print("   ✓ HybridRAG initialized")

    try:
        # 2. Sample documents to ingest
        documents = [
            """
            MongoDB Atlas is a fully managed cloud database service that handles
            deployment, maintenance, and scaling of MongoDB clusters. It provides
            built-in vector search capabilities through Atlas Search, enabling
            semantic search across documents using embeddings.
            """,
            """
            Voyage AI provides state-of-the-art embedding models optimized for
            retrieval. The voyage-3-large model offers 1024-dimensional embeddings
            with excellent performance on semantic search tasks. They also offer
            rerank-2.5 for cross-encoder reranking.
            """,
            """
            Hybrid search combines vector similarity search with keyword-based
            search. MongoDB Atlas supports this through $rankFusion which merges
            results from $vectorSearch and $search pipelines using Reciprocal
            Rank Fusion (RRF) algorithm.
            """,
        ]

        # 3. Ingest documents
        print("\n2. Ingesting documents...")
        await rag.insert(documents)
        print(f"   ✓ Ingested {len(documents)} documents")

        # 4. Query the knowledge base
        queries = [
            "What is MongoDB Atlas?",
            "How does hybrid search work?",
            "What embedding model does Voyage AI provide?",
        ]

        print("\n3. Querying the knowledge base...")
        print("-" * 60)

        for query in queries:
            print(f"\nQ: {query}")

            # Query with default parameters
            response = await rag.query(
                query=query,
                mode="mix",  # Use the recommended all-signals retrieval mode
                top_k=3,
            )

            print(f"A: {response[:500]}...")  # Truncate for readability
            print("-" * 60)

        print("\n✓ Quickstart complete!")
    finally:
        close_shared_client()


if __name__ == "__main__":
    asyncio.run(main())

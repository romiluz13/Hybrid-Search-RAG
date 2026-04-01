"""
Example 06: Conversation Memory

Demonstrates:
- Multi-turn conversations with memory
- Session management
- Context preservation across queries
- Memory retrieval and summarization

Prerequisites:
- MONGODB_URI in .env
- VOYAGE_API_KEY in .env
- ANTHROPIC_API_KEY in .env
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = ["MONGODB_URI", "VOYAGE_API_KEY", "ANTHROPIC_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} not set in environment")


async def example_basic_conversation():
    """Basic conversation with memory."""
    from hybridrag import create_hybridrag

    print("=" * 60)
    print("Example 1: Basic Conversation with Memory")
    print("=" * 60)

    rag = await create_hybridrag()
    session_id = str(uuid.uuid4())

    print(f"\nSession ID: {session_id[:8]}...\n")

    # First query
    print("Turn 1: What is MongoDB Atlas?")
    result1 = await rag.query_with_memory(
        query="What is MongoDB Atlas?",
        session_id=session_id,
        mode="hybrid",
        top_k=5,
    )
    print(f"Answer: {result1['answer'][:200]}...\n")

    # Follow-up query (uses memory of previous context)
    print("Turn 2: What are its main features?")
    result2 = await rag.query_with_memory(
        query="What are its main features?",
        session_id=session_id,
        mode="hybrid",
        top_k=5,
    )
    print(f"Answer: {result2['answer'][:200]}...\n")

    # Another follow-up (pronoun resolution)
    print("Turn 3: How much does it cost?")
    result3 = await rag.query_with_memory(
        query="How much does it cost?",
        session_id=session_id,
        mode="hybrid",
        top_k=5,
    )
    print(f"Answer: {result3['answer'][:200]}...\n")


async def example_session_management():
    """Demonstrate session isolation."""
    from hybridrag import create_hybridrag

    print("=" * 60)
    print("Example 2: Session Management")
    print("=" * 60)

    rag = await create_hybridrag()

    # Create two separate sessions
    session1 = str(uuid.uuid4())
    session2 = str(uuid.uuid4())

    print(f"\nSession 1: {session1[:8]}...")
    print(f"Session 2: {session2[:8]}...\n")

    # Session 1: Talk about vector search
    print("Session 1, Turn 1: Tell me about vector search")
    result1 = await rag.query_with_memory(
        query="Tell me about vector search in MongoDB",
        session_id=session1,
        mode="hybrid",
    )
    print(f"Answer: {result1['answer'][:150]}...\n")

    # Session 2: Talk about full-text search
    print("Session 2, Turn 1: Tell me about full-text search")
    result2 = await rag.query_with_memory(
        query="Tell me about full-text search in MongoDB",
        session_id=session2,
        mode="hybrid",
    )
    print(f"Answer: {result2['answer'][:150]}...\n")

    # Session 1: Follow-up (should remember vector search context)
    print("Session 1, Turn 2: How does it work?")
    result3 = await rag.query_with_memory(
        query="How does it work?",
        session_id=session1,
        mode="hybrid",
    )
    print(f"Answer: {result3['answer'][:150]}...")
    print("  → Should be about vector search\n")

    # Session 2: Follow-up (should remember full-text context)
    print("Session 2, Turn 2: How does it work?")
    result4 = await rag.query_with_memory(
        query="How does it work?",
        session_id=session2,
        mode="hybrid",
    )
    print(f"Answer: {result4['answer'][:150]}...")
    print("  → Should be about full-text search\n")


async def example_history_retrieval():
    """Retrieve and display conversation history."""
    from hybridrag import create_hybridrag

    print("=" * 60)
    print("Example 3: Conversation History")
    print("=" * 60)

    rag = await create_hybridrag()
    session_id = str(uuid.uuid4())

    print(f"\nSession ID: {session_id[:8]}...\n")

    # Have a conversation
    queries = [
        "What is HybridRAG?",
        "How does it combine vector and keyword search?",
        "What are the benefits?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"Turn {i}: {query}")
        await rag.query_with_memory(
            query=query,
            session_id=session_id,
            mode="hybrid",
        )

    # Retrieve history
    print("\n" + "-" * 60)
    print("Conversation History:")
    print("-" * 60 + "\n")

    history = await rag.get_conversation_history(session_id, limit=10)

    for i, message in enumerate(history):
        role = message.get("role", "unknown").upper()
        content = message.get("content", "")[:100]
        timestamp = message.get("timestamp", "N/A")

        print(f"[{i + 1}] {role} ({timestamp}):")
        print(f"    {content}...")
        print()


async def example_memory_summarization():
    """Demonstrate memory summarization for long conversations."""
    from hybridrag import create_hybridrag
    from hybridrag.prompts import MEMORY_SUMMARIZATION_PROMPT

    print("=" * 60)
    print("Example 4: Memory Summarization")
    print("=" * 60)

    rag = await create_hybridrag()
    session_id = str(uuid.uuid4())

    print(f"\nSession ID: {session_id[:8]}...\n")

    # Simulate a long conversation (10 turns)
    print("Simulating 10-turn conversation...\n")

    queries = [
        "What is MongoDB?",
        "What is Atlas?",
        "Tell me about vector search",
        "How does it compare to Pinecone?",
        "What about full-text search?",
        "Can I use both?",
        "How do I set up indexes?",
        "What's the pricing?",
        "Is there a free tier?",
        "How do I get started?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"  Turn {i}: {query}")
        await rag.query_with_memory(
            query=query,
            session_id=session_id,
            mode="hybrid",
        )

    # Get history
    history = await rag.get_conversation_history(session_id, limit=20)

    print(f"\n✓ Conversation complete: {len(history)} messages")

    # Show memory summarization prompt (used internally)
    print("\n" + "-" * 60)
    print("Memory Summarization Prompt (used internally):")
    print("-" * 60)
    print(MEMORY_SUMMARIZATION_PROMPT[:300] + "...\n")

    print("This prompt is used to summarize long conversations")
    print("to prevent context window overflow.\n")


async def example_clear_session():
    """Clear conversation memory for a session."""
    from hybridrag import create_hybridrag

    print("=" * 60)
    print("Example 5: Clear Session Memory")
    print("=" * 60)

    rag = await create_hybridrag()
    session_id = str(uuid.uuid4())

    print(f"\nSession ID: {session_id[:8]}...\n")

    # Have a conversation
    print("Turn 1: What is MongoDB?")
    await rag.query_with_memory(
        query="What is MongoDB?",
        session_id=session_id,
    )

    print("Turn 2: Tell me more")
    await rag.query_with_memory(
        query="Tell me more",
        session_id=session_id,
    )

    # Check history
    history_before = await rag.get_conversation_history(session_id)
    print(f"\n✓ History before clear: {len(history_before)} messages\n")

    # Clear session
    print("Clearing session memory...")
    await rag.clear_conversation_memory(session_id)

    history_after = await rag.get_conversation_history(session_id)
    print(f"✓ History after clear: {len(history_after)} messages\n")

    # New query (should not have context)
    print("Turn 3 (after clear): Tell me more")
    await rag.query_with_memory(
        query="Tell me more",
        session_id=session_id,
    )
    print("  → Should ask for clarification (no context)")


async def main():
    """Run all examples."""
    from hybridrag.core.mongodb_client import close_shared_client

    print("\n" + "=" * 60)
    print("HybridRAG Example 06: Conversation Memory")
    print("=" * 60)

    try:
        # Run examples
        await example_basic_conversation()
        await example_session_management()
        await example_history_retrieval()
        await example_memory_summarization()
        await example_clear_session()

        print("\n" + "=" * 60)
        print("All examples complete!")
        print("=" * 60)
    finally:
        close_shared_client()


if __name__ == "__main__":
    asyncio.run(main())

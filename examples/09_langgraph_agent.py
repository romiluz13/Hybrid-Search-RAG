#!/usr/bin/env python3
"""
LangGraph Agent with HybridRAG Tool
====================================

This example demonstrates how to use HybridRAG as a tool within a LangGraph agent,
showing developers how to build AI agents on the HybridRAG foundation.

Key Concepts:
1. HybridRAG as a Tool - Wrap the complete RAG pipeline as a LangChain tool
2. Agent State Management - Use LangGraph's stateful message handling
3. ReAct Pattern - Agent decides WHEN to use knowledge base vs. answer directly
4. Multi-turn Conversations - Maintain conversation context across queries

This is the "user side" of HybridRAG - demonstrating how to take the boilerplate
and build sophisticated agents on top of it.

Prerequisites:
    pip install mongodb-hybridrag[agent] langchain-google-genai

    # Configure .env with:
    # MONGODB_URI=mongodb+srv://...
    # VOYAGE_API_KEY=pa-...
    # GEMINI_API_KEY=AIza...  (or ANTHROPIC_API_KEY for Claude)

Run:
    python examples/09_langgraph_agent.py

Architecture:
    User Query -> LangGraph Agent -> [Decide: Use Tool?]
                                          |
                    +---------------------+---------------------+
                    |                                           |
              [Yes: Need KB]                           [No: Answer Directly]
                    |                                           |
                    v                                           v
            HybridRAG Tool                              LLM Response
            (query -> embed -> search -> rerank -> generate)
                    |
                    v
            Agent Response (with sources)
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv

# Load .env file before checking environment variables
load_dotenv()

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check for LangGraph dependencies
try:
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
except ImportError as e:
    print("ERROR: LangGraph dependencies not installed.")
    print("Install with: pip install mongodb-hybridrag[agent]")
    print(f"Missing: {e}")
    sys.exit(1)

from hybridrag import create_hybridrag  # noqa: E402

# LLM provider detection (Gemini preferred, Anthropic fallback)
if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI

    _LLMClass = type[ChatGoogleGenerativeAI] | type[ChatAnthropic]

_llm_provider: str | None = None
_llm_class: _LLMClass | None = None

# Check for Gemini first (user has GEMINI_API_KEY)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        _llm_provider = "gemini"
        _llm_class = ChatGoogleGenerativeAI
        # Set the API key for langchain-google-genai
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gemini_key
except ImportError:
    pass

# Fallback to Anthropic
if _llm_provider is None:
    try:
        from langchain_anthropic import ChatAnthropic

        if os.getenv("ANTHROPIC_API_KEY"):
            _llm_provider = "anthropic"
            _llm_class = ChatAnthropic
    except ImportError:
        pass

if _llm_provider is None:
    print("ERROR: No LLM provider available.")
    print("Set GEMINI_API_KEY or ANTHROPIC_API_KEY in your .env file")
    print("Install with: pip install langchain-google-genai  (for Gemini)")
    print("         or: pip install langchain-anthropic  (for Claude)")
    sys.exit(1)

print(f"Using LLM provider: {_llm_provider}")

# Global HybridRAG instance (initialized once)
_hybridrag_instance = None


# =============================================================================
# STEP 1: Define Agent State
# =============================================================================


class AgentState(TypedDict):
    """State for the HybridRAG agent.

    The state contains:
    - messages: Conversation history with add_messages reducer
    - session_id: Optional session ID for multi-turn conversations
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str | None


# =============================================================================
# STEP 2: Create HybridRAG Tool
# =============================================================================


@tool
async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information.

    Use this tool when you need to find information from the ingested documents.
    The tool performs a complete RAG pipeline:
    1. Embeds the query using Voyage AI
    2. Performs hybrid search (vector + keyword) in MongoDB
    3. Reranks results using Voyage rerank-2.5
    4. Returns relevant information

    Args:
        query: The search query to find relevant information.

    Returns:
        Relevant information from the knowledge base, or a message if nothing found.
    """
    global _hybridrag_instance

    if _hybridrag_instance is None:
        return (
            "Error: Knowledge base not initialized. Please initialize HybridRAG first."
        )

    try:
        # Use query_with_sources to get both answer and context
        result = await _hybridrag_instance.query_with_sources(
            query=query,
            mode="mix",  # Use mix mode (KG + vector) for best results
            top_k=5,
        )

        # Format the response with context
        answer = result.get("answer", "No answer generated.")
        context = result.get("context", "")

        # If we got a good answer, return it
        if answer and len(answer) > 50:
            return f"Based on the knowledge base:\n\n{answer}"

        # If answer is short, include context for more info
        if context:
            return f"Found in knowledge base:\n\n{context[:2000]}"

        return "No relevant information found in the knowledge base."

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
async def get_document_stats() -> str:
    """Get statistics about the knowledge base.

    Use this tool when the user asks about what documents are available,
    how many documents are in the system, or wants an overview of the knowledge base.

    Returns:
        Statistics about documents, entities, and relationships in the knowledge base.
    """
    global _hybridrag_instance

    if _hybridrag_instance is None:
        return "Error: Knowledge base not initialized."

    try:
        stats = await _hybridrag_instance.get_knowledge_base_stats()

        return f"""Knowledge Base Statistics:
- Total Documents: {stats.get("documents", {}).get("total", 0)}
- Entities: {stats.get("entities", 0)}
- Relationships: {stats.get("relationships", 0)}
- Chunks: {stats.get("chunks", 0)}
"""
    except Exception as e:
        return f"Error getting stats: {str(e)}"


# =============================================================================
# STEP 3: Create Agent Nodes
# =============================================================================


async def agent_node(state: AgentState) -> dict[str, Any]:
    """The main agent node that decides whether to use tools or respond directly.

    This node:
    1. Takes the current conversation state
    2. Sends it to the LLM with tool definitions
    3. Returns the LLM's decision (tool call or direct response)
    """
    # Create LLM based on available provider
    # _llm_class is guaranteed non-None here (checked at module load)
    assert _llm_class is not None, "LLM class not initialized"
    if _llm_provider == "gemini":
        llm = _llm_class(
            model="gemini-2.0-flash",
            temperature=0,
        )
    else:  # anthropic
        llm = _llm_class(
            model="claude-sonnet-4-20250514",
            temperature=0,
        )

    tools = [search_knowledge_base, get_document_stats]
    llm_with_tools = llm.bind_tools(tools)

    # System message for the agent
    system_message = """You are a helpful AI assistant with access to a knowledge base.

Your capabilities:
1. search_knowledge_base: Search for specific information in the ingested documents
2. get_document_stats: Get statistics about the knowledge base

Guidelines:
- Use search_knowledge_base when users ask questions that require information from documents
- Use get_document_stats when users ask about what's in the knowledge base
- Answer directly (without tools) for:
  - General conversation and greetings
  - Questions you can answer from general knowledge
  - Follow-up questions about previous responses
  - Clarification requests

Be helpful, accurate, and cite sources when using the knowledge base."""

    # Prepare messages with system prompt
    messages = [{"role": "system", "content": system_message}] + list(state["messages"])

    # Get LLM response
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if the agent should continue to tools or end.

    Returns:
        "tools" if the last message has tool calls, "end" otherwise.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# =============================================================================
# STEP 4: Build the Agent Graph
# =============================================================================


def create_agent_graph() -> StateGraph:
    """Create the LangGraph agent with HybridRAG tools.

    The graph structure:
        [agent] --tool_calls--> [tools] --> [agent]
           |
           +--no_tool_calls--> [END]

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create tool node with our HybridRAG tools
    tools = [search_knowledge_base, get_document_stats]
    tool_node = ToolNode(tools)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Tools always return to agent
    graph.add_edge("tools", "agent")

    return graph.compile()


# =============================================================================
# STEP 5: Example Usage
# =============================================================================


async def initialize_knowledge_base() -> None:
    """Initialize HybridRAG and ingest sample documents."""
    global _hybridrag_instance

    print("\n1. Initializing HybridRAG...")
    _hybridrag_instance = await create_hybridrag()
    print("   Done: HybridRAG initialized")

    # Sample documents about AI and RAG
    documents = [
        """
        Retrieval-Augmented Generation (RAG) is a technique that combines retrieval
        and generation in AI systems. It works by first retrieving relevant documents
        from a knowledge base, then using those documents as context for a language
        model to generate accurate responses. RAG helps reduce hallucinations and
        keeps AI responses grounded in factual information.
        """,
        """
        MongoDB Atlas Vector Search enables semantic search by storing document
        embeddings and performing similarity searches. Combined with Atlas Search
        for keyword matching, it enables hybrid search that combines the best of
        both approaches. The $rankFusion operator merges results from multiple
        search pipelines using Reciprocal Rank Fusion (RRF).
        """,
        """
        Voyage AI provides state-of-the-art embedding models like voyage-3-large
        with 1024 dimensions. Their rerank-2.5 model is a cross-encoder that
        reranks search results for better relevance. Using embeddings + reranking
        significantly improves RAG quality compared to embeddings alone.
        """,
        """
        LangGraph is a library for building stateful, multi-actor applications
        with LLMs. It extends LangChain with graph-based workflows, enabling
        complex agent architectures with cycles, branching, and persistent state.
        It's ideal for building agents that need to make decisions about tool use.
        """,
        """
        The ReAct (Reasoning + Acting) pattern is an agent architecture where
        the LLM reasons about what action to take, executes the action (like
        calling a tool), observes the result, and continues until the task is
        complete. This enables agents to dynamically decide when to use tools
        versus respond directly.
        """,
    ]

    print("\n2. Ingesting sample documents...")
    await _hybridrag_instance.insert(documents)
    print(f"   Done: Ingested {len(documents)} documents")


async def run_interactive_demo(agent: StateGraph) -> None:
    """Run an interactive demo of the agent."""
    print("\n" + "=" * 60)
    print("Interactive Agent Demo")
    print("=" * 60)
    print("Type your questions below. The agent will decide whether to")
    print("search the knowledge base or answer directly.")
    print("Type 'quit' or 'exit' to stop.\n")

    session_id = "demo-session"
    messages: list[BaseMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message
        messages.append(HumanMessage(content=user_input))

        # Run agent
        state = AgentState(messages=messages, session_id=session_id)
        result = await agent.ainvoke(state)

        # Extract assistant response
        final_messages = result["messages"]
        assistant_message = final_messages[-1]

        print(f"\nAssistant: {assistant_message.content}\n")

        # Update message history for multi-turn
        messages = list(final_messages)


async def run_demo_queries(agent: StateGraph) -> None:
    """Run predefined demo queries to showcase the agent."""
    demo_queries = [
        # Should use search_knowledge_base
        "What is RAG and how does it work?",
        # Should use search_knowledge_base
        "How does MongoDB Atlas handle hybrid search?",
        # Should use get_document_stats
        "What documents are in the knowledge base?",
        # Should answer directly (general knowledge)
        "What's 2 + 2?",
        # Follow-up (should use context from previous)
        "Can you tell me more about the ReAct pattern?",
    ]

    print("\n" + "=" * 60)
    print("Demo Queries")
    print("=" * 60)

    session_id = "demo-queries"
    messages: list[BaseMessage] = []

    for query in demo_queries:
        print(f"\n{'=' * 60}")
        print(f"User: {query}")
        print("-" * 60)

        messages.append(HumanMessage(content=query))

        state = AgentState(messages=messages, session_id=session_id)
        result = await agent.ainvoke(state)

        final_messages = result["messages"]
        assistant_message = final_messages[-1]

        # Show tool usage if any
        for msg in final_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[Tool Used: {tc['name']}]")

        print(f"\nAssistant: {assistant_message.content}")
        messages = list(final_messages)


async def main() -> None:
    """Main entry point for the LangGraph agent example."""
    from hybridrag.core.mongodb_client import close_shared_client

    print("=" * 60)
    print("LangGraph Agent with HybridRAG")
    print("=" * 60)
    print("\nThis example demonstrates building an AI agent that uses")
    print("HybridRAG as a tool for knowledge retrieval.\n")

    try:
        # Initialize knowledge base
        await initialize_knowledge_base()

        # Create the agent
        print("\n3. Creating LangGraph agent...")
        agent = create_agent_graph()
        print("   Done: Agent created with HybridRAG tools")

        # Run demo queries
        await run_demo_queries(agent)

        # Ask if user wants interactive mode
        print("\n" + "=" * 60)
        try:
            response = input("Start interactive mode? (y/n): ").strip().lower()
            if response in ("y", "yes"):
                await run_interactive_demo(agent)
        except (EOFError, KeyboardInterrupt):
            pass

        print("\nDone! Example complete.")
    finally:
        close_shared_client()


if __name__ == "__main__":
    asyncio.run(main())

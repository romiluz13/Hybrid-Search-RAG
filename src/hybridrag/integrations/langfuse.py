"""
Langfuse Observability Integration for HybridRAG.

Provides tracing and monitoring for:
- LLM calls (generation, entity extraction)
- Embedding calls (Voyage AI)
- RAG queries (full pipeline)
- Document ingestion

Usage:
    from hybridrag.integrations.langfuse import (
        get_langfuse,
        trace_rag_query,
        trace_llm_call,
        trace_embedding_call,
    )

Environment Variables:
    LANGFUSE_PUBLIC_KEY: Your Langfuse public key
    LANGFUSE_SECRET_KEY: Your Langfuse secret key
    LANGFUSE_HOST: Optional custom Langfuse host (default: https://cloud.langfuse.com)
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger("hybridrag.langfuse")

# M41: Langfuse availability check at import (no env reads).
# Environment variable reads are deferred to _check_langfuse_enabled()
# which is called on first use, not at import time.
# [Rule: mongodb-mcp-setup] Avoid import-time side effects.
LANGFUSE_AVAILABLE = False
_langfuse_enabled: bool | None = None  # None = not yet checked
_langfuse_client = None

try:
    from langfuse import Langfuse
    from langfuse.decorators import langfuse_context, observe

    LANGFUSE_AVAILABLE = True

except ImportError:
    logger.debug("[LANGFUSE] Package not installed, tracing disabled")
    Langfuse = None
    langfuse_context = None
    observe = None


def _check_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled (lazy -- reads env on first call only).

    M41: This defers LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY reads
    to first use rather than import time.
    """
    global _langfuse_enabled
    if _langfuse_enabled is None:
        if not LANGFUSE_AVAILABLE:
            _langfuse_enabled = False
        else:
            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            _langfuse_enabled = bool(public_key and secret_key)
            if _langfuse_enabled:
                logger.info("[LANGFUSE] Observability enabled")
            else:
                logger.debug("[LANGFUSE] Keys not configured, tracing disabled")
    return _langfuse_enabled


# Backward-compatible module-level name. External callers that do
# `from hybridrag.integrations.langfuse import LANGFUSE_ENABLED`
# will get False. Internal code uses _check_langfuse_enabled().
LANGFUSE_ENABLED = False


def get_langfuse() -> Langfuse | None:
    """Get or create the Langfuse client singleton."""
    global _langfuse_client

    if not _check_langfuse_enabled():
        return None

    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

    return _langfuse_client


def flush_langfuse() -> None:
    """Flush any pending Langfuse events."""
    if _langfuse_client is not None:
        _langfuse_client.flush()


@dataclass
class TraceContext:
    """Context for a Langfuse trace."""

    trace_id: str | None = None
    span_id: str | None = None
    start_time: float = 0.0
    metadata: dict[str, Any] | None = None


# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


def trace_rag_query(
    query: str,
    mode: str = "mix",
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a RAG query in Langfuse.

    Args:
        query: The user's query
        mode: Query mode (naive, local, global, hybrid, mix)
        metadata: Additional metadata to attach

    Example:
        @trace_rag_query(query="What is MongoDB?", mode="mix")
        async def my_query_function():
            ...
    """

    def decorator(func: F) -> F:
        if not _check_langfuse_enabled():
            return func

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            langfuse = get_langfuse()
            if langfuse is None:
                return await func(*args, **kwargs)

            trace = langfuse.trace(
                name="rag_query",
                input={"query": query, "mode": mode},
                metadata=metadata or {},
            )

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Extract response info
                output_info = {}
                if isinstance(result, dict):
                    output_info["answer_length"] = len(result.get("answer", ""))
                    output_info["context_length"] = len(result.get("context", ""))
                elif isinstance(result, str):
                    output_info["response_length"] = len(result)

                trace.update(
                    output=output_info,
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": round(duration, 3),
                        "status": "success",
                    },
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                trace.update(
                    output={"error": str(e)},
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": round(duration, 3),
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            langfuse = get_langfuse()
            if langfuse is None:
                return func(*args, **kwargs)

            trace = langfuse.trace(
                name="rag_query",
                input={"query": query, "mode": mode},
                metadata=metadata or {},
            )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                trace.update(
                    output={"response_length": len(str(result))},
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": round(duration, 3),
                        "status": "success",
                    },
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                trace.update(
                    output={"error": str(e)},
                    metadata={
                        **(metadata or {}),
                        "duration_seconds": round(duration, 3),
                        "status": "error",
                    },
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


@contextmanager
def trace_span(
    name: str,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager for tracing a span within a larger trace.

    Args:
        name: Name of the span (e.g., "embedding", "rerank", "llm_call")
        input_data: Input data for the span
        metadata: Additional metadata

    Example:
        with trace_span("embedding", input_data={"texts": texts}):
            embeddings = await embed(texts)
    """
    if not _check_langfuse_enabled():
        yield None
        return

    langfuse = get_langfuse()
    if langfuse is None:
        yield None
        return

    span = langfuse.span(
        name=name,
        input=input_data or {},
        metadata=metadata or {},
    )

    start_time = time.time()
    try:
        yield span
        duration = time.time() - start_time
        span.update(
            metadata={
                **(metadata or {}),
                "duration_seconds": round(duration, 3),
                "status": "success",
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        span.update(
            output={"error": str(e)},
            metadata={
                **(metadata or {}),
                "duration_seconds": round(duration, 3),
                "status": "error",
                "error_type": type(e).__name__,
            },
        )
        raise


def create_traced_llm_func(
    llm_func: Callable[..., Any],
    model_name: str = "unknown",
) -> Callable[..., Any]:
    """
    Wrap an LLM function with Langfuse tracing.

    Args:
        llm_func: The original LLM function
        model_name: Name of the model being used

    Returns:
        Wrapped function with tracing

    Example:
        traced_llm = create_traced_llm_func(gemini_llm_func, "gemini-2.5-flash")
        response = await traced_llm(prompt)
    """
    if not _check_langfuse_enabled():
        return llm_func

    @wraps(llm_func)
    async def traced_async(*args: Any, **kwargs: Any) -> Any:
        langfuse = get_langfuse()
        if langfuse is None:
            return await llm_func(*args, **kwargs)

        # Extract prompt from args or kwargs
        prompt = args[0] if args else kwargs.get("prompt", "")
        prompt_preview = str(prompt)[:500] if prompt else ""

        generation = langfuse.generation(
            name="llm_call",
            model=model_name,
            input={"prompt_preview": prompt_preview, "prompt_length": len(str(prompt))},
            metadata={"full_prompt_length": len(str(prompt))},
        )

        start_time = time.time()
        try:
            result = await llm_func(*args, **kwargs)
            duration = time.time() - start_time

            generation.update(
                output={
                    "response_preview": str(result)[:500],
                    "response_length": len(str(result)),
                },
                metadata={
                    "duration_seconds": round(duration, 3),
                    "status": "success",
                },
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            generation.update(
                output={"error": str(e)},
                metadata={
                    "duration_seconds": round(duration, 3),
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            raise

    @wraps(llm_func)
    def traced_sync(*args: Any, **kwargs: Any) -> Any:
        langfuse = get_langfuse()
        if langfuse is None:
            return llm_func(*args, **kwargs)

        prompt = args[0] if args else kwargs.get("prompt", "")

        generation = langfuse.generation(
            name="llm_call",
            model=model_name,
            input={"prompt_length": len(str(prompt))},
        )

        start_time = time.time()
        try:
            result = llm_func(*args, **kwargs)
            duration = time.time() - start_time

            generation.update(
                output={"response_length": len(str(result))},
                metadata={"duration_seconds": round(duration, 3), "status": "success"},
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            generation.update(
                output={"error": str(e)},
                metadata={"duration_seconds": round(duration, 3), "status": "error"},
            )
            raise

    import asyncio

    if asyncio.iscoroutinefunction(llm_func):
        return traced_async
    return traced_sync


def create_traced_embedding_func(
    embed_func: Callable[..., Any],
    model_name: str = "voyage-4-large",
) -> Callable[..., Any]:
    """
    Wrap an embedding function with Langfuse tracing.

    Args:
        embed_func: The original embedding function
        model_name: Name of the embedding model

    Returns:
        Wrapped function with tracing
    """
    if not _check_langfuse_enabled():
        return embed_func

    @wraps(embed_func)
    async def traced_async(texts: list[str]) -> Any:
        langfuse = get_langfuse()
        if langfuse is None:
            return await embed_func(texts)

        span = langfuse.span(
            name="embedding",
            input={"num_texts": len(texts), "total_chars": sum(len(t) for t in texts)},
            metadata={"model": model_name},
        )

        start_time = time.time()
        try:
            result = await embed_func(texts)
            duration = time.time() - start_time

            # Get embedding dimensions
            embed_dim = 0
            if hasattr(result, "shape"):
                embed_dim = result.shape[-1] if len(result.shape) > 1 else len(result)

            span.update(
                output={"num_embeddings": len(texts), "embedding_dim": embed_dim},
                metadata={
                    "duration_seconds": round(duration, 3),
                    "status": "success",
                    "texts_per_second": round(len(texts) / duration, 2)
                    if duration > 0
                    else 0,
                },
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            span.update(
                output={"error": str(e)},
                metadata={"duration_seconds": round(duration, 3), "status": "error"},
            )
            raise

    return traced_async


def log_rag_query(
    query: str,
    mode: str,
    response: str,
    context: str | None = None,
    duration_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Log a complete RAG query to Langfuse (fire-and-forget).

    Use this for simple logging without decorators.

    Args:
        query: The user query
        mode: Query mode used
        response: The generated response
        context: Retrieved context (optional)
        duration_seconds: Query duration (optional)
        metadata: Additional metadata (optional)

    Example:
        log_rag_query(
            query="What is MongoDB?",
            mode="mix",
            response="MongoDB is a document database...",
            duration_seconds=2.5,
        )
    """
    if not _check_langfuse_enabled():
        return

    langfuse = get_langfuse()
    if langfuse is None:
        return

    langfuse.trace(
        name="rag_query",
        input={"query": query, "mode": mode},
        output={
            "response_length": len(response),
            "context_length": len(context) if context else 0,
        },
        metadata={
            **(metadata or {}),
            "duration_seconds": duration_seconds,
            "status": "success",
        },
    )


def log_ingestion(
    file_name: str,
    num_chunks: int,
    num_entities: int,
    num_relations: int,
    duration_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Log a document ingestion to Langfuse.

    Args:
        file_name: Name of the ingested file
        num_chunks: Number of chunks created
        num_entities: Number of entities extracted
        num_relations: Number of relations extracted
        duration_seconds: Ingestion duration
        metadata: Additional metadata
    """
    if not _check_langfuse_enabled():
        return

    langfuse = get_langfuse()
    if langfuse is None:
        return

    langfuse.trace(
        name="document_ingestion",
        input={"file_name": file_name},
        output={
            "num_chunks": num_chunks,
            "num_entities": num_entities,
            "num_relations": num_relations,
        },
        metadata={
            **(metadata or {}),
            "duration_seconds": duration_seconds,
            "status": "success",
        },
    )


# Export status for easy checking
def is_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    return _check_langfuse_enabled()


def get_status() -> dict[str, Any]:
    """Get Langfuse integration status."""
    return {
        "available": LANGFUSE_AVAILABLE,
        "enabled": _check_langfuse_enabled(),
        "host": os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        "public_key_set": bool(os.environ.get("LANGFUSE_PUBLIC_KEY")),
        "secret_key_set": bool(os.environ.get("LANGFUSE_SECRET_KEY")),
    }

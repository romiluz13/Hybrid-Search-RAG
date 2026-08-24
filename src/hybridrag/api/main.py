"""
FastAPI application for HybridRAG.

Production-ready API with:
- Document ingestion
- Query endpoint
- Health checks
- Proper lifecycle management
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .. import __version__
from ..core.mongodb_client import close_shared_client
from ..core.rag import HybridRAG, create_hybridrag
from ..engine.exceptions import (
    RetrievalCapabilityError,
    RetrievalExecutionError,
    RetrievalValidationError,
)
from ..engine.security import (
    api_key_security_context,
    reset_request_security_context,
    scope_document_metadata,
    set_request_security_context,
)
from ..engine.utils import bson_to_jsonable
from .models import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QueryStreamChunk,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("hybridrag.api")

# Global RAG instance
_rag: HybridRAG | None = None
_rate_limit_state: dict[str, list[float]] = {}


def _retrieval_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, RetrievalValidationError):
        return HTTPException(
            status_code=400,
            detail={
                "code": "retrieval_validation_error",
                "message": str(error),
            },
        )
    if isinstance(error, RetrievalCapabilityError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "retrieval_capability_error",
                "message": str(error),
            },
        )
    return HTTPException(
        status_code=502,
        detail={
            "code": "retrieval_execution_error",
            "message": "Retrieval backend failed",
        },
    )


def _get_api_key() -> str | None:
    return os.environ.get("HYBRIDRAG_API_KEY")


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_rate_limit_per_window() -> int:
    return _get_env_int("HYBRIDRAG_RATE_LIMIT_PER_WINDOW", 0)


def _get_rate_limit_window_seconds() -> int:
    return _get_env_int("HYBRIDRAG_RATE_LIMIT_WINDOW_SECONDS", 60)


async def guard_request(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> AsyncGenerator[None, None]:
    """Optional auth and rate-limit seam for the public reference API."""
    required_api_key = _get_api_key()
    if required_api_key and x_api_key != required_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key",
        )

    rate_limit = _get_rate_limit_per_window()
    if rate_limit > 0:
        window_seconds = _get_rate_limit_window_seconds()
        client_host = request.client.host if request.client else "unknown"
        now = time.time()
        recent_requests = _rate_limit_state.setdefault(client_host, [])
        cutoff = now - window_seconds
        recent_requests[:] = [stamp for stamp in recent_requests if stamp >= cutoff]

        if len(recent_requests) >= rate_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for this API instance",
            )

        recent_requests.append(now)

    try:
        security_context = api_key_security_context(x_api_key)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from None

    context_token = set_request_security_context(security_context)
    try:
        yield
    finally:
        reset_request_security_context(context_token)


async def guard_operator_request(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Require a configured API key for detailed operational diagnostics."""
    required_api_key = os.environ.get("HYBRIDRAG_OPERATOR_API_KEY")
    if not required_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Operational diagnostics require HYBRIDRAG_OPERATOR_API_KEY",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, required_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing operator X-API-Key",
        )


def get_rag() -> HybridRAG:
    """Get the initialized RAG instance."""
    if _rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized",
        )
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    The RAG system is initialized at startup when possible. If MongoDB or API
    keys are not configured yet, the app still boots in degraded mode so /health
    and /docs remain reachable with an actionable message.
    """
    global _rag

    # Startup: Initialize HybridRAG (degrade gracefully if not configured)
    try:
        _rag = await create_hybridrag(auto_initialize=True)
    except Exception as exc:
        logger.warning(
            "RAG system not initialized at startup: %s. "
            "Configure MONGODB_URI and API keys in .env, then restart. "
            "API is running in degraded mode (/health reports 'degraded').",
            exc,
        )
        _rag = None

    yield

    # Shutdown: Cleanup
    _rag = None
    close_shared_client()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="HybridRAG API",
        description="State-of-the-art RAG system with MongoDB + Voyage AI",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware - configure allowed origins from environment
    # Default to localhost for development; set CORS_ORIGINS in production
    cors_origins = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
    )
    allowed_origins = [
        origin.strip() for origin in cors_origins.split(",") if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-API-Key"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
    )
    async def health_check() -> HealthResponse:
        """Check system health."""
        components: dict[str, str] = {"api": "healthy"}

        try:
            rag = get_rag()
            status = await rag.get_status()
            components["rag"] = "healthy" if status["initialized"] else "unhealthy"
            components["mongodb"] = "healthy"
        except Exception:
            components["rag"] = "unhealthy"
            components["mongodb"] = "unknown"

        overall = (
            "healthy"
            if all(v == "healthy" for v in components.values())
            else "degraded"
        )

        return HealthResponse(
            status=overall,
            components=components,
            version=__version__,
        )

    @app.get("/ready", tags=["health"])
    async def readiness_check() -> dict:
        """Kubernetes readiness probe."""
        try:
            get_rag()
            return {"ready": True}
        except HTTPException:
            return {"ready": False}

    @app.post(
        "/v1/ingest",
        response_model=IngestResponse,
        dependencies=[Depends(guard_request)],
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["documents"],
    )
    async def ingest_documents(request: IngestRequest) -> IngestResponse:
        """
        Ingest documents into the RAG system.

        Documents are:
        1. Chunked using token-based chunking
        2. Embedded using Voyage AI
        3. Entity/relationship extracted using LLM
        4. Stored in MongoDB (vector + graph)
        """
        rag = get_rag()

        # Validate IDs if provided
        if request.ids and len(request.ids) != len(request.documents):
            raise HTTPException(
                status_code=400,
                detail="IDs list must match documents list length",
            )

        try:
            await rag.insert(
                documents=request.documents,
                ids=request.ids,
                metadata=scope_document_metadata(
                    request.metadata,
                    len(request.documents),
                ),
            )

            return IngestResponse(
                status="success",
                documents_processed=len(request.documents),
                message=f"Successfully ingested {len(request.documents)} documents",
            )
        except Exception as e:
            # M34: Do not leak internal exception details to API callers
            logger.error(f"Ingestion error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during ingestion",
            ) from None

    @app.post(
        "/v1/query",
        response_model=QueryResponse,
        dependencies=[Depends(guard_request)],
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["query"],
    )
    async def query(request: QueryRequest) -> QueryResponse:
        """
        Query the RAG system.

        Query modes:
        - **local**: Entity-focused retrieval via graph neighbors
        - **global**: Community-level summaries
        - **hybrid**: Combines local + global
        - **naive**: Direct vector search without graph
        - **mix**: All modes combined (recommended)
        - **bypass**: Skip retrieval, direct LLM
        """
        rag = get_rag()

        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Use /v1/query/stream when stream=true",
            )

        try:
            if request.include_context or request.include_references:
                result = await rag.query_with_sources(
                    query=request.query,
                    mode=request.mode,
                    top_k=request.top_k,
                    rerank_top_k=request.rerank_top_k,
                    enable_rerank=request.enable_rerank,
                    filter_config=request.filter_config,
                    fusion_strategy=request.fusion_strategy,
                    vector_search_mode=request.vector_search_mode,
                    rerank_strategy=request.rerank_strategy,
                    native_rerank_model=request.native_rerank_model,
                )
                return QueryResponse(
                    answer=result["answer"],
                    context=result["context"] if request.include_context else None,
                    references=(
                        result.get("references", [])
                        if request.include_references
                        else []
                    ),
                    metadata={
                        "mode": result["mode"],
                        "top_k": request.top_k,
                        "rerank_top_k": request.rerank_top_k,
                        **result.get("metadata", {}),
                    },
                )
            else:
                retrieval_options: dict[str, Any] = {}
                if request.filter_config is not None:
                    retrieval_options["filter_config"] = request.filter_config
                if request.fusion_strategy is not None:
                    retrieval_options["fusion_strategy"] = request.fusion_strategy
                retrieval_options["vector_search_mode"] = request.vector_search_mode
                retrieval_options["rerank_strategy"] = request.rerank_strategy
                retrieval_options["native_rerank_model"] = request.native_rerank_model
                answer = await rag.query(
                    query=request.query,
                    mode=request.mode,
                    top_k=request.top_k,
                    rerank_top_k=request.rerank_top_k,
                    enable_rerank=request.enable_rerank,
                    **retrieval_options,
                )
                if not isinstance(answer, str):
                    raise ValueError(
                        "Non-streaming endpoint received a streaming response"
                    )
                return QueryResponse(
                    answer=answer,
                    references=[],
                    metadata={
                        "mode": request.mode,
                        "top_k": request.top_k,
                        "rerank_top_k": request.rerank_top_k,
                    },
                )
        except (RetrievalCapabilityError, RetrievalExecutionError) as e:
            raise _retrieval_http_exception(e) from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        except Exception as e:
            # M34: Do not leak internal exception details to API callers
            logger.error(f"Query error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during query",
            ) from None

    @app.post(
        "/v1/query/stream",
        dependencies=[Depends(guard_request)],
        tags=["query"],
        responses={
            200: {
                "description": "NDJSON stream of metadata followed by answer chunks",
                "content": {
                    "application/x-ndjson": {
                        "schema": QueryStreamChunk.model_json_schema()
                    }
                },
            },
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def stream_query(request: QueryRequest) -> StreamingResponse:
        """Stream a query response as NDJSON with metadata first."""
        rag = get_rag()

        try:
            result = await rag.stream_query(
                query=request.query,
                mode=request.mode,
                top_k=request.top_k,
                rerank_top_k=request.rerank_top_k,
                enable_rerank=request.enable_rerank,
                include_context=request.include_context,
                include_references=request.include_references,
                filter_config=request.filter_config,
                fusion_strategy=request.fusion_strategy,
                vector_search_mode=request.vector_search_mode,
                rerank_strategy=request.rerank_strategy,
                native_rerank_model=request.native_rerank_model,
            )
        except (RetrievalCapabilityError, RetrievalExecutionError) as e:
            raise _retrieval_http_exception(e) from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        except Exception as e:
            logger.error(f"Streaming query error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during streaming query",
            ) from None

        async def stream_generator():
            envelope: dict[str, Any] = {
                "metadata": {
                    "mode": result["mode"],
                    "top_k": request.top_k,
                    "rerank_top_k": request.rerank_top_k,
                    **result.get("metadata", {}),
                }
            }
            if request.include_context:
                envelope["context"] = result.get("context")
            if request.include_references:
                envelope["references"] = result.get("references", [])
            yield json.dumps(bson_to_jsonable(envelope)) + "\n"

            try:
                async for chunk in result["response_iterator"]:
                    if chunk:
                        yield json.dumps({"answer": chunk}) + "\n"
            except Exception as e:
                logger.error(f"Streaming response failure: {e}", exc_info=True)
                yield json.dumps({"error": "Streaming response interrupted"}) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/v1/query/explain",
        dependencies=[Depends(guard_operator_request)],
        tags=["query"],
    )
    async def explain_query(request: QueryRequest) -> dict[str, Any]:
        """Execute a redacted MongoDB explanation for the effective pipeline."""
        rag = get_rag()
        options: dict[str, Any] = {}
        if request.filter_config is not None:
            options["filter_config"] = request.filter_config
        if request.fusion_strategy is not None:
            options["fusion_strategy"] = request.fusion_strategy
        options["vector_search_mode"] = request.vector_search_mode
        options["rerank_strategy"] = request.rerank_strategy
        options["native_rerank_model"] = request.native_rerank_model
        options["enable_rerank"] = request.enable_rerank
        try:
            return bson_to_jsonable(
                await rag.explain_query(
                    request.query,
                    mode=request.mode,
                    top_k=request.top_k,
                    **options,
                )
            )
        except (RetrievalCapabilityError, RetrievalExecutionError) as exc:
            raise _retrieval_http_exception(exc) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except Exception as exc:
            logger.error(f"Query explain error: {exc}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during query explanation",
            ) from None

    @app.get(
        "/v1/search-indexes",
        dependencies=[Depends(guard_operator_request)],
        tags=["diagnostics"],
    )
    async def list_search_indexes() -> list[dict[str, Any]]:
        """Return stable search-index readiness records."""
        try:
            return bson_to_jsonable(await get_rag().list_search_indexes())
        except Exception as exc:
            logger.error(f"Search index status error: {exc}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error while reading search indexes",
            ) from None

    @app.get(
        "/v1/search-indexes/sync",
        dependencies=[Depends(guard_operator_request)],
        tags=["diagnostics"],
    )
    async def verify_index_sync(
        timeout_seconds: float = 60,
        poll_interval_seconds: float = 2,
    ) -> dict[str, Any]:
        """Functional probe: confirm seeded documents are queryable.

        Fires minimal ``$vectorSearch`` and ``$search`` probes to verify
        that Atlas Search indexes have ingested recently-seeded documents.
        ``queryable=True`` alone is insufficient because Atlas Search is
        eventually consistent.
        """
        # Clamp timeout to prevent unbounded long-running HTTP requests
        max_api_timeout = 300.0  # 5 minutes
        timeout_seconds = min(timeout_seconds, max_api_timeout)
        try:
            result = await get_rag().verify_index_sync(
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            return bson_to_jsonable(result)
        except Exception as exc:
            logger.error(f"Index sync probe error: {exc}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during index sync probe",
            ) from None

    @app.delete(
        "/v1/documents/{doc_id}",
        dependencies=[Depends(guard_request)],
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
        tags=["documents"],
    )
    async def delete_document(doc_id: str) -> dict:
        """Delete a document from the RAG system."""
        # M33: Validate doc_id as ObjectId format
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            ObjectId(doc_id)
        except (InvalidId, Exception):
            raise HTTPException(
                status_code=400, detail="Invalid document ID format"
            ) from None

        rag = get_rag()

        try:
            await rag.delete_document(doc_id)
            return {"status": "deleted", "doc_id": doc_id}
        except PermissionError:
            raise HTTPException(status_code=404, detail="Document not found") from None
        except Exception as e:
            # M34: Do not leak internal exception details to API callers
            logger.error(f"Delete error for doc_id={doc_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during document deletion",
            ) from None

    @app.get("/v1/status", tags=["system"])
    async def get_status() -> dict:
        """Get system status and configuration."""
        rag = get_rag()
        return await rag.get_status()


# Create default app instance
app = create_app()

"""
FastAPI application for HybridRAG.

Production-ready API with:
- Document ingestion
- Query endpoint
- Health checks
- Proper lifecycle management
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.mongodb_client import close_shared_client
from ..core.rag import HybridRAG, create_hybridrag
from .models import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("hybridrag.api")

__version__ = "0.2.0"

# Global RAG instance
_rag: HybridRAG | None = None


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
    """Application lifespan manager."""
    global _rag

    # Startup: Initialize HybridRAG
    _rag = await create_hybridrag(auto_initialize=True)

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
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
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

        try:
            if request.include_context or request.include_references:
                result = await rag.query_with_sources(
                    query=request.query,
                    mode=request.mode,
                    top_k=request.top_k,
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
                answer = await rag.query(
                    query=request.query,
                    mode=request.mode,
                    top_k=request.top_k,
                    rerank_top_k=request.rerank_top_k,
                    enable_rerank=request.enable_rerank,
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
        except Exception as e:
            # M34: Do not leak internal exception details to API callers
            logger.error(f"Query error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during query",
            ) from None

    @app.delete(
        "/v1/documents/{doc_id}",
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

"""
Pydantic models for API request/response.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from hybridrag.enhancements.filters import FilterConfig
from hybridrag.enhancements.filters.metadata import normalize_metadata_batch


class IngestRequest(BaseModel):
    """Document ingestion request."""

    documents: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of document contents to ingest (max 100)",
    )
    ids: list[str] | None = Field(
        default=None,
        description="Optional document IDs (must match documents length)",
    )
    metadata: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional metadata object for each document",
    )

    @model_validator(mode="after")
    def validate_cardinality(self) -> IngestRequest:
        self.metadata = normalize_metadata_batch(self.metadata, len(self.documents))
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": ["Document content 1...", "Document content 2..."],
                "metadata": [{"category": "docs"}, {"category": "guides"}],
            }
        }
    }


class IngestResponse(BaseModel):
    """Document ingestion response."""

    status: Literal["success", "partial", "failed"]
    documents_processed: int
    message: str | None = None


class RetrievalRequestOptions(BaseModel):
    """Retrieval options shared by both HTTP query surfaces."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )
    filter_config: FilterConfig | None = Field(
        default=None,
        description="Backend-neutral metadata filter for naive retrieval",
    )
    fusion_strategy: Literal["rank", "score"] | None = Field(
        default=None,
        description="Optional MongoDB hybrid fusion strategy",
    )
    vector_search_mode: Literal["ann", "exact"] = Field(
        default="ann",
        description="Approximate or exact MongoDB vector execution",
    )
    rerank_strategy: Literal["native", "external"] = Field(
        default="native",
        description="MongoDB native or configured external reranking",
    )
    native_rerank_model: Literal[
        "rerank-2.5", "rerank-2.5-lite", "rerank-2", "rerank-2-lite"
    ] = Field(
        default="rerank-2.5",
        description="Voyage model for MongoDB native reranking",
    )

    @model_validator(mode="after")
    def validate_retrieval_options(self) -> RetrievalRequestOptions:
        if self.filter_config is not None and self.mode != "naive":
            raise ValueError("filter_config is supported only with mode='naive' in v1")
        if self.fusion_strategy is not None and self.mode not in {"naive", "mix"}:
            raise ValueError(
                "fusion_strategy is supported only with mode='naive' or mode='mix'"
            )
        return self


class QueryRequest(RetrievalRequestOptions):
    """Query request."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Search query",
    )
    top_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="Number of results to retrieve",
    )
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results after reranking",
    )
    enable_rerank: bool = Field(
        default=True,
        description="Enable reranking",
    )
    include_context: bool = Field(
        default=False,
        description="Include source context in response",
    )
    include_references: bool = Field(
        default=False,
        description="Include structured source references in response",
    )
    stream: bool = Field(
        default=False,
        description="Stream the answer as NDJSON chunks instead of returning one JSON payload",
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is MongoDB vector search?",
                "mode": "mix",
                "top_k": 60,
                "enable_rerank": True,
            }
        }
    }


class QueryResponse(BaseModel):
    """Query response."""

    answer: str
    context: str | None = None
    references: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryStreamChunk(BaseModel):
    """NDJSON chunk emitted by the streaming query endpoint."""

    answer: str | None = None
    context: str | None = None
    references: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    components: dict[str, str]
    version: str


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str | dict[str, str]

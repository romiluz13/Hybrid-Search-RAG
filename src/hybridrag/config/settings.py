"""
Configuration settings for HybridRAG.

Uses pydantic-settings for type-safe configuration from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """HybridRAG configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB Atlas — defaults to the local atlas-local:preview stack so the
    # API/CLI/demo boot out-of-the-box. Set MONGODB_URI in .env for Atlas/cloud.
    mongodb_uri: SecretStr = Field(
        default=SecretStr("mongodb://localhost:27018/?directConnection=true"),
        description="MongoDB connection URI. Defaults to local atlas-local:preview; set MONGODB_URI in .env for Atlas/cloud.",
    )

    # [M2] Validate URI starts with mongodb:// or mongodb+srv://
    @field_validator("mongodb_uri")
    @classmethod
    def validate_mongodb_uri(cls, v: SecretStr) -> SecretStr:
        """Ensure MongoDB URI has a valid scheme."""
        uri = v.get_secret_value()
        if not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
            raise ValueError("MongoDB URI must start with mongodb:// or mongodb+srv://")
        return v

    mongodb_database: str = Field(
        default="hybridrag",
        description="MongoDB database name",
    )
    mongodb_workspace: str = Field(
        default="default",
        description="Workspace prefix for MongoDB collection names in the engine layer. "
        "Used by engine/kg/mongo_impl.py to namespace collections (e.g., 'myworkspace_kg_edges'). "
        "Set to empty string for no prefix.",
    )

    # MongoDB Connection Pool [Rule: consistency-read-concern-levels]
    mongodb_max_pool_size: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum connection pool size per client",
    )
    mongodb_min_pool_size: int = Field(
        default=0,
        ge=0,
        description="Minimum connection pool size (0 = on-demand)",
    )
    mongodb_max_idle_time_ms: int = Field(
        default=60000,
        ge=0,
        description="Maximum idle time for connections in ms",
    )

    # MongoDB TLS [M4] - Expose TLS settings for secure connections
    mongodb_tls: bool = Field(
        default=False,
        description="Enable TLS for MongoDB connection. "
        "Atlas connections (mongodb+srv://) use TLS by default via URI.",
    )
    mongodb_tls_allow_invalid_certificates: bool = Field(
        default=False,
        description="Allow invalid TLS certificates (dev only). "
        "NEVER enable in production.",
    )

    # MongoDB Timeouts [Rule: mongodb-connection] - Fail fast on connection issues
    mongodb_server_selection_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        description="Quick failover for replica set topology changes (5s default per skill)",
    )
    mongodb_connect_timeout_ms: int = Field(
        default=10000,
        ge=1000,
        description="Fail fast on connection issues (10s default per skill)",
    )
    mongodb_socket_timeout_ms: int = Field(
        default=0,
        ge=0,
        description="Socket timeout in ms. 0=no timeout for long-running operations. "
        "Set 30000 for OLTP workloads to prevent hanging queries.",
    )

    # MongoDB Read/Write Concerns [Rule: fundamental-commit-write-concern]
    # [M1] Defaults changed to "majority" for production durability.
    # RAG knowledge base: data loss = costly re-ingestion. Durability > latency.
    mongodb_read_concern: Literal["local", "majority", "snapshot"] = Field(
        default="majority",
        description="Read concern level. 'majority' for production durability. "
        "'local' for lower latency with eventual consistency.",
    )
    mongodb_write_concern: Literal["0", "1", "majority"] = Field(
        default="majority",
        description="Write concern level. 'majority' for production durability. "
        "'1' for lower latency with less durability guarantee.",
    )

    # Query Validation
    max_query_length: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum query length in characters to prevent abuse",
    )

    # Aggregation Timeout [Rule: ops-transaction-runtime-limit]
    mongodb_aggregate_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Maximum time for aggregation pipelines in ms",
    )
    filterable_metadata_fields: dict[
        str, Literal["token", "number", "date", "boolean", "objectId", "uuid"]
    ] = Field(
        default_factory=lambda: {
            "metadata.category": "token",
            "metadata.year": "number",
        },
        description="Public metadata filter paths and Atlas Search mapping types",
    )
    vector_embedding_backend: Literal["client", "automated"] = Field(
        default="automated",
        description="Client-generated vectors or MongoDB Automated Embedding",
    )
    automated_embedding_model: str = Field(
        default="voyage-4-large",
        description="MongoDB Automated Embedding model",
    )

    @field_validator("filterable_metadata_fields")
    @classmethod
    def validate_filterable_metadata_fields(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        for path in value:
            if (
                not path.startswith("metadata.")
                or path.count(".") != 1
                or "$" in path
                or "\x00" in path
            ):
                raise ValueError("Filterable metadata paths must use metadata.<field>")
        return value

    # Voyage AI (for embeddings and reranking)
    voyage_api_key: SecretStr | None = Field(
        default=None,
        description="Voyage AI API key. Use a `pa-...` key for Voyage AI direct, "
        "or an `al-...` key for the MongoDB-hosted endpoint (set voyage_base_url).",
    )
    voyage_base_url: str | None = Field(
        default=None,
        description="Optional base URL for the Voyage client. Set to the MongoDB-hosted "
        "endpoint (e.g. https://ai.mongodb.com/v1) to use an `al-...` Atlas key "
        "instead of a Voyage AI direct `pa-...` key. Defaults to Voyage AI direct.",
    )
    voyage_embedding_model: str = Field(
        default="voyage-4-large",
        description="Voyage embedding model (voyage-4-large, voyage-4, voyage-4-lite, voyage-code-3)",
    )
    voyage_context_model: str = Field(
        default="voyage-context-3",
        description="Voyage contextualized embedding model",
    )
    voyage_rerank_model: str = Field(
        default="rerank-2.5",
        description="Voyage reranking model",
    )
    voyage_rerank_instructions: str | None = Field(
        default=None,
        description="Optional custom default instructions for Voyage reranker. "
        "Applied to all queries unless overridden. Examples: "
        "'Prioritize recent sources' or 'Focus on technical documentation'. "
        "If None, uses intelligent mode-aware defaults when enabled.",
    )
    enable_smart_rerank_instructions: bool = Field(
        default=True,
        description="Enable intelligent mode-aware default instructions. "
        "When True, automatically generates context-appropriate instructions "
        "based on query mode (mix, local, global, etc.). "
        "When False, no default instructions are applied unless "
        "voyage_rerank_instructions is set.",
    )

    # Tavily AI (for web content extraction)
    tavily_api_key: SecretStr | None = Field(
        default=None,
        description="Tavily API key for web content extraction. "
        "Optional - enables ingest_url() and ingest_website() functionality. "
        "Get your key at https://tavily.com",
    )

    # LLM Provider Selection
    llm_provider: Literal["anthropic", "openai", "gemini", "grove"] = Field(
        default="anthropic",
        description="LLM provider to use (anthropic, openai, gemini, grove). "
        "grove = MongoDB internal OpenAI-compatible gateway.",
    )
    enable_llm: bool = Field(
        default=True,
        description="Enable LLM generation. Set to False for retrieval-only workflows "
        "that still need embeddings, ingestion, and context retrieval.",
    )

    # Anthropic (Claude)
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (required if llm_provider=anthropic)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model for generation",
    )

    # OpenAI
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (required if llm_provider=openai)",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Custom OpenAI-compatible API endpoint (for Azure/gateway proxies)",
    )
    openai_extra_headers: str | None = Field(
        default=None,
        description='Extra headers as JSON string, e.g. \'{"api-key": "..."}\' for Azure gateways',
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model",
    )

    # Google Gemini
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google AI API key (required if llm_provider=gemini)",
    )

    # Grove — MongoDB internal OpenAI-compatible LLM gateway.
    # Reuses the OpenAI client with a custom base_url + api key.
    grove_api_key: SecretStr | None = Field(
        default=None,
        description="Grove API key (required if llm_provider=grove). MongoDB internal.",
    )
    grove_base_url: str | None = Field(
        default=None,
        description="Grove gateway base URL, e.g. https://grove.example.mongodb.com/v1",
    )
    grove_model: str = Field(
        default="gpt-4o",
        description="Model name accepted by the Grove gateway",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model (gemini-2.5-flash, gemini-2.0-flash)",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model",
    )

    # Embedding Provider - VOYAGE ONLY (best quality)
    # Note: We only support Voyage AI for embeddings - no fallback to OpenAI/Gemini
    embedding_provider: Literal["voyage"] = Field(
        default="voyage",
        description="Embedding provider (Voyage AI only - best quality)",
    )

    # Query settings
    default_query_mode: Literal[
        "local", "global", "hybrid", "naive", "mix", "bypass"
    ] = Field(
        default="mix",
        description="Default query mode",
    )
    default_top_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Default number of results to retrieve",
    )
    default_rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of results after reranking",
    )
    enable_rerank: bool = Field(
        default=True,
        description="Enable reranking by default",
    )

    # Enhancement settings
    enable_implicit_expansion: bool = Field(
        default=True,
        description="Enable implicit entity expansion",
    )
    implicit_expansion_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for implicit expansion",
    )
    implicit_expansion_max: int = Field(
        default=10,
        ge=1,
        description="Maximum entities from implicit expansion",
    )
    enable_entity_boosting: bool = Field(
        default=True,
        description="Enable entity boosting in reranking",
    )
    entity_boost_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for entity overlap boost",
    )

    # Embedding settings
    embedding_dim: int = Field(
        default=1024,
        description="Embedding dimension (1024 default for voyage-4-large)",
    )
    max_token_size: int = Field(
        default=4096,
        description="Maximum tokens for embedding",
    )
    embedding_batch_size: int = Field(
        default=128,
        ge=1,
        le=128,
        description="Batch size for embedding API calls",
    )

    # Context limits
    max_token_for_text_unit: int = Field(
        default=4000,
        description="Maximum tokens per text unit",
    )
    max_token_for_local_context: int = Field(
        default=4000,
        description="Maximum tokens for local context",
    )
    max_token_for_global_context: int = Field(
        default=4000,
        description="Maximum tokens for global context",
    )

    # Observability (optional)
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public key",
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        description="Langfuse secret key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL",
    )


@lru_cache
def _get_settings_cached() -> Settings:
    """Internal cached settings factory."""
    return Settings()


def get_settings() -> Settings:
    """Get cached settings instance."""
    return _get_settings_cached()


def clear_settings_cache() -> None:
    """Clear the settings cache. Used in tests for isolation.

    [M3] Allows tests to reset cached settings between test cases.
    """
    _get_settings_cached.cache_clear()

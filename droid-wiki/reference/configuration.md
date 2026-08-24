# Configuration

`Settings` in `src/hybridrag/config/settings.py` loads environment variables and `.env` through pydantic-settings. Field names map to case-insensitive uppercase environment names, for example `mongodb_uri` becomes `MONGODB_URI`.

Secret fields use Pydantic `SecretStr`. Defaults are suitable for local development, not a statement that every provider-backed operation works without credentials.

## MongoDB

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `MONGODB_URI` | `SecretStr` | `mongodb://localhost:27018/?directConnection=true` | MongoDB connection URI; accepts only `mongodb://` or `mongodb+srv://` |
| `MONGODB_DATABASE` | `str` | `hybridrag` | Database name |
| `MONGODB_WORKSPACE` | `str` | `default` | Prefix for engine collection names; empty disables the prefix |
| `MONGODB_MAX_POOL_SIZE` | `int` | `100` | Maximum client pool size, range 1–500 |
| `MONGODB_MIN_POOL_SIZE` | `int` | `0` | Minimum client pool size; zero is on demand |
| `MONGODB_MAX_IDLE_TIME_MS` | `int` | `60000` | Maximum connection idle time in milliseconds |
| `MONGODB_TLS` | `bool` | `false` | Enable TLS; SRV Atlas URIs already imply TLS |
| `MONGODB_TLS_ALLOW_INVALID_CERTIFICATES` | `bool` | `false` | Allow invalid certificates for development only |
| `MONGODB_SERVER_SELECTION_TIMEOUT_MS` | `int` | `5000` | Server-selection timeout, minimum 1000 ms |
| `MONGODB_CONNECT_TIMEOUT_MS` | `int` | `10000` | Connection timeout, minimum 1000 ms |
| `MONGODB_SOCKET_TIMEOUT_MS` | `int` | `0` | Socket timeout; zero means no timeout |
| `MONGODB_READ_CONCERN` | `local \| majority \| snapshot` | `majority` | Read durability and consistency level |
| `MONGODB_WRITE_CONCERN` | `0 \| 1 \| majority` | `majority` | Write acknowledgement level |
| `MONGODB_AGGREGATE_TIMEOUT_MS` | `int` | `30000` | Aggregation timeout, range 1,000–300,000 ms |

`MONGODB_WORKSPACE` is used by `src/hybridrag/engine/kg/mongo_impl.py`; the default therefore produces collections such as `default_chunks`.

## Query validation and search schema

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `MAX_QUERY_LENGTH` | `int` | `10000` | Maximum query length, range 100–100,000 characters |
| `FILTERABLE_METADATA_FIELDS` | `dict[str, type]` | `{"metadata.category":"token","metadata.year":"number"}` | Public metadata paths and Atlas mapping types |
| `VECTOR_EMBEDDING_BACKEND` | `client \| automated` | `client` | Generate vectors in the client or use MongoDB Automated Embedding |
| `AUTOMATED_EMBEDDING_MODEL` | `str` | `voyage-4-large` | Model used by Automated Embedding |

Filterable paths must be exactly `metadata.<field>`, cannot contain `$` or a null byte, and may use `token`, `number`, `date`, `boolean`, `objectId`, or `uuid`.

## Voyage AI and web extraction

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `VOYAGE_API_KEY` | `SecretStr \| None` | `None` | Direct `pa-...` or MongoDB-hosted `al-...` key |
| `VOYAGE_BASE_URL` | `str \| None` | `None` | Optional Voyage-compatible endpoint; unset uses Voyage direct |
| `VOYAGE_EMBEDDING_MODEL` | `str` | `voyage-4-large` | Embedding model |
| `VOYAGE_CONTEXT_MODEL` | `str` | `voyage-context-3` | Contextualized embedding model |
| `VOYAGE_RERANK_MODEL` | `str` | `rerank-2.5` | Reranking model |
| `VOYAGE_RERANK_INSTRUCTIONS` | `str \| None` | `None` | Custom default reranker instruction |
| `ENABLE_SMART_RERANK_INSTRUCTIONS` | `bool` | `true` | Generate query-mode-aware reranking instructions |
| `TAVILY_API_KEY` | `SecretStr \| None` | `None` | Enables URL and website ingestion through Tavily |

When using an `al-...` key, set `VOYAGE_BASE_URL` to the MongoDB-hosted endpoint. Keep the configured embedding dimensions aligned with the selected model and Atlas index.

## LLM providers

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `LLM_PROVIDER` | `anthropic \| openai \| gemini \| grove` | `anthropic` | Generation provider |
| `ENABLE_LLM` | `bool` | `true` | Disable for retrieval-only ingestion and context workflows |
| `ANTHROPIC_API_KEY` | `SecretStr \| None` | `None` | Required when provider is Anthropic |
| `ANTHROPIC_MODEL` | `str` | `claude-sonnet-4-20250514` | Claude generation model |
| `OPENAI_API_KEY` | `SecretStr \| None` | `None` | Required when provider is OpenAI |
| `OPENAI_MODEL` | `str` | `gpt-4o` | OpenAI or compatible generation model |
| `OPENAI_BASE_URL` | `str \| None` | `None` | Custom OpenAI-compatible endpoint |
| `OPENAI_EXTRA_HEADERS` | `str \| None` | `None` | JSON object encoded as a string for gateway headers |
| `OPENAI_EMBEDDING_MODEL` | `str` | `text-embedding-3-large` | OpenAI embedding model setting |
| `GEMINI_API_KEY` | `SecretStr \| None` | `None` | Required when provider is Gemini |
| `GEMINI_MODEL` | `str` | `gemini-2.5-flash` | Gemini generation model |
| `GEMINI_EMBEDDING_MODEL` | `str` | `text-embedding-004` | Gemini embedding model setting |
| `GROVE_API_KEY` | `SecretStr \| None` | `None` | Key for the internal OpenAI-compatible Grove gateway |
| `GROVE_BASE_URL` | `str \| None` | `None` | Grove gateway base URL |
| `GROVE_MODEL` | `str` | `gpt-4o` | Model name accepted by Grove |
| `EMBEDDING_PROVIDER` | `voyage` | `voyage` | Supported embedding provider; currently fixed to Voyage |

The OpenAI and Gemini embedding model fields remain in the typed settings, but `EMBEDDING_PROVIDER` accepts only `voyage`.

## Query defaults

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `DEFAULT_QUERY_MODE` | `local \| global \| hybrid \| naive \| mix \| bypass` | `mix` | Default retrieval mode |
| `DEFAULT_TOP_K` | `int` | `60` | Initial result count, range 1–200 |
| `DEFAULT_RERANK_TOP_K` | `int` | `10` | Result count after reranking, range 1–50 |
| `ENABLE_RERANK` | `bool` | `true` | Enable reranking by default |

Callers may override these values per query through the Python or REST API.

## Retrieval enhancements

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ENABLE_IMPLICIT_EXPANSION` | `bool` | `true` | Expand a query with implicitly related entities |
| `IMPLICIT_EXPANSION_THRESHOLD` | `float` | `0.75` | Similarity threshold, range 0–1 |
| `IMPLICIT_EXPANSION_MAX` | `int` | `10` | Maximum implicitly expanded entities |
| `ENABLE_ENTITY_BOOSTING` | `bool` | `true` | Boost results with entity overlap |
| `ENTITY_BOOST_WEIGHT` | `float` | `0.2` | Entity-overlap weight, range 0–1 |

## Embedding and context limits

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `EMBEDDING_DIM` | `int` | `1024` | Vector dimensions; default matches `voyage-4-large` |
| `MAX_TOKEN_SIZE` | `int` | `4096` | Maximum tokens sent for embedding |
| `EMBEDDING_BATCH_SIZE` | `int` | `128` | Embedding API batch size, range 1–128 |
| `MAX_TOKEN_FOR_TEXT_UNIT` | `int` | `4000` | Per-text-unit token budget |
| `MAX_TOKEN_FOR_LOCAL_CONTEXT` | `int` | `4000` | Local retrieval context budget |
| `MAX_TOKEN_FOR_GLOBAL_CONTEXT` | `int` | `4000` | Global retrieval context budget |

## Observability

| Environment variable | Type | Default | Description |
| --- | --- | --- | --- |
| `LANGFUSE_PUBLIC_KEY` | `str \| None` | `None` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | `SecretStr \| None` | `None` | Langfuse secret key |
| `LANGFUSE_HOST` | `str` | `https://cloud.langfuse.com` | Langfuse endpoint |

## HTTP-only environment controls

The following variables are read directly by the API and security modules rather than `Settings`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HYBRIDRAG_API_KEY` | unset | Protect reference API ingestion, query, and deletion |
| `HYBRIDRAG_OPERATOR_API_KEY` | unset | Protect explain and index diagnostic routes |
| `HYBRIDRAG_RATE_LIMIT_PER_WINDOW` | `0` | Requests per client window; zero disables |
| `HYBRIDRAG_RATE_LIMIT_WINDOW_SECONDS` | `60` | In-memory rate-limit window |
| `CORS_ORIGINS` | local ports 3000 and 8000 | Comma-separated allowed browser origins |
| `HYBRIDRAG_TENANT_FIELD` | unset | Authoritative metadata path for tenant ownership |
| `HYBRIDRAG_API_KEY_TENANTS` | `{}` | JSON mapping from API keys to tenant IDs |
| `HYBRIDRAG_TENANT_CLAIM` | `username` | JWT principal claim used as tenant ID |

Settings are cached by `get_settings()`. Tests that mutate environment values should call `clear_settings_cache()` before constructing a new settings object.

See [Security](../security.md) for tenant behavior and [Deployment](../deployment.md) for a production environment example.

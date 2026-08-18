from __future__ import annotations

from datetime import UTC, datetime
from math import inf, nan
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from bson import ObjectId
from bson.binary import Binary, UuidRepresentation
from pymongo.errors import OperationFailure, PyMongoError

from hybridrag import FilterConfig as ExportedFilterConfig
from hybridrag import FilterPredicate as ExportedFilterPredicate
from hybridrag.core.rag import HybridRAG, QueryParam
from hybridrag.engine import base_engine as base_engine_module
from hybridrag.engine.api.routers import query_routes as engine_query_routes
from hybridrag.engine.api.routers.document_routes import pipeline_index_texts
from hybridrag.engine.api.routers.query_routes import (
    QueryRequest as EngineAPIQueryRequest,
)
from hybridrag.engine.base import QueryParam as EngineQueryParam
from hybridrag.engine.base_engine import BaseRAGEngine
from hybridrag.engine.exceptions import (
    RetrievalCapabilityError,
    RetrievalExecutionError,
)
from hybridrag.engine.kg.mongo_impl import MongoVectorDBStorage
from hybridrag.engine.operate import (
    _perform_kg_search,
    _retrieval_cache_identity,
    naive_query,
)
from hybridrag.engine.utils import apply_rerank_if_enabled
from hybridrag.enhancements.filters import (
    FilterConfig,
    FilterPredicate,
    RetrievalSecurityContext,
    compile_filter_to_atlas,
    compile_filter_to_mql,
    compile_retrieval_filter_to_atlas,
    compile_retrieval_filter_to_mql,
)


@pytest.mark.asyncio
async def test_insert_forwards_validated_document_metadata() -> None:
    engine = SimpleNamespace(ainsert=AsyncMock())
    rag = HybridRAG(settings=cast(Any, SimpleNamespace()))
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True
    metadata = [
        {"category": "docs", "year": 2026},
        {"category": "guides", "year": 2025},
    ]

    await rag.insert(["first", "second"], metadata=metadata)

    assert engine.ainsert.await_args.kwargs["metadata"] == metadata


@pytest.mark.asyncio
async def test_insert_rejects_metadata_cardinality_mismatch() -> None:
    engine = SimpleNamespace(ainsert=AsyncMock())
    rag = HybridRAG(settings=cast(Any, SimpleNamespace()))
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True

    with pytest.raises(ValueError, match="metadata.*documents"):
        await rag.insert(["first", "second"], metadata=[{"category": "docs"}])

    engine.ainsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_insert_validates_metadata_against_configured_search_mapping() -> None:
    engine = SimpleNamespace(ainsert=AsyncMock())
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(
                filterable_metadata_fields={
                    "metadata.category": "token",
                    "metadata.year": "number",
                }
            ),
        )
    )
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True

    with pytest.raises(ValueError, match="metadata.year.*number"):
        await rag.insert(["first"], metadata=[{"year": "2026"}])

    engine.ainsert.assert_not_awaited()


@pytest.mark.parametrize(
    ("field_type", "value", "message"),
    [
        ("token", "x" * 8182, "8181"),
        ("number", 2**63, "int64"),
    ],
)
@pytest.mark.asyncio
async def test_insert_enforces_search_metadata_backend_bounds(
    field_type: str,
    value: object,
    message: str,
) -> None:
    engine = SimpleNamespace(ainsert=AsyncMock())
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(filterable_metadata_fields={"metadata.value": field_type}),
        )
    )
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True

    with pytest.raises(ValueError, match=message):
        await rag.insert(["first"], metadata=[{"value": value}])

    engine.ainsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_enqueue_persists_document_metadata() -> None:
    engine = SimpleNamespace(
        doc_status=SimpleNamespace(
            filter_keys=AsyncMock(return_value={"doc-a"}),
            upsert=AsyncMock(),
        ),
        full_docs=SimpleNamespace(
            upsert=AsyncMock(),
            index_done_callback=AsyncMock(),
        ),
    )

    await BaseRAGEngine.apipeline_enqueue_documents(
        cast(Any, engine),
        input=["document"],
        ids=["doc-a"],
        file_paths=["inline://doc-a"],
        metadata=[{"category": "docs", "year": 2026}],
        track_id="insert-test",
    )

    engine.full_docs.upsert.assert_awaited_once_with(
        {
            "doc-a": {
                "content": "document",
                "file_path": "inline://doc-a",
                "metadata": {"category": "docs", "year": 2026},
            }
        }
    )
    pending_status = engine.doc_status.upsert.await_args.args[0]["doc-a"]
    assert pending_status["metadata"] == {"category": "docs", "year": 2026}


def test_engine_chunk_builder_propagates_document_metadata() -> None:
    chunks = base_engine_module._build_document_chunks(
        [{"content": "chunk", "tokens": 1, "chunk_order_index": 0}],
        doc_id="doc-a",
        file_path="inline://doc-a",
        metadata={"category": "docs", "year": 2026},
    )

    assert next(iter(chunks.values()))["metadata"] == {
        "category": "docs",
        "year": 2026,
    }


def test_public_filter_types_are_exported_from_package_root() -> None:
    assert ExportedFilterConfig is FilterConfig
    assert ExportedFilterPredicate is FilterPredicate


def test_engine_api_query_request_preserves_typed_filter_config() -> None:
    request = EngineAPIQueryRequest.model_validate(
        {
            "query": "filtered question",
            "mode": "naive",
            "filter_config": {
                "predicates": [
                    {
                        "field": "metadata.category",
                        "operator": "eq",
                        "value": "docs",
                    }
                ]
            },
            "fusion_strategy": "score",
        }
    )

    param = request.to_query_params(False)

    assert isinstance(param.filter_config, FilterConfig)
    assert param.filter_config.predicates[0].field == "metadata.category"
    assert param.fusion_strategy == "score"


def test_query_param_repr_redacts_filter_values() -> None:
    filter_config = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.tenant_id",
                operator="eq",
                value="secret-tenant",
            )
        ]
    )
    param = EngineQueryParam(
        filter_config=filter_config,
        security_context=RetrievalSecurityContext(
            mandatory_filter=filter_config,
        ),
    )

    rendered = repr(param)

    assert "secret-tenant" not in rendered
    assert "filter_config" not in rendered
    assert "security_context" not in rendered


def test_request_security_context_is_conjoined_with_static_constraints() -> None:
    from hybridrag.engine.security import (
        reset_request_security_context,
        resolve_retrieval_security_context,
        set_request_security_context,
    )

    static_context = RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field="metadata.region",
                    operator="eq",
                    value="emea",
                )
            ]
        )
    )
    request_context = RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field="metadata.tenant_id",
                    operator="eq",
                    value="tenant-a",
                )
            ]
        )
    )

    token = set_request_security_context(request_context)
    try:
        effective = resolve_retrieval_security_context(static_context)
    finally:
        reset_request_security_context(token)

    assert effective is not None
    assert [config.predicates[0].field for config in effective.mandatory_filters] == [
        "metadata.region",
        "metadata.tenant_id",
    ]


def test_validated_principal_becomes_mandatory_tenant_filter(monkeypatch) -> None:
    from hybridrag.engine.security import principal_security_context

    monkeypatch.setenv("HYBRIDRAG_TENANT_FIELD", "metadata.tenant_id")
    monkeypatch.setenv("HYBRIDRAG_TENANT_CLAIM", "tenant_id")

    context = principal_security_context(
        {
            "username": "alice",
            "metadata": {"tenant_id": "tenant-a"},
        }
    )

    assert context is not None
    predicate = context.mandatory_filter.predicates[0]
    assert (predicate.field, predicate.operator, predicate.value) == (
        "metadata.tenant_id",
        "eq",
        "tenant-a",
    )


def test_data_query_param_preserves_retrieval_contract() -> None:
    filter_config = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.category",
                operator="eq",
                value="docs",
            )
        ]
    )
    param = EngineQueryParam(
        mode="naive",
        filter_config=filter_config,
        fusion_strategy="score",
    )

    data_param = base_engine_module._data_query_param(param)

    assert data_param.filter_config is filter_config
    assert data_param.fusion_strategy == "score"


@pytest.mark.asyncio
async def test_engine_api_text_pipeline_forwards_document_metadata() -> None:
    rag = SimpleNamespace(
        apipeline_enqueue_documents=AsyncMock(),
        apipeline_process_enqueue_documents=AsyncMock(),
    )
    metadata = [{"category": "docs"}, {"category": "guides"}]

    await pipeline_index_texts(
        cast(Any, rag),
        ["first", "second"],
        file_sources=["first.md", "second.md"],
        metadata=metadata,
        track_id="insert-test",
    )

    rag.apipeline_enqueue_documents.assert_awaited_once_with(
        input=["first", "second"],
        file_paths=["first.md", "second.md"],
        track_id="insert-test",
        metadata=metadata,
    )


def test_tenant_metadata_scope_is_server_owned(monkeypatch) -> None:
    from hybridrag.engine.security import (
        api_key_security_context,
        scope_document_metadata,
    )

    monkeypatch.setenv("HYBRIDRAG_TENANT_FIELD", "metadata.tenant_id")
    monkeypatch.setenv(
        "HYBRIDRAG_API_KEY_TENANTS",
        '{"tenant-a-key":"tenant-a"}',
    )
    context = api_key_security_context("tenant-a-key")

    scoped = scope_document_metadata(
        [{"tenant_id": "tenant-b"}, {"category": "docs"}],
        2,
        context,
    )

    assert scoped == [
        {"tenant_id": "tenant-a"},
        {"category": "docs", "tenant_id": "tenant-a"},
    ]


def test_document_ownership_rejects_another_tenant(monkeypatch) -> None:
    from hybridrag.engine.security import (
        api_key_security_context,
        require_document_ownership,
    )

    monkeypatch.setenv("HYBRIDRAG_TENANT_FIELD", "metadata.tenant_id")
    monkeypatch.setenv(
        "HYBRIDRAG_API_KEY_TENANTS",
        '{"tenant-a-key":"tenant-a"}',
    )
    context = api_key_security_context("tenant-a-key")

    with pytest.raises(PermissionError, match="not found"):
        require_document_ownership(
            {"metadata": {"tenant_id": "tenant-b"}},
            context,
        )


def test_tenant_principal_cannot_run_global_document_mutation(monkeypatch) -> None:
    from hybridrag.engine.security import (
        api_key_security_context,
        require_unscoped_document_operation,
    )

    monkeypatch.setenv("HYBRIDRAG_TENANT_FIELD", "metadata.tenant_id")
    monkeypatch.setenv(
        "HYBRIDRAG_API_KEY_TENANTS",
        '{"tenant-a-key":"tenant-a"}',
    )
    context = api_key_security_context("tenant-a-key")

    with pytest.raises(PermissionError, match="global document operation"):
        require_unscoped_document_operation(context)


def test_engine_api_maps_typed_retrieval_errors() -> None:
    capability = engine_query_routes._query_http_exception(
        RetrievalCapabilityError("score fusion is unavailable")
    )
    execution = engine_query_routes._query_http_exception(
        RetrievalExecutionError("backend detail")
    )

    assert capability.status_code == 422
    assert capability.detail["code"] == "retrieval_capability_error"
    assert execution.status_code == 502
    assert execution.detail == {
        "code": "retrieval_execution_error",
        "message": "Retrieval backend failed",
    }


def test_filter_config_compiles_backend_neutral_expression() -> None:
    object_id = ObjectId()
    tenant_id = UUID("12345678-1234-5678-1234-567812345678")
    config = FilterConfig(
        predicates=[
            FilterPredicate(field="metadata.category", operator="eq", value="docs"),
            FilterPredicate(field="metadata.created_at", operator="gte", value=10),
            FilterPredicate(field="metadata.object_id", operator="eq", value=object_id),
            FilterPredicate(field="metadata.tenant_id", operator="ne", value=tenant_id),
        ]
    )

    assert compile_filter_to_mql(config) == {
        "$and": [
            {"metadata.category": {"$eq": "docs"}},
            {"metadata.created_at": {"$gte": 10}},
            {"metadata.object_id": {"$eq": object_id}},
            {
                "metadata.tenant_id": {
                    "$ne": Binary.from_uuid(
                        tenant_id,
                        uuid_representation=UuidRepresentation.STANDARD,
                    )
                }
            },
        ]
    }
    assert compile_filter_to_atlas(config) == [
        {"equals": {"path": "metadata.category", "value": "docs"}},
        {"range": {"path": "metadata.created_at", "gte": 10}},
        {"equals": {"path": "metadata.object_id", "value": object_id}},
        {
            "compound": {
                "mustNot": [
                    {
                        "equals": {
                            "path": "metadata.tenant_id",
                            "value": Binary.from_uuid(
                                tenant_id,
                                uuid_representation=UuidRepresentation.STANDARD,
                            ),
                        }
                    }
                ]
            }
        },
    ]


def test_filter_config_supports_or_and_negation() -> None:
    config = FilterConfig(
        predicates=[
            FilterPredicate(field="category", operator="in", value=["a", "b"]),
            FilterPredicate(field="status", operator="eq", value="draft"),
        ],
        logic="or",
        negate=True,
    )

    assert compile_filter_to_mql(config) == {
        "$nor": [
            {
                "$or": [
                    {"category": {"$in": ["a", "b"]}},
                    {"status": {"$eq": "draft"}},
                ]
            }
        ]
    }
    assert compile_filter_to_atlas(config) == [
        {
            "compound": {
                "mustNot": [
                    {
                        "compound": {
                            "should": [
                                {"in": {"path": "category", "value": ["a", "b"]}},
                                {"equals": {"path": "status", "value": "draft"}},
                            ],
                            "minimumShouldMatch": 1,
                        }
                    }
                ]
            }
        }
    ]


def test_server_security_filter_is_always_conjoined_with_public_filter() -> None:
    security_context = RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field="metadata.tenant_id",
                    operator="eq",
                    value="tenant-a",
                )
            ]
        )
    )
    public_filter = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.category",
                operator="eq",
                value="docs",
            )
        ]
    )

    assert compile_retrieval_filter_to_mql(public_filter, security_context) == {
        "$and": [
            {"metadata.tenant_id": {"$eq": "tenant-a"}},
            {"metadata.category": {"$eq": "docs"}},
        ]
    }
    assert compile_retrieval_filter_to_atlas(public_filter, security_context) == [
        {"equals": {"path": "metadata.tenant_id", "value": "tenant-a"}},
        {"equals": {"path": "metadata.category", "value": "docs"}},
    ]


def test_filter_config_rejects_operator_injection() -> None:
    with pytest.raises(ValueError, match="field"):
        FilterPredicate(field="$where", operator="eq", value="bad")


def test_filter_predicate_rejects_nested_mapping_value() -> None:
    with pytest.raises(ValueError, match="scalar"):
        FilterPredicate(
            field="tenant_id",
            operator="eq",
            value={"$ne": "tenant-a"},
        )


def test_filter_membership_requires_one_bson_value_type() -> None:
    with pytest.raises(ValueError, match="one BSON type"):
        FilterPredicate(
            field="metadata.year",
            operator="in",
            value=[2025, "2026"],
        )


@pytest.mark.parametrize(
    "field",
    ["metadata..tenant", "metadata.tenant.", "metadata.\x00tenant"],
)
def test_filter_predicate_rejects_malformed_field_path(field: str) -> None:
    with pytest.raises(ValueError, match="field"):
        FilterPredicate(field=field, operator="eq", value="tenant-a")


@pytest.mark.parametrize("value", [[], list(range(101))])
def test_membership_filter_rejects_empty_or_oversized_lists(value: list[int]) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        FilterPredicate(field="year", operator="in", value=value)


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_filter_predicate_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        FilterPredicate(field="score", operator="gte", value=value)


def test_filter_predicate_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FilterPredicate(
            field="created_at",
            operator="gte",
            value=datetime(2026, 8, 18),
        )


def test_scalar_filter_rejects_collection_value() -> None:
    with pytest.raises(ValueError, match="scalar operator"):
        FilterPredicate(field="category", operator="eq", value=["docs"])


def test_filter_predicate_rejects_unsupported_scalar_type() -> None:
    with pytest.raises(ValueError, match="supported scalar"):
        FilterPredicate(field="tenant", operator="eq", value=object())


def test_filter_config_bounds_predicate_count() -> None:
    predicates = [
        FilterPredicate(field=f"field_{index}", operator="eq", value=index)
        for index in range(33)
    ]

    with pytest.raises(ValueError, match="between 1 and 32"):
        FilterConfig(predicates=predicates)


def test_filter_predicate_accepts_tagged_http_bson_values() -> None:
    object_id = ObjectId()
    tenant_id = UUID("12345678-1234-5678-1234-567812345678")
    config = FilterConfig.model_validate(
        {
            "predicates": [
                {
                    "field": "metadata.object_id",
                    "operator": "eq",
                    "value": {"type": "objectId", "value": str(object_id)},
                },
                {
                    "field": "metadata.tenant_id",
                    "operator": "eq",
                    "value": {"type": "uuid", "value": str(tenant_id)},
                },
            ]
        }
    )

    assert compile_filter_to_mql(config) == {
        "$and": [
            {"metadata.object_id": {"$eq": object_id}},
            {
                "metadata.tenant_id": {
                    "$eq": Binary.from_uuid(
                        tenant_id,
                        uuid_representation=UuidRepresentation.STANDARD,
                    )
                }
            },
        ]
    }


def test_filter_predicate_accepts_tagged_http_date_values() -> None:
    config = FilterConfig.model_validate(
        {
            "predicates": [
                {
                    "field": "metadata.published_on",
                    "operator": "gte",
                    "value": {"type": "date", "value": "2026-08-18"},
                },
                {
                    "field": "metadata.created_at",
                    "operator": "lt",
                    "value": {
                        "type": "datetime",
                        "value": "2026-08-19T00:00:00Z",
                    },
                },
            ]
        }
    )

    assert compile_filter_to_mql(config) == {
        "$and": [
            {"metadata.published_on": {"$gte": datetime(2026, 8, 18, tzinfo=UTC)}},
            {"metadata.created_at": {"$lt": datetime(2026, 8, 19, tzinfo=UTC)}},
        ]
    }


@pytest.mark.asyncio
async def test_external_rerank_missing_provider_fails_explicitly() -> None:
    with pytest.raises(RetrievalCapabilityError, match="not configured"):
        await apply_rerank_if_enabled(
            query="question",
            retrieved_docs=[{"content": "answer"}],
            global_config={},
        )


@pytest.mark.asyncio
async def test_external_rerank_failure_does_not_keep_original_ranking() -> None:
    rerank = AsyncMock(side_effect=RuntimeError("provider failed"))

    with pytest.raises(RetrievalExecutionError, match="External reranking failed"):
        await apply_rerank_if_enabled(
            query="question",
            retrieved_docs=[{"content": "answer"}],
            global_config={"rerank_model_func": rerank},
        )


def test_security_context_rejects_unscoped_kg_modes() -> None:
    filter_config = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.tenant_id",
                operator="eq",
                value="tenant-a",
            )
        ]
    )

    with pytest.raises(RetrievalCapabilityError, match="naive"):
        base_engine_module._enforce_security_context_mode(
            EngineQueryParam(
                mode="mix",
                security_context=RetrievalSecurityContext(
                    mandatory_filter=filter_config,
                ),
            )
        )


@pytest.mark.parametrize(
    ("value", "mql", "atlas"),
    [
        (
            True,
            {"metadata.category": {"$exists": True}},
            [{"exists": {"path": "metadata.category"}}],
        ),
        (
            False,
            {"metadata.category": {"$exists": False}},
            [{"compound": {"mustNot": [{"exists": {"path": "metadata.category"}}]}}],
        ),
    ],
)
def test_filter_config_compiles_exists_operator(
    value: bool,
    mql: dict[str, Any],
    atlas: list[dict[str, Any]],
) -> None:
    config = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.category",
                operator="exists",
                value=value,
            )
        ]
    )

    assert compile_filter_to_mql(config) == mql
    assert compile_filter_to_atlas(config) == atlas


def test_exists_filter_requires_boolean_value() -> None:
    with pytest.raises(ValueError, match="boolean"):
        FilterPredicate(field="metadata.category", operator="exists", value="yes")


@pytest.mark.asyncio
async def test_query_data_forwards_filter_and_fusion_to_internal_query_param() -> None:
    engine = SimpleNamespace(
        aquery_llm=AsyncMock(return_value={"data": {"chunks": []}})
    )
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(
                default_query_mode="mix",
                default_top_k=60,
                default_rerank_top_k=10,
                enable_rerank=True,
            ),
        )
    )
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True
    config = FilterConfig(
        predicates=[FilterPredicate(field="category", operator="eq", value="docs")]
    )

    await rag.query_data(
        "question",
        mode="naive",
        filter_config=config,
        fusion_strategy="score",
    )

    internal = engine.aquery_llm.await_args.kwargs["param"]
    assert internal.filter_config == config
    assert internal.fusion_strategy == "score"


@pytest.mark.asyncio
async def test_query_data_forwards_request_scoped_security_context() -> None:
    from hybridrag.engine.security import (
        reset_request_security_context,
        set_request_security_context,
    )

    engine = SimpleNamespace(
        aquery_llm=AsyncMock(return_value={"data": {"chunks": []}})
    )
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(
                default_query_mode="naive",
                default_top_k=60,
                default_rerank_top_k=10,
                enable_rerank=True,
            ),
        )
    )
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True
    request_context = RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field="metadata.tenant_id",
                    operator="eq",
                    value="tenant-a",
                )
            ]
        )
    )

    token = set_request_security_context(request_context)
    try:
        await rag.query_data("question", mode="naive")
    finally:
        reset_request_security_context(token)

    internal = engine.aquery_llm.await_args.kwargs["param"]
    assert internal.security_context == request_context


@pytest.mark.asyncio
async def test_query_rejects_filters_for_modes_without_safe_provenance() -> None:
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(
                default_query_mode="mix",
                default_top_k=60,
                default_rerank_top_k=10,
                enable_rerank=True,
                max_query_length=5000,
            ),
        )
    )
    rag._rag_engine = cast(Any, SimpleNamespace())
    rag._initialized = True
    config = FilterConfig(
        predicates=[FilterPredicate(field="category", operator="eq", value="docs")]
    )

    with pytest.raises(ValueError, match="naive"):
        await rag.query("question", mode="mix", filter_config=config)


@pytest.mark.asyncio
async def test_filtered_query_skips_unscoped_implicit_entity_expansion() -> None:
    entity_query = AsyncMock(return_value=[{"entity_name": "Tenant A Secret"}])
    engine = SimpleNamespace(
        entities_vdb=SimpleNamespace(query=entity_query),
        aquery=AsyncMock(return_value="answer"),
    )
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(
                default_query_mode="mix",
                default_top_k=60,
                default_rerank_top_k=10,
                enable_rerank=True,
                enable_implicit_expansion=True,
                enable_llm=True,
                max_query_length=5000,
            ),
        )
    )
    rag._rag_engine = cast(Any, engine)
    rag._initialized = True
    config = FilterConfig(
        predicates=[FilterPredicate(field="tenant_id", operator="eq", value="b")]
    )

    result = await rag.query("question", mode="naive", filter_config=config)

    assert result == "answer"
    entity_query.assert_not_awaited()
    assert engine.aquery.await_args.kwargs["query"] == "question"


def test_public_query_param_preserves_retrieval_options() -> None:
    config = FilterConfig(
        predicates=[FilterPredicate(field="category", operator="eq", value="docs")]
    )
    param = QueryParam(
        mode="naive",
        filter_config=config,
        fusion_strategy="rank",
    )._to_internal()

    assert param.filter_config == config
    assert param.fusion_strategy == "rank"


def test_external_reranker_identity_is_part_of_query_cache_key() -> None:
    first = AsyncMock()
    first.cache_identity = {"provider": "voyage", "model": "rerank-2.5"}
    second = AsyncMock()
    second.cache_identity = {"provider": "voyage", "model": "rerank-2.5-lite"}
    param = EngineQueryParam(rerank_strategy="external")

    assert _retrieval_cache_identity(
        param, {"rerank_model_func": first}
    ) != _retrieval_cache_identity(param, {"rerank_model_func": second})


class _Cursor:
    async def to_list(self, length=None):
        return []


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _QueryCache:
    def __init__(self) -> None:
        self.global_config = {"enable_llm_cache": True}
        self.entries: dict[str, dict[str, Any]] = {}

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    async def upsert(self, entries: dict[str, dict[str, Any]]) -> None:
        self.entries.update(entries)


class _ChunkStorage:
    cosine_better_than_threshold = 0.0

    async def query(
        self, query: str, top_k: int, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": "chunk-1",
                "content": "tenant-scoped evidence",
                "file_path": "tenant.md",
                "distance": 0.9,
            }
        ]


class _FailingChunkStorage:
    cosine_better_than_threshold = 0.0

    async def query(
        self, query: str, top_k: int, **kwargs: Any
    ) -> list[dict[str, Any]]:
        raise RetrievalExecutionError("score fusion failed")


class _HybridChunkStorage:
    cosine_better_than_threshold = 0.0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def hybrid_query(
        self, query: str, top_k: int, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [
            {
                "id": "chunk-1",
                "content": "evidence",
                "file_path": "doc.md",
                "distance": 0.9,
            }
        ]


@pytest.mark.asyncio
async def test_mix_orchestration_skips_client_embedding_for_automated_chunks() -> None:
    chunks = _HybridChunkStorage()
    chunks.global_config = {"vector_embedding_backend": "automated"}
    client_embedding = AsyncMock(side_effect=AssertionError("provider must not run"))
    text_chunks = SimpleNamespace(
        global_config={"kg_chunk_pick_method": "VECTOR"},
        embedding_func=client_embedding,
    )

    result = await _perform_kg_search(
        query="question",
        ll_keywords="",
        hl_keywords="",
        knowledge_graph_inst=cast(Any, SimpleNamespace()),
        entities_vdb=cast(Any, SimpleNamespace()),
        relationships_vdb=cast(Any, SimpleNamespace()),
        text_chunks_db=cast(Any, text_chunks),
        query_param=EngineQueryParam(mode="mix", enable_rerank=False),
        chunks_vdb=cast(Any, chunks),
    )

    client_embedding.assert_not_awaited()
    assert result["vector_chunks"][0]["content"] == "evidence"


@pytest.mark.asyncio
async def test_query_cache_isolated_by_effective_filter() -> None:
    model = AsyncMock(side_effect=["answer A", "answer B"])
    cache = _QueryCache()
    global_config = {
        "llm_model_func": model,
        "tokenizer": _Tokenizer(),
    }
    filter_a = FilterConfig(
        predicates=[FilterPredicate(field="tenant_id", operator="eq", value="a")]
    )
    filter_b = FilterConfig(
        predicates=[FilterPredicate(field="tenant_id", operator="eq", value="b")]
    )

    first = await naive_query(
        "same question",
        cast(Any, _HybridChunkStorage()),
        EngineQueryParam(mode="naive", enable_rerank=False, filter_config=filter_a),
        global_config,
        hashing_kv=cast(Any, cache),
    )
    second = await naive_query(
        "same question",
        cast(Any, _HybridChunkStorage()),
        EngineQueryParam(mode="naive", enable_rerank=False, filter_config=filter_b),
        global_config,
        hashing_kv=cast(Any, cache),
    )

    assert first is not None and first.content == "answer A"
    assert second is not None and second.content == "answer B"
    assert model.await_count == 2


@pytest.mark.asyncio
async def test_query_cache_isolated_by_fusion_strategy() -> None:
    model = AsyncMock(side_effect=["rank answer", "score answer"])
    cache = _QueryCache()
    global_config = {
        "llm_model_func": model,
        "tokenizer": _Tokenizer(),
    }

    rank_result = await naive_query(
        "same question",
        cast(Any, _HybridChunkStorage()),
        EngineQueryParam(mode="naive", enable_rerank=False, fusion_strategy="rank"),
        global_config,
        hashing_kv=cast(Any, cache),
    )
    score_result = await naive_query(
        "same question",
        cast(Any, _HybridChunkStorage()),
        EngineQueryParam(mode="naive", enable_rerank=False, fusion_strategy="score"),
        global_config,
        hashing_kv=cast(Any, cache),
    )

    assert rank_result is not None and rank_result.content == "rank answer"
    assert score_result is not None and score_result.content == "score answer"
    assert model.await_count == 2


@pytest.mark.asyncio
async def test_retrieval_failure_is_not_converted_to_empty_context() -> None:
    global_config = {
        "llm_model_func": AsyncMock(return_value="unused"),
        "tokenizer": _Tokenizer(),
    }

    with pytest.raises(RetrievalExecutionError, match="score fusion failed"):
        await naive_query(
            "question",
            cast(Any, _FailingChunkStorage()),
            EngineQueryParam(mode="naive", enable_rerank=False),
            global_config,
        )


@pytest.mark.asyncio
async def test_omitted_fusion_strategy_uses_score_fusion() -> None:
    storage = _HybridChunkStorage()
    result = await naive_query(
        "question",
        cast(Any, storage),
        EngineQueryParam(
            mode="naive",
            enable_rerank=False,
            only_need_context=True,
        ),
        {
            "llm_model_func": AsyncMock(return_value="unused"),
            "tokenizer": _Tokenizer(),
        },
    )

    assert result is not None
    assert storage.calls[0]["use_rank_fusion"] is False


@pytest.mark.asyncio
async def test_explicit_fusion_requires_native_hybrid_storage() -> None:
    storage = _ChunkStorage()

    with pytest.raises(RetrievalCapabilityError, match="native hybrid retrieval"):
        await naive_query(
            "question",
            cast(Any, storage),
            EngineQueryParam(
                mode="naive",
                enable_rerank=False,
                only_need_context=True,
                fusion_strategy="score",
            ),
            {
                "llm_model_func": AsyncMock(return_value="unused"),
                "tokenizer": _Tokenizer(),
            },
        )


def _mongo_vector_storage(
    version: str = "8.3.0",
) -> tuple[MongoVectorDBStorage, AsyncMock]:
    storage = object.__new__(MongoVectorDBStorage)
    storage.workspace = ""
    storage.namespace = "chunks"
    storage._collection_name = "chunks"
    storage._index_name = "vector_knn_index"
    storage.global_config = {
        "filterable_metadata_fields": {
            "metadata.category": "token",
            "metadata.tenant_id": "token",
        }
    }
    aggregate = AsyncMock(return_value=_Cursor())
    database = SimpleNamespace(command=AsyncMock(return_value={"version": version}))
    storage._data = cast(Any, SimpleNamespace(aggregate=aggregate, database=database))
    return storage, aggregate


@pytest.mark.asyncio
async def test_hybrid_pipeline_applies_filter_to_both_rank_fusion_inputs() -> None:
    storage, aggregate = _mongo_vector_storage()
    config = FilterConfig(
        predicates=[
            FilterPredicate(field="metadata.category", operator="eq", value="docs")
        ]
    )

    await storage.hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
        filter_config=config,
        use_rank_fusion=True,
    )

    assert aggregate.await_args is not None
    pipeline = aggregate.await_args.args[0]
    inputs = pipeline[0]["$rankFusion"]["input"]["pipelines"]
    assert inputs["vectorPipeline"][0]["$vectorSearch"]["filter"] == {
        "metadata.category": {"$eq": "docs"}
    }
    assert inputs["textPipeline"][0]["$search"]["compound"]["filter"] == [
        {"equals": {"path": "metadata.category", "value": "docs"}}
    ]


@pytest.mark.asyncio
async def test_hybrid_pipeline_cannot_drop_server_owned_filter() -> None:
    storage, aggregate = _mongo_vector_storage()
    security_context = RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field="metadata.tenant_id",
                    operator="eq",
                    value="tenant-a",
                )
            ]
        )
    )
    public_filter = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.category",
                operator="eq",
                value="docs",
            )
        ]
    )

    await storage.hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
        filter_config=public_filter,
        security_context=security_context,
    )

    pipeline = aggregate.await_args.args[0]
    inputs = pipeline[0]["$scoreFusion"]["input"]["pipelines"]
    assert inputs["vectorPipeline"][0]["$vectorSearch"]["filter"] == {
        "$and": [
            {"metadata.tenant_id": {"$eq": "tenant-a"}},
            {"metadata.category": {"$eq": "docs"}},
        ]
    }


@pytest.mark.asyncio
async def test_hybrid_query_rejects_filter_fields_outside_server_allowlist() -> None:
    storage, aggregate = _mongo_vector_storage()
    config = FilterConfig(
        predicates=[
            FilterPredicate(field="metadata.private", operator="eq", value="secret")
        ]
    )

    with pytest.raises(ValueError, match="not configured as filterable"):
        await storage.hybrid_query(
            "question",
            top_k=5,
            query_embedding=[0.1, 0.2],
            filter_config=config,
        )

    aggregate.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_query_rejects_operator_incompatible_with_mapping() -> None:
    storage, aggregate = _mongo_vector_storage()
    config = FilterConfig(
        predicates=[
            FilterPredicate(
                field="metadata.category",
                operator="gte",
                value="docs",
            )
        ]
    )

    with pytest.raises(ValueError, match="range.*token"):
        await storage.hybrid_query(
            "question",
            top_k=5,
            query_embedding=[0.1, 0.2],
            filter_config=config,
        )

    aggregate.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_pipeline_selects_score_fusion() -> None:
    storage, aggregate = _mongo_vector_storage()

    await storage.hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
    )

    assert aggregate.await_args is not None
    pipeline = aggregate.await_args.args[0]
    assert "$scoreFusion" in pipeline[0]


@pytest.mark.asyncio
async def test_exact_hybrid_search_uses_exact_vector_execution() -> None:
    storage, aggregate = _mongo_vector_storage()

    await storage.hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
        vector_search_mode="exact",
    )

    pipeline = aggregate.await_args.args[0]
    vector_search = pipeline[0]["$scoreFusion"]["input"]["pipelines"]["vectorPipeline"][
        0
    ]["$vectorSearch"]
    assert vector_search["exact"] is True
    assert "numCandidates" not in vector_search


@pytest.mark.asyncio
async def test_automated_embedding_uses_query_text_without_client_embedding() -> None:
    storage, aggregate = _mongo_vector_storage()
    storage.global_config.update(
        {
            "vector_embedding_backend": "automated",
            "automated_embedding_model": "voyage-4-large",
        }
    )

    await storage.hybrid_query("question", top_k=5)

    pipeline = aggregate.await_args.args[0]
    vector_search = pipeline[0]["$scoreFusion"]["input"]["pipelines"]["vectorPipeline"][
        0
    ]["$vectorSearch"]
    assert vector_search["path"] == "content"
    assert vector_search["query"] == {"text": "question"}
    assert vector_search["model"] == "voyage-4-large"
    assert "queryVector" not in vector_search


@pytest.mark.asyncio
async def test_automated_embedding_upsert_does_not_call_client_provider() -> None:
    storage, _ = _mongo_vector_storage()
    client_embedding = AsyncMock(side_effect=AssertionError("provider must not run"))
    storage.embedding_func = client_embedding
    storage.meta_fields = {"content", "full_doc_id"}
    storage._max_batch_size = 10
    storage.global_config.update(
        {
            "vector_embedding_backend": "automated",
            "automated_embedding_model": "voyage-4-large",
        }
    )
    storage._data.bulk_write = AsyncMock()

    await storage.upsert(
        {"chunk-1": {"content": "MongoDB embeds this", "full_doc_id": "doc-1"}}
    )

    client_embedding.assert_not_awaited()
    operation = storage._data.bulk_write.await_args.args[0][0]
    assert operation._doc["$set"]["content"] == "MongoDB embeds this"
    assert "vector" not in operation._doc["$set"]


@pytest.mark.asyncio
async def test_native_rerank_preserves_fusion_and_rerank_scores() -> None:
    storage, aggregate = _mongo_vector_storage()

    await storage.hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
        native_rerank_model="rerank-2.5",
    )

    pipeline = aggregate.await_args.args[0]
    rerank_index = next(
        index for index, stage in enumerate(pipeline) if "$rerank" in stage
    )
    assert pipeline[rerank_index - 1] == {
        "$set": {
            "content": {"$ifNull": ["$content", ""]},
            "fusion_score": {"$meta": "score"},
        }
    }
    assert pipeline[rerank_index]["$rerank"] == {
        "query": {"text": "question"},
        "path": "content",
        "numDocsToRerank": 10,
        "model": "rerank-2.5",
    }
    assert pipeline[rerank_index + 1] == {
        "$addFields": {
            "rerank_score": {"$meta": "score"},
            "score": {"$meta": "score"},
        }
    }
    assert {
        "$addFields": {
            "scoreDetails": {"$meta": "scoreDetails"},
            "score": {"$meta": "score"},
            "fusion_score": {"$meta": "score"},
        }
    } in pipeline
    assert pipeline[0]["$scoreFusion"]["combination"]["expression"] == {
        "$sum": [
            {"$multiply": ["$$vectorPipeline", 0.6]},
            {"$multiply": ["$$textPipeline", 0.4]},
        ]
    }


@pytest.mark.asyncio
async def test_server_explain_executes_and_redacts_query_vectors() -> None:
    storage, _ = _mongo_vector_storage()
    storage._data.database.command.return_value = {
        "queryPlanner": {"queryVector": [0.1, 0.2]}
    }

    explanation = await storage.explain_hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
    )

    assert explanation["kind"] == "server_explain"
    assert explanation["execution"]["queryPlanner"]["queryVector"] == "<redacted>"
    vector_search = explanation["plan"]["pipeline"][0]["$scoreFusion"]["input"][
        "pipelines"
    ]["vectorPipeline"][0]["$vectorSearch"]
    assert vector_search["queryVector"] == "<redacted>"


@pytest.mark.asyncio
async def test_compiled_plan_redacts_query_vectors() -> None:
    storage, _ = _mongo_vector_storage()
    embedding = AsyncMock(side_effect=AssertionError("provider must not run"))
    storage.embedding_func = embedding

    plan = await storage.compile_hybrid_query(
        "question",
        top_k=5,
    )

    vector_search = plan["pipeline"][0]["$scoreFusion"]["input"]["pipelines"][
        "vectorPipeline"
    ][0]["$vectorSearch"]
    assert vector_search["queryVector"] == "<redacted>"
    embedding.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_fusion_failure_does_not_execute_fallback_search() -> None:
    storage, aggregate = _mongo_vector_storage()
    aggregate.side_effect = PyMongoError("native fusion failed")

    with pytest.raises(RetrievalExecutionError, match="rank fusion failed"):
        await storage.hybrid_query(
            "question",
            top_k=5,
            query_embedding=[0.1, 0.2],
            use_rank_fusion=True,
        )

    assert aggregate.await_count == 1


@pytest.mark.asyncio
async def test_filtered_vector_failure_does_not_retry_without_filter() -> None:
    storage, aggregate = _mongo_vector_storage()
    aggregate.side_effect = [PyMongoError("filtered search failed"), _Cursor()]

    with pytest.raises(RetrievalExecutionError, match="filtered vector search failed"):
        await storage.query_with_filters(
            [0.1, 0.2],
            top_k=5,
            equality_filters={"tenant_id": "tenant-a"},
        )

    assert aggregate.await_count == 1


@pytest.mark.asyncio
async def test_score_fusion_detects_capability_by_execution() -> None:
    storage, aggregate = _mongo_vector_storage()
    storage._data.database.command.side_effect = AssertionError(
        "score fusion must not inspect a numeric server version"
    )
    aggregate.side_effect = OperationFailure("score fusion is not enabled")

    with pytest.raises(RetrievalCapabilityError, match="score fusion is unavailable"):
        await storage.hybrid_query(
            "question",
            top_k=5,
            query_embedding=[0.1, 0.2],
            use_rank_fusion=False,
        )

    storage._data.database.command.assert_not_awaited()
    assert aggregate.await_count == 1

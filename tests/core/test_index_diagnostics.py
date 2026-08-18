from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from hybridrag.core.rag import HybridRAG
from hybridrag.engine.exceptions import SearchIndexApplyError, SearchIndexLifecycleError
from hybridrag.engine.kg.mongo_impl import (
    MongoGraphStorage,
    MongoVectorDBStorage,
    _known_collections,
    get_or_create_collection,
)
from hybridrag.enhancements.filters import (
    FilterConfig,
    FilterPredicate,
    RetrievalSecurityContext,
)


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, length=None):
        return self.rows


@pytest.mark.asyncio
async def test_collection_cache_is_scoped_to_database_and_client() -> None:
    _known_collections.clear()
    first_client = object()
    second_client = object()
    first_collection = object()
    second_collection = object()
    first_db = SimpleNamespace(
        name="first",
        client=first_client,
        list_collection_names=AsyncMock(return_value=[]),
        create_collection=AsyncMock(return_value=first_collection),
        get_collection=Mock(),
    )
    second_db = SimpleNamespace(
        name="second",
        client=second_client,
        list_collection_names=AsyncMock(return_value=[]),
        create_collection=AsyncMock(return_value=second_collection),
        get_collection=Mock(),
    )

    assert (
        await get_or_create_collection(cast(Any, first_db), "chunks")
        is first_collection
    )
    assert (
        await get_or_create_collection(cast(Any, second_db), "chunks")
        is second_collection
    )

    first_db.create_collection.assert_awaited_once_with("chunks")
    second_db.create_collection.assert_awaited_once_with("chunks")


def _storage(
    indexes: list[dict] | None = None,
    embedding_dim: int = 1024,
    filterable_metadata_fields: dict[str, str] | None = None,
) -> tuple[MongoVectorDBStorage, Any]:
    collection = SimpleNamespace(
        list_search_indexes=AsyncMock(return_value=_Cursor(indexes or [])),
        create_search_index=AsyncMock(),
        update_search_index=AsyncMock(),
        drop_search_index=AsyncMock(),
        aggregate=AsyncMock(),
    )
    storage = object.__new__(MongoVectorDBStorage)
    storage.workspace = ""
    storage.namespace = "chunks"
    storage._collection_name = "chunks"
    storage._index_name = "vector_knn_index"
    storage._data = cast(Any, collection)
    storage.embedding_func = cast(Any, SimpleNamespace(embedding_dim=embedding_dim))
    storage.global_config = {
        "filterable_metadata_fields": filterable_metadata_fields or {}
    }
    return storage, collection


@pytest.mark.asyncio
async def test_search_index_statuses_are_normalized() -> None:
    indexes = [
        {
            "name": "vector_knn_index",
            "type": "vectorSearch",
            "status": "READY",
            "queryable": True,
        },
        {
            "name": "text_search_index_chunks",
            "type": "search",
            "status": "FAILED",
            "queryable": False,
            "statusDetail": "invalid analyzer",
        },
    ]
    storage, _ = _storage(indexes)
    indexes[0]["latestDefinition"] = storage.build_vector_index_definition()

    statuses = await storage.list_search_index_statuses()

    assert statuses == [
        {
            "name": "vector_knn_index",
            "type": "vectorSearch",
            "status": "ready",
            "queryable": True,
            "failure": None,
            "exists": True,
            "fresh": True,
            "transitioning": False,
            "healthy": True,
            "status_detail": None,
            "main_index": None,
            "staged_index": None,
            "backend_metadata": {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "status": "READY",
                "queryable": True,
                "latestDefinition": storage.build_vector_index_definition(),
            },
        },
        {
            "name": "text_search_index_chunks",
            "type": "search",
            "status": "failed",
            "queryable": False,
            "failure": "invalid analyzer",
            "exists": True,
            "fresh": False,
            "transitioning": False,
            "healthy": False,
            "status_detail": "invalid analyzer",
            "main_index": None,
            "staged_index": None,
            "backend_metadata": {
                "name": "text_search_index_chunks",
                "type": "search",
                "status": "FAILED",
                "queryable": False,
                "statusDetail": "invalid analyzer",
            },
        },
    ]


@pytest.mark.asyncio
async def test_search_index_statuses_preserve_queryable_transition_and_missing() -> (
    None
):
    storage, _ = _storage(
        [
            {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "status": "BUILDING",
                "queryable": True,
            }
        ]
    )

    statuses = await storage.list_search_index_statuses()

    assert statuses[0]["status"] == "building"
    assert statuses[0]["queryable"] is True
    assert statuses[0]["transitioning"] is True
    assert statuses[1] == {
        "name": "text_search_index_chunks",
        "type": "search",
        "status": "does_not_exist",
        "queryable": False,
        "failure": None,
        "exists": False,
        "fresh": False,
        "transitioning": False,
        "healthy": False,
        "status_detail": None,
        "main_index": None,
        "staged_index": None,
        "backend_metadata": {},
    }


@pytest.mark.asyncio
async def test_search_index_status_preserves_main_and_staged_generations() -> None:
    main = {"definitionVersion": {"createdAt": "old"}}
    staged = {"definitionVersion": {"createdAt": "new"}}
    storage, _ = _storage(
        [
            {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "status": "BUILDING",
                "queryable": True,
                "statusDetail": "staged definition is building",
                "mainIndex": main,
                "stagedIndex": staged,
            }
        ]
    )

    status = (await storage.list_search_index_statuses())[0]

    assert status["failure"] is None
    assert status["status_detail"] == "staged definition is building"
    assert status["main_index"] == main
    assert status["staged_index"] == staged


def test_binary_quantization_requires_byte_aligned_dimensions() -> None:
    storage, _ = _storage(embedding_dim=1025)

    with pytest.raises(ValueError, match="multiple of 8"):
        storage.build_vector_index_definition("binary")


def test_vector_index_definition_includes_explicit_quantization() -> None:
    storage, _ = _storage()

    definition = storage.build_vector_index_definition("scalar")

    vector_field = definition["fields"][0]
    assert vector_field["quantization"] == "scalar"
    assert vector_field["numDimensions"] == 1024


def test_vector_index_definition_supports_flat_and_custom_hnsw() -> None:
    storage, _ = _storage()

    flat = storage.build_vector_index_definition(indexing_method="flat")
    hnsw = storage.build_vector_index_definition(
        indexing_method="hnsw",
        hnsw_options={"maxEdges": 32, "numEdgeCandidates": 200},
    )

    assert flat["fields"][0]["indexingMethod"] == "flat"
    assert "hnswOptions" not in flat["fields"][0]
    assert hnsw["fields"][0]["hnswOptions"] == {
        "maxEdges": 32,
        "numEdgeCandidates": 200,
    }


def test_vector_index_definition_supports_automated_embedding_backend() -> None:
    storage, _ = _storage()
    storage.global_config.update(
        {
            "vector_embedding_backend": "automated",
            "automated_embedding_model": "voyage-4-large",
        }
    )

    definition = storage.build_vector_index_definition()

    assert definition["fields"][0] == {
        "type": "autoEmbed",
        "modality": "text",
        "path": "content",
        "model": "voyage-4-large",
    }


def test_automated_embedding_supports_current_vector_index_options() -> None:
    storage, _ = _storage()
    storage.global_config.update(
        {
            "vector_embedding_backend": "automated",
            "automated_embedding_model": "voyage-4-large",
        }
    )

    definition = storage.build_vector_index_definition(
        quantization="binaryNoRescore",
        indexing_method="hnsw",
        hnsw_options={"maxEdges": 64, "numEdgeCandidates": 3200},
        num_dimensions=1024,
        similarity="dotProduct",
    )

    assert definition["fields"][0] == {
        "type": "autoEmbed",
        "modality": "text",
        "path": "content",
        "model": "voyage-4-large",
        "quantization": "binaryNoRescore",
        "numDimensions": 1024,
        "similarity": "dotProduct",
        "indexingMethod": "hnsw",
        "hnswOptions": {"maxEdges": 64, "numEdgeCandidates": 3200},
    }


@pytest.mark.parametrize(
    "options",
    [
        {"maxEdges": 15},
        {"maxEdges": 65},
        {"numEdgeCandidates": 99},
        {"numEdgeCandidates": 3201},
    ],
)
def test_vector_index_definition_enforces_hnsw_documented_bounds(
    options: dict[str, int],
) -> None:
    storage, _ = _storage()

    with pytest.raises(ValueError, match="documented range"):
        storage.build_vector_index_definition(
            indexing_method="hnsw",
            hnsw_options=options,
        )


def test_vector_index_definition_rejects_hnsw_options_for_flat_index() -> None:
    storage, _ = _storage()

    with pytest.raises(ValueError, match="hnswOptions"):
        storage.build_vector_index_definition(
            indexing_method="flat",
            hnsw_options={"maxEdges": 32},
        )


def test_search_index_definitions_include_configured_metadata_fields() -> None:
    storage, _ = _storage(
        filterable_metadata_fields={
            "metadata.category": "token",
            "metadata.year": "number",
        }
    )

    vector_definition = storage.build_vector_index_definition()
    text_definition = storage.build_text_search_index_definition()

    assert {field["path"] for field in vector_definition["fields"][1:]} >= {
        "metadata.category",
        "metadata.year",
    }
    assert text_definition["mappings"]["fields"]["metadata"] == {
        "type": "document",
        "dynamic": False,
        "fields": {
            "category": {"type": "token"},
            "year": {"type": "number"},
        },
    }


@pytest.mark.asyncio
async def test_vector_index_plan_never_applies_implicitly() -> None:
    storage_definition = {
        "fields": [
            {
                "type": "vector",
                "numDimensions": 1024,
                "path": "vector",
                "similarity": "cosine",
            }
        ]
    }
    storage, collection = _storage(
        [
            {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "latestDefinition": storage_definition,
            }
        ]
    )

    plan = await storage.plan_vector_index("scalar")

    assert plan["action"] == "rebuild"
    assert plan["current"] == storage_definition
    assert plan["desired"]["fields"][0]["quantization"] == "scalar"
    collection.update_search_index.assert_not_awaited()
    collection.create_search_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_vector_index_plan_requires_explicit_call() -> None:
    storage, collection = _storage(
        [
            {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "latestDefinition": {"fields": []},
            }
        ]
    )

    result = await storage.apply_vector_index_plan("scalar")

    assert result["action"] == "rebuild"
    collection.update_search_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_text_index_plan_and_apply_are_explicit() -> None:
    storage, collection = _storage()

    plan = await storage.plan_text_search_index()

    assert plan["action"] == "create"
    collection.create_search_index.assert_not_awaited()

    applied = await storage.apply_text_search_index_plan()

    assert applied["action"] == "create"
    assert applied["acknowledged"] is True
    collection.create_search_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_vector_index_plan_retains_rollback_definition() -> None:
    previous = {"fields": [{"type": "vector", "numDimensions": 1024}]}
    storage, collection = _storage(
        [
            {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "latestDefinition": previous,
            }
        ]
    )

    applied = await storage.apply_vector_index_plan("scalar")
    rolled_back = await storage.rollback_vector_index(applied)

    assert applied["current"] == previous
    assert rolled_back == {
        "action": "restore",
        "index_name": "vector_knn_index",
        "definition": previous,
        "acknowledged": True,
    }
    collection.update_search_index.assert_any_await(
        "vector_knn_index",
        previous,
    )


@pytest.mark.asyncio
async def test_rollback_removes_index_created_by_plan() -> None:
    storage, collection = _storage([])

    applied = await storage.apply_vector_index_plan("scalar")
    rolled_back = await storage.rollback_vector_index(applied)

    assert rolled_back["action"] == "drop"
    collection.drop_search_index.assert_awaited_once_with("vector_knn_index")


@pytest.mark.asyncio
async def test_wait_for_search_index_observes_async_readiness() -> None:
    storage, collection = _storage()
    desired = storage.build_vector_index_definition()
    collection.list_search_indexes.side_effect = [
        _Cursor(
            [
                {
                    "name": "vector_knn_index",
                    "type": "vectorSearch",
                    "status": "BUILDING",
                    "queryable": False,
                }
            ]
        ),
        _Cursor(
            [
                {
                    "name": "vector_knn_index",
                    "type": "vectorSearch",
                    "status": "READY",
                    "queryable": True,
                    "latestDefinition": desired,
                }
            ]
        ),
    ]

    status = await storage.wait_for_search_index(
        "vector_knn_index",
        timeout_seconds=1,
        poll_interval_seconds=0,
    )

    assert status["healthy"] is True


@pytest.mark.asyncio
async def test_vector_index_plan_ignores_server_added_definition_defaults() -> None:
    storage, collection = _storage()
    observed = storage.build_vector_index_definition()
    observed["serverDefault"] = True
    observed["fields"][0]["serverDefault"] = "value"
    collection.list_search_indexes.return_value = _Cursor(
        [{"name": "vector_knn_index", "latestDefinition": observed}]
    )

    plan = await storage.plan_vector_index()

    assert plan["action"] == "noop"


@pytest.mark.asyncio
async def test_vector_index_plan_detects_behavioral_default_drift() -> None:
    storage, collection = _storage()
    observed = storage.build_vector_index_definition()
    observed["fields"][0]["quantization"] = "scalar"
    observed["fields"][0]["indexingMethod"] = "flat"
    collection.list_search_indexes.return_value = _Cursor(
        [{"name": "vector_knn_index", "latestDefinition": observed}]
    )

    plan = await storage.plan_vector_index()

    assert plan["action"] == "rebuild"


@pytest.mark.asyncio
async def test_multi_index_apply_preserves_partial_rollback_material() -> None:
    first = SimpleNamespace(
        apply_vector_index_plan=AsyncMock(
            return_value={"action": "create", "current": None}
        ),
        apply_text_search_index_plan=AsyncMock(side_effect=RuntimeError("failed")),
    )
    rag = HybridRAG()
    rag._initialized = True
    rag._rag_engine = cast(
        Any,
        SimpleNamespace(
            chunks_vdb=first,
            entities_vdb=SimpleNamespace(),
            relationships_vdb=SimpleNamespace(),
            chunk_entity_relation_graph=SimpleNamespace(),
        ),
    )

    with pytest.raises(SearchIndexApplyError) as caught:
        await rag.apply_search_index_plans()

    assert caught.value.applied_plans == [
        {
            "action": "create",
            "current": None,
            "storage": "chunks",
            "index_kind": "vector",
        }
    ]


@pytest.mark.asyncio
async def test_wait_for_search_index_raises_on_failed_build() -> None:
    storage, collection = _storage()
    collection.list_search_indexes.return_value = _Cursor(
        [
            {
                "name": "vector_knn_index",
                "type": "vectorSearch",
                "status": "FAILED",
                "queryable": False,
                "statusDetail": "invalid definition",
            }
        ]
    )

    with pytest.raises(SearchIndexLifecycleError, match="invalid definition"):
        await storage.wait_for_search_index(
            "vector_knn_index",
            timeout_seconds=1,
            poll_interval_seconds=0,
        )


@pytest.mark.asyncio
async def test_hybrid_explain_returns_pipeline_without_querying() -> None:
    storage, collection = _storage()

    explanation = await storage.hybrid_query(
        "question",
        top_k=5,
        query_embedding=[0.1, 0.2],
        use_rank_fusion=True,
        explain=True,
    )

    assert explanation["fusion_strategy"] == "rank"
    assert "$rankFusion" in explanation["pipeline"][0]
    collection.aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_public_diagnostic_methods_delegate_to_chunk_storage() -> None:
    async def explain_hybrid_query(*args, use_rank_fusion: bool, **kwargs):
        return {
            "fusion_strategy": "rank" if use_rank_fusion else "score",
            "pipeline": [],
        }

    chunks = SimpleNamespace(
        list_search_index_statuses=AsyncMock(return_value=[{"status": "ready"}]),
        compile_hybrid_query=AsyncMock(side_effect=explain_hybrid_query),
        explain_hybrid_query=AsyncMock(
            return_value={"kind": "server_explain", "execution": {}}
        ),
        plan_vector_index=AsyncMock(return_value={"action": "noop"}),
    )
    entities = SimpleNamespace(
        list_search_index_statuses=AsyncMock(
            return_value=[{"name": "entities-vector", "status": "ready"}]
        )
    )
    relationships = SimpleNamespace(
        list_search_index_statuses=AsyncMock(
            return_value=[{"name": "relationships-vector", "status": "building"}]
        )
    )
    rag = HybridRAG(
        settings=cast(
            Any,
            SimpleNamespace(
                default_query_mode="naive",
                default_top_k=5,
                enable_rerank=True,
            ),
        )
    )
    rag._rag_engine = cast(
        Any,
        SimpleNamespace(
            chunks_vdb=chunks,
            entities_vdb=entities,
            relationships_vdb=relationships,
        ),
    )
    rag._initialized = True
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
    rag.retrieval_security_context = security_context

    assert await rag.list_search_indexes() == [
        {"status": "ready", "storage": "chunks"},
        {"name": "entities-vector", "status": "ready", "storage": "entities"},
        {
            "name": "relationships-vector",
            "status": "building",
            "storage": "relationships",
        },
    ]
    assert (await rag.compile_query_plan("question"))["fusion_strategy"] == "score"
    assert (await rag.explain_query("question"))["kind"] == "server_explain"
    assert (
        chunks.compile_hybrid_query.await_args.kwargs["security_context"]
        == security_context
    )
    assert (
        chunks.explain_hybrid_query.await_args.kwargs["security_context"]
        == security_context
    )
    assert (await rag.plan_vector_index("scalar"))["action"] == "noop"


@pytest.mark.asyncio
async def test_wait_for_search_indexes_handles_names_reused_across_storages() -> None:
    rag = HybridRAG()
    rag.list_search_indexes = AsyncMock(
        return_value=[
            {
                "name": "vector_knn_index",
                "storage": storage,
                "status": "ready",
                "healthy": True,
            }
            for storage in ("chunks", "entities", "relationships")
        ]
        + [
            {
                "name": "text_search_index_chunks",
                "storage": "chunks",
                "status": "ready",
                "healthy": True,
            },
            {
                "name": "entity_id_search_idx",
                "storage": "graph",
                "status": "ready",
                "healthy": True,
            },
        ]
    )

    statuses = await rag.wait_for_search_indexes(
        [
            "vector_knn_index",
            "text_search_index_chunks",
            "vector_knn_index",
            "vector_knn_index",
            "entity_id_search_idx",
        ],
        timeout_seconds=0.1,
        poll_interval_seconds=0,
    )

    assert len(statuses) == 5


@pytest.mark.asyncio
async def test_wait_for_search_indexes_uses_applied_plan_definition() -> None:
    desired = {"fields": [{"type": "vector", "quantization": "scalar"}]}
    rag = HybridRAG()
    rag.list_search_indexes = AsyncMock(
        return_value=[
            {
                "name": "vector_knn_index",
                "storage": "chunks",
                "status": "ready",
                "queryable": True,
                "fresh": False,
                "healthy": False,
                "backend_metadata": {"latestDefinition": desired},
            }
        ]
    )

    statuses = await rag.wait_for_search_indexes(
        [
            {
                "index_name": "vector_knn_index",
                "storage": "chunks",
                "desired": desired,
            }
        ],
        timeout_seconds=0.1,
        poll_interval_seconds=0,
    )

    assert statuses[0]["healthy"] is True


@pytest.mark.asyncio
async def test_graph_search_index_plan_can_be_rolled_back() -> None:
    graph = object.__new__(MongoGraphStorage)
    graph.collection = cast(
        Any,
        SimpleNamespace(
            drop_search_index=AsyncMock(),
            update_search_index=AsyncMock(),
        ),
    )

    result = await graph.rollback_search_index_plan(
        {
            "action": "create",
            "index_name": "entity_id_search_idx",
            "current": None,
        }
    )

    assert result == {
        "action": "drop",
        "index_name": "entity_id_search_idx",
        "acknowledged": True,
    }
    graph.collection.drop_search_index.assert_awaited_once_with("entity_id_search_idx")


@pytest.mark.asyncio
async def test_public_all_index_rollback_dispatches_in_reverse_order() -> None:
    chunks = SimpleNamespace(
        rollback_vector_index=AsyncMock(return_value={"action": "drop"}),
        rollback_text_search_index=AsyncMock(return_value={"action": "drop"}),
    )
    entities = SimpleNamespace(
        rollback_vector_index=AsyncMock(return_value={"action": "drop"})
    )
    relationships = SimpleNamespace(
        rollback_vector_index=AsyncMock(return_value={"action": "drop"})
    )
    graph = SimpleNamespace(
        rollback_search_index_plan=AsyncMock(return_value={"action": "drop"})
    )
    rag = HybridRAG()
    rag._rag_engine = cast(
        Any,
        SimpleNamespace(
            chunks_vdb=chunks,
            entities_vdb=entities,
            relationships_vdb=relationships,
            chunk_entity_relation_graph=graph,
        ),
    )
    rag._initialized = True
    applied = [
        {"storage": "chunks", "index_kind": "vector", "index_name": "cv"},
        {"storage": "chunks", "index_kind": "text", "index_name": "ct"},
        {"storage": "entities", "index_kind": "vector", "index_name": "ev"},
        {
            "storage": "relationships",
            "index_kind": "vector",
            "index_name": "rv",
        },
        {"storage": "graph", "index_kind": "text", "index_name": "gt"},
    ]

    rolled_back = await rag.rollback_search_index_plans(applied)

    assert [result["storage"] for result in rolled_back] == [
        "graph",
        "relationships",
        "entities",
        "chunks",
        "chunks",
    ]
    graph.rollback_search_index_plan.assert_awaited_once_with(applied[4])
    relationships.rollback_vector_index.assert_awaited_once_with(applied[3])
    entities.rollback_vector_index.assert_awaited_once_with(applied[2])
    chunks.rollback_text_search_index.assert_awaited_once_with(applied[1])
    chunks.rollback_vector_index.assert_awaited_once_with(applied[0])

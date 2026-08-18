from __future__ import annotations

import json

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from hybridrag.api import main as api_main
from hybridrag.engine.exceptions import RetrievalCapabilityError


class _FakeRAG:
    def __init__(self) -> None:
        self.stream_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self.insert_calls: list[dict] = []
        self.source_calls: list[dict] = []

    async def get_status(self) -> dict:
        return {"initialized": True}

    async def insert(self, documents, ids=None, metadata=None) -> None:
        self.insert_calls.append(
            {"documents": documents, "ids": ids, "metadata": metadata}
        )

    async def query_with_sources(
        self,
        query: str,
        mode: str,
        top_k: int,
        rerank_top_k: int,
        enable_rerank: bool,
        filter_config=None,
        fusion_strategy=None,
        vector_search_mode="ann",
        rerank_strategy="native",
        native_rerank_model="rerank-2.5",
    ) -> dict:
        self.source_calls.append(
            {
                "filter_config": filter_config,
                "fusion_strategy": fusion_strategy,
                "vector_search_mode": vector_search_mode,
                "rerank_strategy": rerank_strategy,
                "native_rerank_model": native_rerank_model,
                "rerank_top_k": rerank_top_k,
                "enable_rerank": enable_rerank,
            }
        )
        return {
            "answer": "MongoDB Atlas Vector Search enables semantic retrieval.",
            "context": "Atlas Vector Search stores and retrieves embedded chunks.",
            "mode": mode,
            "references": [
                {"reference_id": "ref-1", "file_path": "docs/atlas-vector-search.md"}
            ],
            "metadata": {"query_mode": mode},
        }

    async def query(
        self,
        query: str,
        mode: str,
        top_k: int,
        rerank_top_k: int,
        enable_rerank: bool,
        filter_config=None,
        fusion_strategy=None,
        vector_search_mode="ann",
        rerank_strategy="native",
        native_rerank_model="rerank-2.5",
    ) -> str:
        from hybridrag.engine.security import get_request_security_context

        self.query_calls.append(
            {
                "filter_config": filter_config,
                "fusion_strategy": fusion_strategy,
                "vector_search_mode": vector_search_mode,
                "rerank_strategy": rerank_strategy,
                "native_rerank_model": native_rerank_model,
                "security_context": get_request_security_context(),
            }
        )
        return "MongoDB answer"

    async def stream_query(
        self,
        query: str,
        mode: str,
        top_k: int,
        rerank_top_k: int,
        enable_rerank: bool,
        include_context: bool,
        include_references: bool,
        filter_config=None,
        fusion_strategy=None,
        vector_search_mode="ann",
        rerank_strategy="native",
        native_rerank_model="rerank-2.5",
    ) -> dict:
        self.stream_calls.append(
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "rerank_top_k": rerank_top_k,
                "enable_rerank": enable_rerank,
                "include_context": include_context,
                "include_references": include_references,
                "filter_config": filter_config,
                "fusion_strategy": fusion_strategy,
                "vector_search_mode": vector_search_mode,
                "rerank_strategy": rerank_strategy,
                "native_rerank_model": native_rerank_model,
            }
        )

        async def _chunks():
            yield "MongoDB "
            yield "streamed answer"

        return {
            "mode": mode,
            "context": "Streaming context",
            "references": [{"reference_id": "ref-stream", "file_path": "docs/live.md"}],
            "metadata": {"query_mode": mode, "transport": "ndjson"},
            "response_iterator": _chunks(),
        }

    async def explain_query(self, query: str, **kwargs) -> dict:
        return {"query": query, "mode": kwargs.get("mode"), "pipeline": []}

    async def list_search_indexes(self) -> list[dict]:
        return [{"name": "vector_knn_index", "status": "ready"}]

    async def delete_document(self, doc_id: str) -> None:
        return None


def test_ingest_api_forwards_per_document_metadata(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/ingest",
            json={
                "documents": ["first", "second"],
                "metadata": [
                    {"category": "docs", "year": 2026},
                    {"category": "guides", "year": 2025},
                ],
            },
        )

    assert response.status_code == 200
    assert fake_rag.insert_calls[0]["metadata"] == [
        {"category": "docs", "year": 2026},
        {"category": "guides", "year": 2025},
    ]


def test_query_api_includes_references_when_requested(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "What is vector search?",
                "mode": "mix",
                "top_k": 3,
                "include_context": True,
                "include_references": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["context"]
    assert payload["references"] == [
        {"reference_id": "ref-1", "file_path": "docs/atlas-vector-search.md"}
    ]
    assert payload["metadata"]["mode"] == "mix"


def test_query_api_requires_stream_endpoint_when_stream_requested(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "Stream this please",
                "mode": "mix",
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Use /v1/query/stream when stream=true"


def test_query_stream_endpoint_returns_ndjson(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query/stream",
            json={
                "query": "What does the stack support?",
                "mode": "mix",
                "include_context": True,
                "include_references": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) == 3
    assert '"metadata"' in lines[0]
    assert '"references"' in lines[0]
    assert '"answer": "MongoDB "' in lines[1]
    assert '"answer": "streamed answer"' in lines[2]
    assert fake_rag.stream_calls[0]["include_references"]


def test_query_api_forwards_filter_and_fusion_options(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "Filtered query",
                "mode": "naive",
                "filter_config": {
                    "predicates": [
                        {"field": "category", "operator": "eq", "value": "docs"}
                    ]
                },
                "fusion_strategy": "score",
            },
        )

    assert response.status_code == 200
    assert fake_rag.query_calls[0]["filter_config"].predicates[0].field == "category"
    assert fake_rag.query_calls[0]["fusion_strategy"] == "score"


def test_query_api_maps_typed_retrieval_capability_error(monkeypatch):
    fake_rag = _FakeRAG()

    async def _failed_query(*args, **kwargs):
        raise RetrievalCapabilityError("score fusion is unavailable")

    fake_rag.query = _failed_query

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query",
            json={"query": "Use score fusion", "mode": "naive"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "retrieval_capability_error",
        "message": "score fusion is unavailable",
    }


def test_query_api_forwards_filters_on_source_aware_surface(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "Filtered query",
                "mode": "naive",
                "include_references": True,
                "filter_config": {
                    "predicates": [
                        {"field": "category", "operator": "eq", "value": "docs"}
                    ]
                },
            },
        )

    assert response.status_code == 200
    assert fake_rag.source_calls[0]["filter_config"].predicates[0].value == "docs"
    assert fake_rag.source_calls[0]["rerank_top_k"] == 10
    assert fake_rag.source_calls[0]["enable_rerank"] is True


def test_query_stream_forwards_filter_and_fusion_options(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query/stream",
            json={
                "query": "Filtered stream",
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
            },
        )

    assert response.status_code == 200
    assert fake_rag.stream_calls[0]["filter_config"].predicates[0].value == "docs"
    assert fake_rag.stream_calls[0]["fusion_strategy"] == "score"


def test_query_explain_and_index_status_endpoints(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)
    monkeypatch.setenv("HYBRIDRAG_OPERATOR_API_KEY", "operator-token")

    with TestClient(api_main.create_app()) as client:
        explain = client.post(
            "/v1/query/explain",
            headers={"X-API-Key": "operator-token"},
            json={"query": "Explain retrieval", "mode": "naive"},
        )
        indexes = client.get(
            "/v1/search-indexes",
            headers={"X-API-Key": "operator-token"},
        )

    assert explain.status_code == 200
    assert explain.json()["pipeline"] == []
    assert indexes.status_code == 200
    assert indexes.json()[0]["status"] == "ready"


def test_diagnostic_endpoints_require_configured_operator_key(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)
    monkeypatch.delenv("HYBRIDRAG_OPERATOR_API_KEY", raising=False)

    with TestClient(api_main.create_app()) as client:
        explain = client.post(
            "/v1/query/explain",
            json={"query": "Explain retrieval", "mode": "naive"},
        )
        indexes = client.get("/v1/search-indexes")

    assert explain.status_code == 503
    assert indexes.status_code == 503


def test_public_api_key_cannot_access_operator_diagnostics(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)
    monkeypatch.setenv("HYBRIDRAG_API_KEY", "public-token")
    monkeypatch.setenv("HYBRIDRAG_OPERATOR_API_KEY", "operator-token")

    with TestClient(api_main.create_app()) as client:
        response = client.get(
            "/v1/search-indexes",
            headers={"X-API-Key": "public-token"},
        )

    assert response.status_code == 403


def test_query_api_optional_key_guard(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)
    monkeypatch.setenv("HYBRIDRAG_API_KEY", "secret-token")

    with TestClient(api_main.create_app()) as client:
        forbidden = client.post("/v1/query", json={"query": "Hello", "mode": "mix"})
        allowed = client.post(
            "/v1/query",
            headers={"X-API-Key": "secret-token"},
            json={"query": "Hello", "mode": "mix"},
        )

    assert forbidden.status_code == 403
    assert allowed.status_code == 200
    monkeypatch.delenv("HYBRIDRAG_API_KEY", raising=False)


def test_query_api_binds_api_key_to_mandatory_tenant_filter(monkeypatch):
    fake_rag = _FakeRAG()

    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return fake_rag

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)
    monkeypatch.setenv("HYBRIDRAG_API_KEY", "tenant-a-key")
    monkeypatch.setenv("HYBRIDRAG_TENANT_FIELD", "metadata.tenant_id")
    monkeypatch.setenv(
        "HYBRIDRAG_API_KEY_TENANTS",
        json.dumps({"tenant-a-key": "tenant-a"}),
    )

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/v1/query",
            headers={"X-API-Key": "tenant-a-key"},
            json={"query": "Tenant-scoped query", "mode": "naive"},
        )

    assert response.status_code == 200
    security_context = fake_rag.query_calls[0]["security_context"]
    predicate = security_context.mandatory_filter.predicates[0]
    assert predicate.field == "metadata.tenant_id"
    assert predicate.operator == "eq"
    assert predicate.value == "tenant-a"


def test_engine_jwt_binds_principal_to_mandatory_tenant_filter(monkeypatch):
    from hybridrag.engine.api import utils_api
    from hybridrag.engine.security import get_request_security_context

    monkeypatch.setattr(utils_api, "auth_configured", True)
    monkeypatch.setattr(
        utils_api.auth_handler,
        "validate_token",
        lambda token: {
            "username": "alice",
            "role": "user",
            "metadata": {"tenant_id": "tenant-a"},
        },
    )
    monkeypatch.setenv("HYBRIDRAG_TENANT_FIELD", "metadata.tenant_id")
    monkeypatch.setenv("HYBRIDRAG_TENANT_CLAIM", "tenant_id")

    app = FastAPI()
    auth_dependency = utils_api.get_combined_auth_dependency()

    @app.get("/secured", dependencies=[Depends(auth_dependency)])
    async def secured():
        context = get_request_security_context()
        assert context is not None
        return context.mandatory_filter.model_dump(mode="json")

    with TestClient(app) as client:
        response = client.get(
            "/secured",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert response.json()["predicates"][0] == {
        "field": "metadata.tenant_id",
        "operator": "eq",
        "value": "tenant-a",
    }


def test_query_api_optional_rate_limit(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

    monkeypatch.setattr(api_main, "create_hybridrag", _fake_create_hybridrag)
    monkeypatch.setenv("HYBRIDRAG_RATE_LIMIT_PER_WINDOW", "1")
    monkeypatch.setenv("HYBRIDRAG_RATE_LIMIT_WINDOW_SECONDS", "60")
    api_main._rate_limit_state.clear()

    with TestClient(api_main.create_app()) as client:
        first = client.post("/v1/query", json={"query": "Hello", "mode": "mix"})
        second = client.post("/v1/query", json={"query": "Hello again", "mode": "mix"})

    assert first.status_code == 200
    assert second.status_code == 429
    api_main._rate_limit_state.clear()
    monkeypatch.delenv("HYBRIDRAG_RATE_LIMIT_PER_WINDOW", raising=False)
    monkeypatch.delenv("HYBRIDRAG_RATE_LIMIT_WINDOW_SECONDS", raising=False)

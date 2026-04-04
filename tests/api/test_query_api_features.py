from __future__ import annotations

from fastapi.testclient import TestClient

from hybridrag.api import main as api_main


class _FakeRAG:
    def __init__(self) -> None:
        self.stream_calls: list[dict] = []

    async def get_status(self) -> dict:
        return {"initialized": True}

    async def insert(self, documents, ids=None) -> None:
        return None

    async def query_with_sources(self, query: str, mode: str, top_k: int) -> dict:
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
    ) -> str:
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

    async def delete_document(self, doc_id: str) -> None:
        return None


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
    assert fake_rag.stream_calls[0]["include_references"] is True


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

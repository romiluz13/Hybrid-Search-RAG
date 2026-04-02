from __future__ import annotations

from fastapi.testclient import TestClient

from hybridrag.api import main as api_main


class _FakeRAG:
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

    async def query(self, query: str, mode: str, top_k: int, rerank_top_k: int, enable_rerank: bool) -> str:
        return "MongoDB answer"

    async def delete_document(self, doc_id: str) -> None:
        return None


def test_query_api_includes_references_when_requested(monkeypatch):
    async def _fake_create_hybridrag(auto_initialize: bool = True):
        return _FakeRAG()

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

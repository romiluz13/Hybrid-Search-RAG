#!/usr/bin/env python3
"""
Deterministic live release gate for HybridRAG.

This script uses:
- real MongoDB Atlas Local / atlas-local:preview
- real Voyage embeddings
- real LLM generation through an OpenAI-compatible endpoint
- a fresh seeded database and workspace per run

Run:
    source .venv/bin/activate
    export MONGODB_URI='mongodb://localhost:27018/?directConnection=true'
    export VOYAGE_API_KEY='...'
    export OPENAI_API_KEY='...'
    export OPENAI_BASE_URL='https://your-openai-compatible-endpoint/v1'  # optional
    export OPENAI_EXTRA_HEADERS='{"api-key":"..."}'  # optional JSON string
    python tests/e2e_real_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from pydantic import SecretStr
from pymongo import AsyncMongoClient

from hybridrag import create_hybridrag
from hybridrag.config import Settings, get_settings
from hybridrag.core.mongodb_client import close_shared_client
from hybridrag.engine.kg.mongo_impl import (
    ClientManager,
    get_collection_name,
)
from hybridrag.engine.kg.shared_storage import finalize_share_data

load_dotenv()


LIVE_DOCS: list[dict[str, str]] = [
    {
        "id": "atlas-vector-overview",
        "content": (
            "MongoDB Atlas Vector Search stores embedding vectors and supports "
            "semantic retrieval over document chunks for retrieval-augmented "
            "generation systems."
        ),
    },
    {
        "id": "hybrid-search-primer",
        "content": (
            "Hybrid search combines lexical search and vector search so exact "
            "terms and semantic meaning work together in a single retrieval flow."
        ),
    },
    {
        "id": "graph-rag-primer",
        "content": (
            "Knowledge graphs connect entities and relationships, which helps "
            "answer multi-hop questions and improves explainability in retrieval "
            "pipelines."
        ),
    },
    {
        "id": "numcandidates-tuning",
        "content": (
            "The numCandidates setting trades off recall and latency in vector "
            "search by widening the candidate pool before final scoring."
        ),
    },
    {
        "id": "reranking-benefits",
        "content": (
            "Reranking improves result quality by reordering retrieved chunks "
            "after the initial search stage with a stronger relevance model."
        ),
    },
    {
        "id": "production-rag-requirements",
        "content": (
            "Production RAG systems need citations, observability, seeded tests, "
            "and explicit failures rather than hidden fallback behavior."
        ),
    },
    {
        "id": "memory-multiturn",
        "content": (
            "Conversation memory stores sessions and messages so multi-turn "
            "assistants can answer with prior context when needed."
        ),
    },
    {
        "id": "single-db-architecture",
        "content": (
            "MongoDB can hold document chunks, graph edges, sessions, metadata, "
            "and operational telemetry in a single AI application architecture."
        ),
    },
    {
        "id": "modern-ai-app-surface",
        "content": (
            "Streaming responses, structured outputs, and source attribution are "
            "common expectations for modern AI application reference implementations."
        ),
    },
    {
        "id": "reference-boilerplate",
        "content": (
            "A strong reference boilerplate should expose realistic API surfaces, "
            "reproducible tests, and end-to-end validation with real providers."
        ),
    },
]


MODE_CASES = [
    {
        "name": "Local Retrieval",
        "query": "What is MongoDB Atlas Vector Search used for?",
        "mode": "local",
        "minimum_context": 80,
        "expects": ["vector", "search", "retrieval"],
    },
    {
        "name": "Global Graph Retrieval",
        "query": "Why would a RAG system use a knowledge graph?",
        "mode": "global",
        "minimum_context": 80,
        "expects": ["graph", "knowledge", "relationships"],
    },
    {
        "name": "Hybrid Graph Retrieval",
        "query": "How does hybrid search combine lexical and semantic relevance?",
        "mode": "hybrid",
        "minimum_context": 80,
        "expects": ["hybrid", "lexical", "semantic"],
    },
    {
        "name": "Naive Vector Retrieval",
        "query": "What does numCandidates affect in vector search?",
        "mode": "naive",
        "minimum_context": 80,
        "expects": ["numcandidates", "latency", "recall"],
    },
    {
        "name": "Mix Retrieval",
        "query": "Why is reranking useful after retrieval?",
        "mode": "mix",
        "minimum_context": 80,
        "expects": ["reranking", "quality", "retrieval"],
    },
]


STRESS_CASES = [
    ("What is MongoDB Atlas Vector Search used for?", "mix", False),
    ("How does hybrid search combine lexical and semantic relevance?", "mix", True),
    ("Why would a RAG system use a knowledge graph?", "hybrid", False),
    ("What does numCandidates affect in vector search?", "naive", False),
    ("Why is reranking useful after retrieval?", "mix", False),
    ("What should a production RAG boilerplate expose?", "mix", True),
    ("How does conversation memory help multi-turn assistants?", "hybrid", False),
    ("Why keep chunks, graph edges, and sessions in one database?", "mix", True),
    ("What makes citations important in RAG systems?", "mix", True),
    ("Why are seeded tests better than ad-hoc demo confidence?", "mix", False),
]


@dataclass
class TestRecord:
    name: str
    passed: bool
    details: str
    duration_s: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveContext:
    mongodb_uri: str
    database_name: str
    workspace: str
    settings: Settings


RESULTS: list[TestRecord] = []


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def record(name: str, passed: bool, details: str, start: float, **extra: Any) -> None:
    duration_s = time.perf_counter() - start
    RESULTS.append(
        TestRecord(
            name=name,
            passed=passed,
            details=details,
            duration_s=round(duration_s, 2),
            extra=extra,
        )
    )
    icon = "PASS" if passed else "FAIL"
    print(f"[{icon}] {name} ({duration_s:.2f}s)")
    print(f"       {details}")


def build_live_settings() -> LiveContext:
    base = get_settings()
    mongodb_uri = require_env("MONGODB_URI")
    require_env("VOYAGE_API_KEY")

    run_id = uuid.uuid4().hex[:8]
    database_name = os.getenv("HYBRIDRAG_TEST_DB", f"hybridrag_live_gate_{run_id}")
    workspace = os.getenv("HYBRIDRAG_TEST_WORKSPACE", f"livegate_{run_id}")

    updates: dict[str, Any] = {
        "mongodb_uri": SecretStr(mongodb_uri),
        "mongodb_database": database_name,
        "mongodb_workspace": workspace,
        "voyage_api_key": SecretStr(require_env("VOYAGE_API_KEY")),
        "enable_llm": True,
    }

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        updates.update(
            {
                "llm_provider": "openai",
                "openai_api_key": SecretStr(openai_api_key),
                "openai_model": os.getenv("OPENAI_MODEL", base.openai_model),
                "openai_base_url": os.getenv("OPENAI_BASE_URL", base.openai_base_url),
                "openai_extra_headers": os.getenv(
                    "OPENAI_EXTRA_HEADERS", base.openai_extra_headers
                ),
            }
        )
    else:
        raise RuntimeError("Missing real LLM credentials. Set OPENAI_API_KEY.")

    settings = base.model_copy(update=updates)
    return LiveContext(
        mongodb_uri=mongodb_uri,
        database_name=database_name,
        workspace=workspace,
        settings=settings,
    )


async def reset_runtime_state() -> None:
    await ClientManager.reset()
    finalize_share_data()
    close_shared_client()


async def drop_database(ctx: LiveContext) -> None:
    client = AsyncMongoClient(ctx.mongodb_uri)
    try:
        await client.drop_database(ctx.database_name)
    finally:
        await client.close()


async def create_rag(ctx: LiveContext):
    await reset_runtime_state()
    return await create_hybridrag(
        settings=ctx.settings,
        working_dir=f"/tmp/{ctx.database_name}_{ctx.workspace}",
    )


async def test_mongodb_connection(ctx: LiveContext) -> None:
    start = time.perf_counter()
    client = AsyncMongoClient(ctx.mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        result = await client.admin.command("ping")
        passed = result.get("ok") == 1.0
        record(
            "MongoDB Connection",
            passed,
            f"Connected to {ctx.mongodb_uri} / db={ctx.database_name}",
            start,
        )
    finally:
        await client.close()


async def test_voyage_embeddings() -> list[float] | None:
    start = time.perf_counter()
    try:
        import voyageai

        client = voyageai.Client(api_key=require_env("VOYAGE_API_KEY"))
        result = client.embed(
            texts=["What is MongoDB Atlas Vector Search?"], model="voyage-4-large"
        )
        embedding = result.embeddings[0]
        passed = len(embedding) == 1024
        record(
            "Voyage Embeddings",
            passed,
            f"Generated live embedding with dimension {len(embedding)}",
            start,
        )
        return embedding
    except Exception as exc:
        record("Voyage Embeddings", False, str(exc), start)
        return None


async def seed_corpus(rag) -> list[str]:
    start = time.perf_counter()
    ids = [doc["id"] for doc in LIVE_DOCS]
    for doc in LIVE_DOCS:
        await rag.insert(doc["content"], ids=[doc["id"]])
    record(
        "Corpus Seeding",
        True,
        f"Inserted {len(LIVE_DOCS)} realistic documents with explicit IDs",
        start,
    )
    return ids


async def test_engine_status(rag) -> None:
    start = time.perf_counter()
    status = await rag.get_status()
    passed = bool(status.get("initialized"))
    record(
        "Engine Initialization",
        passed,
        f"LLM={status.get('llm_model')} Embedding={status.get('embedding_model')}",
        start,
        status=status,
    )


async def test_knowledge_base_stats(rag) -> None:
    start = time.perf_counter()
    stats = await rag.get_knowledge_base_stats()
    failed_docs = stats["documents"]["by_status"].get("failed", 0)
    passed = (
        stats["documents"]["total"] >= len(LIVE_DOCS)
        and failed_docs == 0
        and stats["entities"] > 0
        and stats["relationships"] > 0
        and stats["chunks"] > 0
    )
    record(
        "Knowledge Base Stats",
        passed,
        (
            f"docs={stats['documents']['total']} failed_docs={failed_docs} chunks={stats['chunks']} "
            f"entities={stats['entities']} relationships={stats['relationships']}"
        ),
        start,
        stats=stats,
    )


async def test_direct_vector_search(
    ctx: LiveContext, query_vector: list[float]
) -> None:
    start = time.perf_counter()
    client = AsyncMongoClient(ctx.mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        db = client[ctx.database_name]
        candidate_collections = []
        workspaced_collection = get_collection_name(ctx.workspace, "chunks")
        if workspaced_collection:
            candidate_collections.append(workspaced_collection)
        candidate_collections.append("chunks")

        selected_collection = None
        vector_index_name = None
        selected_count = 0
        for collection_name in dict.fromkeys(candidate_collections):
            if collection_name not in await db.list_collection_names():
                continue

            indexes_cursor = await db[collection_name].list_search_indexes()
            indexes = await indexes_cursor.to_list(length=20)
            doc_count = await db[collection_name].count_documents({})
            index_name = next(
                (
                    index["name"]
                    for index in indexes
                    if index.get("type") == "vectorSearch"
                    or str(index.get("name", "")).startswith("vector_knn_index")
                ),
                None,
            )

            if doc_count > 0 and index_name:
                selected_collection = collection_name
                vector_index_name = index_name
                selected_count = doc_count
                break

        if not selected_collection or not vector_index_name:
            raise RuntimeError(
                "Could not find a populated chunks collection with a vector search index"
            )

        pipeline = [
            {
                "$vectorSearch": {
                    "index": vector_index_name,
                    "path": "vector",
                    "queryVector": query_vector,
                    "numCandidates": 50,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0,
                }
            },
        ]
        cursor = await db[selected_collection].aggregate(pipeline)
        results = await cursor.to_list(length=5)
        passed = bool(results) and all("score" in row for row in results)
        top_score = round(results[0]["score"], 4) if results else None
        record(
            "Direct Vector Search",
            passed,
            (
                f"collection={selected_collection} docs={selected_count} "
                f"index={vector_index_name} results={len(results)} top_score={top_score}"
            ),
            start,
        )
    except Exception as exc:
        record("Direct Vector Search", False, str(exc), start)
    finally:
        await client.close()


def _requires_native_hybrid(ctx: LiveContext) -> bool:
    return "localhost:27018" in ctx.mongodb_uri or "127.0.0.1:27018" in ctx.mongodb_uri


async def test_query_modes(rag, ctx: LiveContext) -> None:
    for case in MODE_CASES:
        start = time.perf_counter()
        result = await rag.query_data(
            query=case["query"],
            mode=case["mode"],
            top_k=5,
        )
        context = result.get("context", "")
        lowered = context.lower()
        keyword_hit = any(keyword in lowered for keyword in case["expects"])
        data = result.get("data", {})
        metadata = result.get("metadata", {})
        non_empty_data = any(
            len(data.get(key, [])) > 0
            for key in ("chunks", "entities", "relationships")
        )
        passed = (
            len(context) >= case["minimum_context"] and keyword_hit and non_empty_data
        )
        if _requires_native_hybrid(ctx) and case["mode"] in {"naive", "hybrid", "mix"}:
            passed = passed and metadata.get("fallback_used") is False
        record(
            case["name"],
            passed,
            (
                f"mode={case['mode']} context_len={len(context)} "
                f"chunks={len(data.get('chunks', []))} "
                f"entities={len(data.get('entities', []))} "
                f"relationships={len(data.get('relationships', []))} "
                f"fallback_used={metadata.get('fallback_used')}"
            ),
            start,
        )


async def test_query_with_sources(rag, ctx: LiveContext) -> None:
    start = time.perf_counter()
    result = await rag.query_with_sources(
        query="How does hybrid search combine lexical and semantic relevance?",
        mode="mix",
        top_k=5,
    )
    metadata = result.get("metadata", {})
    passed = (
        len(result.get("answer", "")) > 80
        and len(result.get("context", "")) > 80
        and len(result.get("references", [])) > 0
    )
    if _requires_native_hybrid(ctx):
        passed = passed and metadata.get("fallback_used") is False
    record(
        "Query With Sources",
        passed,
        (
            f"answer_len={len(result.get('answer', ''))} "
            f"context_len={len(result.get('context', ''))} "
            f"references={len(result.get('references', []))} "
            f"fallback_used={metadata.get('fallback_used')}"
        ),
        start,
    )


async def test_stream_query(rag, ctx: LiveContext) -> None:
    start = time.perf_counter()
    result = await rag.stream_query(
        query="What should a production RAG boilerplate expose?",
        mode="mix",
        top_k=5,
        include_context=True,
        include_references=True,
    )
    chunks: list[str] = []
    async for chunk in result["response_iterator"]:
        if chunk:
            chunks.append(chunk)

    answer = "".join(chunks)
    metadata = result.get("metadata", {})
    passed = (
        len(answer) > 80
        and len(result.get("context", "")) > 80
        and len(result.get("references", [])) > 0
    )
    if _requires_native_hybrid(ctx):
        passed = passed and metadata.get("fallback_used") is False
    record(
        "Streaming Query",
        passed,
        (
            f"answer_len={len(answer)} "
            f"context_len={len(result.get('context', ''))} "
            f"references={len(result.get('references', []))} "
            f"fallback_used={metadata.get('fallback_used')}"
        ),
        start,
    )


async def test_query_with_memory(rag) -> None:
    session_id = f"live-gate-{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()
    first = await rag.query_with_memory(
        query="What does the corpus say about conversation memory?",
        session_id=session_id,
        mode="mix",
        top_k=5,
    )
    second = await rag.query_with_memory(
        query="Explain that in the context of multi-turn assistants.",
        session_id=session_id,
        mode="mix",
        top_k=5,
    )
    history = await rag.get_conversation_history(session_id)
    passed = (
        len(first.get("answer", "")) > 50
        and len(second.get("answer", "")) > 50
        and second.get("history_used", 0) >= 1
        and len(history) >= 4
    )
    record(
        "Query With Memory",
        passed,
        (
            f"first_len={len(first.get('answer', ''))} "
            f"second_len={len(second.get('answer', ''))} "
            f"history_used={second.get('history_used', 0)} "
            f"stored_messages={len(history)}"
        ),
        start,
    )


async def test_delete_document(rag, seeded_ids: list[str]) -> None:
    start = time.perf_counter()
    before = await rag.get_knowledge_base_stats()
    await rag.delete_document(seeded_ids[0])
    after = await rag.get_knowledge_base_stats()
    passed = after["documents"]["total"] <= before["documents"]["total"] - 1
    record(
        "Delete Document",
        passed,
        (
            f"before_docs={before['documents']['total']} "
            f"after_docs={after['documents']['total']}"
        ),
        start,
    )


async def test_realistic_stress_queries(rag) -> None:
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for query, mode, with_sources in STRESS_CASES:
        q_start = time.perf_counter()
        if with_sources:
            output = await rag.query_with_sources(query=query, mode=mode, top_k=5)
            ok = (
                len(output.get("answer", "")) > 80
                and len(output.get("context", "")) > 80
                and len(output.get("references", [])) > 0
            )
            results.append(
                {
                    "query": query,
                    "mode": mode,
                    "with_sources": True,
                    "ok": ok,
                    "latency_s": round(time.perf_counter() - q_start, 2),
                    "answer_len": len(output.get("answer", "")),
                    "references": len(output.get("references", [])),
                }
            )
        else:
            output = await rag.query(query=query, mode=mode, top_k=5)
            ok = isinstance(output, str) and len(output) > 80
            results.append(
                {
                    "query": query,
                    "mode": mode,
                    "with_sources": False,
                    "ok": ok,
                    "latency_s": round(time.perf_counter() - q_start, 2),
                    "answer_len": len(output) if isinstance(output, str) else 0,
                }
            )

    passed_queries = sum(1 for item in results if item["ok"])
    average_latency = round(mean(item["latency_s"] for item in results), 2)
    passed = passed_queries == len(results)
    record(
        "Realistic Stress Queries",
        passed,
        f"passed={passed_queries}/{len(results)} avg_latency_s={average_latency}",
        start,
        results=results,
    )


async def main() -> None:
    overall_start = time.perf_counter()
    ctx = build_live_settings()
    print("=" * 72)
    print("HybridRAG Live Release Gate")
    print("=" * 72)
    print(f"MongoDB URI: {ctx.mongodb_uri}")
    print(f"Database: {ctx.database_name}")
    print(f"Workspace: {ctx.workspace!r}")
    print(f"LLM Provider: {ctx.settings.llm_provider}")
    print(
        f"LLM Model: {ctx.settings.openai_model if ctx.settings.llm_provider == 'openai' else ctx.settings.gemini_model}"
    )
    print(f"Embedding Model: {ctx.settings.voyage_embedding_model}")
    print("=" * 72)

    await drop_database(ctx)
    rag = None
    try:
        await test_mongodb_connection(ctx)
        embedding = await test_voyage_embeddings()
        rag = await create_rag(ctx)
        await test_engine_status(rag)
        seeded_ids = await seed_corpus(rag)
        await test_knowledge_base_stats(rag)
        if embedding is not None:
            await test_direct_vector_search(ctx, embedding)
        await test_query_modes(rag, ctx)
        await test_query_with_sources(rag, ctx)
        await test_stream_query(rag, ctx)
        await test_query_with_memory(rag)
        await test_realistic_stress_queries(rag)
        await test_delete_document(rag, seeded_ids)
    finally:
        await reset_runtime_state()
        await drop_database(ctx)

    passed = sum(1 for record_ in RESULTS if record_.passed)
    failed = len(RESULTS) - passed
    summary = {
        "database": ctx.database_name,
        "workspace": ctx.workspace,
        "passed": passed,
        "failed": failed,
        "duration_s": round(time.perf_counter() - overall_start, 2),
        "tests": [asdict(record_) for record_ in RESULTS],
    }
    print(json.dumps(summary, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
HybridRAG — 'See MongoDB value in 60 seconds' demo (NO API keys required).

This showcase runs the REAL MongoDB 8.2+ native hybrid-search pipeline against
a local `mongodb/mongodb-atlas-local:preview` container — no Voyage key, no LLM
key, no signup. It demonstrates the MongoDB value that HybridRAG is built on:

    $rankFusion  →  merges  $vectorSearch  +  $search (lexical)  via RRF
    $graphLookup →  knowledge-graph traversal from a seed entity

Embeddings are Voyage AI's job in production. Here we use small sample vectors
(labeled) so you can see the mechanics with zero setup. Swap in real Voyage
embeddings + an LLM key for full generative RAG (see `make demo-full`).

Usage:
    make demo            # starts local MongoDB + runs this script
    python scripts/demo.py --uri mongodb://localhost:27018/?directConnection=true

Requirements: a running atlas-local container on localhost:27018
             (`docker compose -f docker/docker-compose.local.yml up -d`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from pymongo import MongoClient

# Reuse the repo's TLS helper so the demo also works against Atlas
# (mongodb+srv://) on macOS + python.org Python without SSL errors.
from hybridrag.core.mongodb_client import _tls_kwargs

DEFAULT_URI = "mongodb://localhost:27018/?directConnection=true"
DB_NAME = "hybridrag_demo"
CHUNKS = "demo_chunks"
EDGES = "demo_graph_edges"
VECTOR_DIM = 8  # small for readable output; production uses 1024 (voyage-4-large)


# --- Sample corpus -----------------------------------------------------------
# content is real text; `vector` is a hand-crafted sample embedding chosen so the
# "hybrid search" document is nearest to the query vector (cosine). In production
# these vectors come from Voyage AI.
DOCS: list[dict[str, Any]] = [
    {
        "_id": "d1",
        "content": "MongoDB Atlas is a managed cloud database with built-in vector search via Atlas Search.",
        "topic": "atlas",
        "vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "_id": "d2",
        "content": "Hybrid search combines vector similarity with keyword search using $rankFusion and Reciprocal Rank Fusion.",
        "topic": "hybrid",
        "vector": [0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "_id": "d3",
        "content": "Voyage AI provides embedding models and rerankers optimized for retrieval.",
        "topic": "voyage",
        "vector": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "_id": "d4",
        "content": "Knowledge graphs capture entity relationships; MongoDB $graphLookup traverses them in one pipeline.",
        "topic": "graph",
        "vector": [0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0],
    },
    {
        "_id": "d5",
        "content": "$vectorSearch returns nearest neighbors by cosine similarity over stored embeddings.",
        "topic": "vector",
        "vector": [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    },
    {
        "_id": "d6",
        "content": "Lexical search uses BM25 scoring in a $search stage with fuzzy and phrase matching.",
        "topic": "lexical",
        "vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
]

# Query whose sample vector is closest to the "hybrid search" doc (d2).
QUERY_TEXT = "How does hybrid search work?"
QUERY_VECTOR = [0.0, 0.95, 0.05, 0.0, 0.0, 0.0, 0.05, 0.0]

# --- Knowledge graph edges (source_node -> target_node, relationship) ---------
EDGES_DATA: list[dict[str, Any]] = [
    {
        "_id": "e1",
        "source_node_id": "hybrid",
        "target_node_id": "vector",
        "relationship_type": "combines",
    },
    {
        "_id": "e2",
        "source_node_id": "hybrid",
        "target_node_id": "lexical",
        "relationship_type": "combines",
    },
    {
        "_id": "e3",
        "source_node_id": "hybrid",
        "target_node_id": "atlas",
        "relationship_type": "runs_on",
    },
    {
        "_id": "e4",
        "source_node_id": "atlas",
        "target_node_id": "vector",
        "relationship_type": "provides",
    },
    {
        "_id": "e5",
        "source_node_id": "vector",
        "target_node_id": "voyage",
        "relationship_type": "powered_by",
    },
]


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def wait_for_index_ready(coll: Any, index_name: str, timeout_s: int = 30) -> bool:
    """Atlas Search indexes build asynchronously; wait until queryable."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            for ix in coll.list_search_indexes():
                if ix.get("name") == index_name:
                    if ix.get("status", ix.get("queryable")) in ("READY", True, "true"):
                        return True
                    # atlas-local reports queryable=True once ready
                    if ix.get("queryable") is True:
                        return True
        except Exception:
            pass
        time.sleep(1)
    return False


def reset(client: MongoClient) -> None:
    db = client[DB_NAME]
    db[CHUNKS].drop()
    db[EDGES].drop()


def seed(client: MongoClient) -> None:
    db = client[DB_NAME]
    db[CHUNKS].insert_many(DOCS)
    db[EDGES].insert_many(EDGES_DATA)
    print(
        f"Seeded {len(DOCS)} chunks + {len(EDGES_DATA)} graph edges into '{DB_NAME}'."
    )


def create_indexes(client: MongoClient) -> None:
    db = client[DB_NAME]
    chunks = db[CHUNKS]

    chunks.create_search_index(
        {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "vector",
                        "numDimensions": VECTOR_DIM,
                        "similarity": "cosine",
                    }
                ]
            },
        }
    )

    chunks.create_search_index(
        {
            "name": "text_index",
            "type": "search",
            "definition": {
                "mappings": {"dynamic": True},
            },
        }
    )

    ready_v = wait_for_index_ready(chunks, "vector_index")
    ready_t = wait_for_index_ready(chunks, "text_index")
    print(f"vector_index ready={ready_v} | text_index ready={ready_t}")
    if not (ready_v and ready_t):
        raise RuntimeError("Search indexes did not become ready in time.")


def run_vector_search(db: Any) -> list[dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector",
                "queryVector": QUERY_VECTOR,
                "numCandidates": 10,
                "limit": 3,
            }
        },
        {
            "$project": {
                "_id": 1,
                "content": 1,
                "topic": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(db[CHUNKS].aggregate(pipeline))


def run_lexical_search(db: Any) -> list[dict[str, Any]]:
    pipeline = [
        {
            "$search": {
                "index": "text_index",
                "text": {"path": "content", "query": "hybrid search rank fusion"},
            }
        },
        {"$limit": 3},
        {
            "$project": {
                "_id": 1,
                "content": 1,
                "topic": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
    return list(db[CHUNKS].aggregate(pipeline))


def run_rank_fusion(db: Any) -> list[dict[str, Any]]:
    """The MongoDB 8.2 native hybrid search: $rankFusion of vector + lexical.

    Results are returned in fused rank order. (The `rankFusionScore` $meta is not
    exposed on all atlas-local:preview builds, so we rank by position — the fusion
    order is the demonstration of value.)
    """
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": "vector_index",
                                    "path": "vector",
                                    "queryVector": QUERY_VECTOR,
                                    "numCandidates": 10,
                                    "limit": 3,
                                }
                            },
                        ],
                        "text": [
                            {
                                "$search": {
                                    "index": "text_index",
                                    "text": {
                                        "path": "content",
                                        "query": "hybrid search rank fusion",
                                    },
                                }
                            },
                            {"$limit": 3},
                        ],
                    }
                },
                "combination": {"weights": {"vector": 0.6, "text": 0.4}},
                "scoreDetails": True,
            }
        },
        {"$project": {"_id": 1, "content": 1, "topic": 1}},
    ]
    return list(db[CHUNKS].aggregate(pipeline))


def run_graph_lookup(db: Any) -> list[dict[str, Any]]:
    """Knowledge-graph traversal from the 'hybrid' entity in a single pipeline."""
    pipeline = [
        {
            "$graphLookup": {
                "from": EDGES,
                "startWith": "hybrid",
                "connectFromField": "target_node_id",
                "connectToField": "source_node_id",
                "as": "graph",
                "maxDepth": 2,
                "depthField": "depth",
            }
        },
        {"$limit": 1},
        {
            "$project": {
                "graph": {"relationship_type": 1, "target_node_id": 1, "depth": 1}
            }
        },
    ]
    # graphLookup starts from a documents collection; run against chunks but
    # only use the traversal result.
    res = list(db[CHUNKS].aggregate(pipeline))
    return res[0].get("graph", []) if res else []


def show(
    name: str,
    rows: list[dict[str, Any]],
    pipeline: list[dict[str, Any]] | None = None,
    lesson: str | None = None,
) -> None:
    banner(name)
    if lesson:
        print(f"\n💡 What this shows: {lesson}")
    if pipeline is not None:
        print("\nMongoDB aggregation pipeline:")
        print(json.dumps(pipeline, indent=2))
    print("\nResults:")
    if not rows:
        print("  (no rows)")
        return
    for r in rows:
        score = r.get("score")
        score_s = f"  score={score:.4f}" if isinstance(score, int | float) else ""
        print(f"  - [{r.get('_id')}] ({r.get('topic')}){score_s}")
        print(f"      {r.get('content', '')[:90]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HybridRAG no-keys MongoDB value demo."
    )
    parser.add_argument(
        "--uri", default=DEFAULT_URI, help="MongoDB URI (default: local atlas-local)"
    )
    args = parser.parse_args()

    banner("HybridRAG — See MongoDB value in 60s (no API keys)")
    print(f"Connecting to: {args.uri}")
    print(f'Query: "{QUERY_TEXT}"')
    print("Sample vectors used (replace with Voyage AI embeddings in production).")

    client = MongoClient(
        args.uri, serverSelectionTimeoutMS=5000, **_tls_kwargs(args.uri)
    )
    try:
        ver = client.server_info()["version"]
        print(f"Connected to MongoDB {ver}")
    except Exception as exc:
        print(
            f"\nCould not connect to MongoDB at {args.uri}.\n"
            f"Options:\n"
            f"  - Local (Docker):  docker compose -f docker/docker-compose.local.yml up -d\n"
            f"  - Atlas (no Docker): set MONGODB_URI in .env to your Atlas connection string\n\n"
            f"Error: {exc}"
        )
        return 1

    reset(client)
    seed(client)
    create_indexes(client)

    db = client[DB_NAME]

    show(
        "$vectorSearch — semantic nearest neighbors",
        run_vector_search(db),
        [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "vector",
                    "queryVector": "<query_vector>",
                    "numCandidates": 10,
                    "limit": 3,
                }
            }
        ],
        lesson="MongoDB finds the chunks whose vectors are closest to the query vector "
        "(cosine similarity). Notice the 'hybrid' chunk ranks highest — its sample "
        "vector was nearest to the query. In production these vectors come from "
        "Voyage AI; here they are hand-crafted so you can see the mechanics with no API key.",
    )

    show(
        "$search — lexical / BM25 text matching",
        run_lexical_search(db),
        [
            {
                "$search": {
                    "index": "text_index",
                    "text": {"path": "content", "query": "hybrid search rank fusion"},
                }
            }
        ],
        lesson="MongoDB also does classic keyword search with BM25 scoring. Notice it can "
        "surface documents that vector search might miss — pure lexical relevance, "
        "no embeddings involved. This is the other half of hybrid search.",
    )

    show(
        "$rankFusion — MongoDB 8.2 native hybrid search (vector + lexical via RRF)",
        run_rank_fusion(db),
        lesson="This is the headline. $rankFusion runs BOTH pipelines (vector + text) and "
        "merges them with Reciprocal Rank Fusion in a SINGLE database operation — "
        "no app-side merging, no Pinecone+Postgres glue. Notice the result order "
        "blends semantic and lexical relevance. One MongoDB aggregation, one "
        "atomic result set. This is what replaces the fragmented Pinecone+Neo4j+"
        "Redis stack with one database.",
    )

    banner("$graphLookup — knowledge-graph traversal from 'hybrid'")
    print(
        "\n💡 What this shows: MongoDB traverses entity relationships in ONE aggregation "
        "stage — no Neo4j, no separate graph database. Starting from the 'hybrid' "
        "entity, it walks the edges collection up to 2 hops. Notice it discovers that "
        "'hybrid' combines 'vector' and 'lexical', runs on 'atlas', which provides "
        "'vector' (powered by 'voyage') — a multi-hop traversal, all in MongoDB."
    )
    graph = run_graph_lookup(db)
    print("\nMongoDB aggregation pipeline:")
    print(
        json.dumps(
            [
                {
                    "$graphLookup": {
                        "from": EDGES,
                        "startWith": "hybrid",
                        "connectFromField": "target_node_id",
                        "connectToField": "source_node_id",
                        "as": "graph",
                        "maxDepth": 2,
                        "depthField": "depth",
                    }
                }
            ],
            indent=2,
        )
    )
    print("\nTraversal results (entity relationships reachable from 'hybrid'):")
    if not graph:
        print("  (no edges traversed)")
    for edge in graph:
        print(
            f"  - depth {edge.get('depth')}: hybrid --{edge.get('relationship_type')}--> {edge.get('target_node_id')}"
        )

    banner(
        "That's MongoDB-native hybrid search — one database, atomic, no Pinecone/Neo4j/Redis."
    )
    print("Next: `make demo-full` (bring Voyage + LLM keys) for full generative RAG.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

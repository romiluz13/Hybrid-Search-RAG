# Architecture

HybridRAG is a layered Python library with MongoDB Atlas as the single data store. The architecture has four tiers: a public API surface (SDK, REST, CLI, UI), an enhancement layer (hybrid search, filters, entity boosting, memory), an engine core (query execution, LLM orchestration, storage), and the MongoDB storage implementation.

## System diagram

```mermaid
graph TD
    subgraph Interfaces
        SDK["Python SDK<br/>HybridRAG class"]
        REST["FastAPI REST API"]
        CLI["Typer/Rich CLI"]
        UI["Chainlit Chat UI"]
    end

    subgraph Enhancements
        HS["Hybrid Search<br/>$rankFusion / $scoreFusion"]
        FL["Filter System<br/>Vector / Atlas / Lexical"]
        EB["Entity Boosting"]
        MM["Mix Mode Search"]
        IE["Implicit Expansion"]
        CM["Conversation Memory"]
    end

    subgraph Engine Core
        RAG["core/rag.py<br/>Query orchestration"]
        OPS["engine/operate.py<br/>Query execution pipeline"]
        LLM["LLM Providers<br/>OpenAI / Claude / Gemini"]
        SEC["Security<br/>Tenant isolation"]
    end

    subgraph Storage
        MONGO["MongoDB Atlas<br/>Vector + Graph + KV"]
        VS["$vectorSearch"]
        SRCH["$search"]
        GF["$graphLookup"]
        RF["$rankFusion / $scoreFusion"]
    end

    SDK --> RAG
    REST --> RAG
    CLI --> RAG
    UI --> RAG

    RAG --> OPS
    RAG --> HS
    RAG --> CM
    OPS --> HS
    OPS --> FL
    OPS --> EB
    OPS --> MM
    OPS --> IE
    OPS --> SEC

    HS --> MONGO
    FL --> MONGO
    OPS --> MONGO

    MONGO --> VS
    MONGO --> SRCH
    MONGO --> GF
    MONGO --> RF

    LLM --> RAG
```

## Query flow

When a user calls `rag.query()`, the request flows through these stages:

```mermaid
sequenceDiagram
    participant User
    participant RAG as core/rag.py
    participant Ops as engine/operate.py
    participant HS as Hybrid Search
    participant Mongo as MongoDB Atlas
    participant LLM as LLM Provider

    User->>RAG: query("How does X work?", mode="mix")
    RAG->>Ops: execute query with QueryParam
    Ops->>HS: build fusion pipeline (vector + text)
    HS->>Mongo: $rankFusion / $scoreFusion aggregation
    Mongo-->>HS: fused results with scores
    Ops->>Mongo: $graphLookup for KG entities
    Mongo-->>Ops: entity relationships
    Ops->>Ops: entity boosting + reranking
    Ops->>LLM: generate answer with context
    LLM-->>Ops: streamed response
    Ops-->>RAG: answer + context + references
    RAG-->>User: response
```

## Data flow

All data lives in MongoDB. A single document contains the text chunk, vector embedding, metadata, and graph entity references. This makes updates atomic: vector + metadata + graph in one transaction, never inconsistent.

```mermaid
graph LR
    subgraph Ingestion
        DOC["Document"]
        PROC["Processor"]
        CHUNK["Chunker"]
        EMB["Voyage Embedder"]
    end

    subgraph MongoDB Collections
        CHUNKS["chunks collection<br/>text + vector + metadata"]
        ENT["entities collection<br/>name + type + description"]
        REL["relationships collection<br/>src + tgt + description"]
        DOCS["documents collection<br/>metadata + status"]
    end

    DOC --> PROC --> CHUNK --> EMB
    EMB -->|insert| CHUNKS
    PROC -->|extract| ENT
    PROC -->|extract| REL
    PROC -->|metadata| DOCS
```

## Key architectural decisions

- **Single database**: MongoDB Atlas handles vectors, graphs, key-value, and metadata. No Pinecone, Neo4j, or Redis. See [design decisions](../background/design-decisions.md) for the full ADR list.
- **MongoDB 8.2+ features**: The library uses `$rankFusion`, `$scoreFusion`, and `$search.vectorSearch` with lexical prefilters. Capability is probed by execution, not by version number.
- **Blessed stack**: One supported path (MongoDB + Voyage + OpenAI-compatible LLM) is exercised by the release gate. Unsupported capabilities fail explicitly.
- **Async-first**: All public APIs are async. The storage layer uses Motor (async MongoDB driver). LLM calls are async with streaming support.
- **Filter translation**: A public `FilterConfig` translates to three backend-specific MongoDB syntaxes: MQL for `$vectorSearch`, Atlas Search compound for `$search`, and lexical prefilters for `$search.vectorSearch`. See [Filters](../features/filters.md).

## Language breakdown

The codebase is 100% Python (61,321 lines across 110 source files). It targets Python 3.11+ with full type hints. See [by the numbers](../by-the-numbers.md) for detailed statistics.

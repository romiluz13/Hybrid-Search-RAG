"""Tests for Phase 3B: Query & Pipeline Performance (16 MEDIUM findings).

M11: to_list(length=None) unbounded -> must use _DEFAULT_CURSOR_LIMIT
M12: to_list(length=10000) already acceptable (no test needed)
M13: No projection on message fetch -> add projection to exclude large fields
M14: Redundant single-field index -> remove it
M16: Missing $limit in vector pipeline of $rankFusion
M20: $graphLookup already fixed in Phase 2B (verify)
M21: $in with regex array -> use pre-lowercased string equality
M22: Case-sensitive comparison already fixed in Phase 2B (verify)
M23: Rule priority conflicts in query optimizer -> normalize weights
M24: _fallback_regex_search COLLSCAN -> use anchored regex
M25: Workspace namespace logic copy-pasted -> extract helper
M26: Sequential storage initialization -> asyncio.gather
M27: Migration loads all nodes+edges into memory -> batch processing
M28: count_documents({}) before search_labels -> estimated_document_count
M32: Socket timeout already set at client level (no test needed)
M48: shared_storage uses time.time as type annotation -> float

Backing skill: mongodb-query-optimizer
- ESR (Equality-Sort-Range) for compound indexes
- NEVER use $regex for search -- use Atlas Search instead
- Filter early in pipelines with $match
- Use covered queries when possible
- to_list(length=None) is dangerous -- always use explicit limits
"""

import inspect
import re

# ---- M11: to_list(length=None) unbounded in 17+ locations ----


class TestM11BoundedCursorLimits:
    """All to_list calls in mongo_impl.py must use explicit length limits."""

    def test_default_cursor_limit_constant_exists(self) -> None:
        """Module must define _DEFAULT_CURSOR_LIMIT constant."""
        from hybridrag.engine.kg import mongo_impl

        assert hasattr(mongo_impl, "_DEFAULT_CURSOR_LIMIT"), (
            "mongo_impl must define _DEFAULT_CURSOR_LIMIT constant"
        )
        assert isinstance(mongo_impl._DEFAULT_CURSOR_LIMIT, int)
        assert mongo_impl._DEFAULT_CURSOR_LIMIT == 10000

    def test_no_to_list_length_none_in_mongo_impl(self) -> None:
        """No to_list(length=None) calls should remain in mongo_impl.py."""
        import hybridrag.engine.kg.mongo_impl as mod

        source = inspect.getsource(mod)
        matches = re.findall(r"to_list\(length=None\)", source)
        assert len(matches) == 0, (
            f"Found {len(matches)} to_list(length=None) calls in mongo_impl.py. "
            "All must use to_list(length=_DEFAULT_CURSOR_LIMIT) per query-optimizer skill: "
            "'to_list(length=None) is dangerous -- always use explicit limits'"
        )


# ---- M13: No projection on message fetch ----


class TestM13MessageFetchProjection:
    """Message fetch must use projection to exclude large fields."""

    def test_get_messages_from_collection_uses_projection(self) -> None:
        """_get_messages_from_collection should exclude embedding/vector fields via projection."""
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory._get_messages_from_collection)
        # Should have a projection that excludes embedding fields
        assert "embedding" in source and "0" in source, (
            "_get_messages_from_collection must use projection to exclude large embedding fields "
            "per query-optimizer skill: 'Use covered queries when possible'"
        )


# ---- M14: Redundant single-field index ----


class TestM14RedundantIndex:
    """Single-field index on session_id_key should be removed since compound indexes cover it."""

    def test_messages_collection_no_single_field_session_id_index(self) -> None:
        """Messages collection should NOT create a single-field session_id index.

        The compound indexes (session_id + timestamp) and (session_id + message_index)
        already cover the prefix, so a standalone session_id index is redundant.
        Per core-indexing-principles: compound index covers prefix queries.
        """
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory.initialize)

        # Count how many create_index calls use session_id_key alone (not in a tuple/list)
        # The compound indexes use [(session_id_key, 1), ...] - that's fine
        # The redundant one would be: create_index(self._session_id_key) by itself on messages
        lines = source.split("\n")
        redundant_found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for: messages_collection.create_index(self._session_id_key)
            # But NOT: messages_collection.create_index([(self._session_id_key, ...
            if (
                "_messages_collection.create_index" in stripped
                and "session_id_key" in stripped
                and "[(" not in stripped  # Not a compound index
                and "unique" not in stripped  # Not the sessions unique index
            ):
                redundant_found = True
                break

        assert not redundant_found, (
            "Messages collection should NOT have a single-field index on session_id_key. "
            "The compound indexes (session_id + timestamp) and (session_id + message_index) "
            "already cover this prefix per core-indexing-principles."
        )


# ---- M16: Missing $limit in vector pipeline of $rankFusion ----


class TestM16VectorPipelineLimit:
    """Vector pipeline inside $rankFusion must have an explicit $limit stage."""

    def test_rank_fusion_vector_pipeline_has_limit(self) -> None:
        """The vector pipeline passed to $rankFusion must include $limit."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_rank_fusion,
        )

        source = inspect.getsource(hybrid_search_with_rank_fusion)
        # The vector pipeline should append a $limit stage before being passed to $rankFusion
        # After building vector_pipeline, there should be a $limit appended
        assert "vector_pipeline.append" in source and "$limit" in source, (
            "Vector pipeline must have an explicit $limit stage appended before "
            "being passed to $rankFusion. Per mongodb-search-and-ai skill: "
            "'$search does not auto-limit results -- always add $limit inside the input pipeline'"
        )


# ---- M20: Already fixed in Phase 2B (H14) -- verify ----


class TestM20GraphLookupLimitVerification:
    """Verify $limit:100 was added after $match in Phase 2B."""

    def test_graph_lookup_pipeline_has_limit_100_after_match(self) -> None:
        """$graphLookup pipeline must have $limit:100 after $match (done in H14)."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("test_entity", config)

        # Find $match and next stage
        match_idx = None
        for i, stage in enumerate(pipeline):
            if "$match" in stage:
                match_idx = i
                break

        assert match_idx is not None
        limit_stage = pipeline[match_idx + 1]
        assert "$limit" in limit_stage, "Must have $limit after $match"
        assert limit_stage["$limit"] == 100, "Limit should be 100"


# ---- M21: $in with regex array O(n*m) ----


class TestM21NoRegexInEntityChunkLookup:
    """find_entity_chunks must NOT use regex in $in arrays."""

    def test_entity_chunk_lookup_no_regex(self) -> None:
        """Entity chunk lookup should use pre-lowercased string equality, not regex."""
        from hybridrag.enhancements.graph_search import get_chunks_for_entities

        source = inspect.getsource(get_chunks_for_entities)
        # Should NOT use re.compile in $in arrays
        assert "re.compile" not in source, (
            "get_chunks_for_entities must NOT use re.compile patterns in $in arrays. "
            "O(n*m) regex matching is expensive. Use pre-lowercased string equality "
            "per query-optimizer skill: 'NEVER use $regex for search'"
        )
        assert "re.IGNORECASE" not in source, (
            "get_chunks_for_entities must NOT use case-insensitive regex. "
            "Use $toLower in $filter or pre-normalized names instead."
        )


# ---- M22: Already fixed in Phase 2B (H13) -- verify ----


class TestM22CaseSensitiveComparisonVerification:
    """Verify case-sensitive comparison was fixed in Phase 2B."""

    def test_graph_lookup_uses_lowercased_entity(self) -> None:
        """build_graph_lookup_pipeline must use lowercased entity name."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("MongoDB", config)
        match_stage = pipeline[0]["$match"]
        conditions = match_stage["$or"]
        source_val = conditions[0].get(config.source_field)
        assert source_val == "mongodb", "Entity name must be lowercased"


# ---- M23: Rule priority conflicts in query optimizer ----


class TestM23WeightNormalization:
    """Query optimizer must normalize final weights to sum to 1.0."""

    def test_weights_always_sum_to_one(self) -> None:
        """After all rules applied, weights must sum to 1.0."""
        from hybridrag.enhancements.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()

        # Test various scenarios that trigger different rules
        test_queries = [
            # Short query (Rule 1)
            "hello",
            # Long query with entities (Rule 2 + Rule 3)
            "how do I configure the MongoDB Atlas Search index for my application with vector search and embeddings",
            # Summary-type query (Rule 5)
            "summarize everything about database optimization techniques for production deployments",
            # Troubleshooting query (Rule 5)
            "error BSON type 10 invalid codec for MongoDB aggregation pipeline troubleshoot fix",
            # How-to query (Rule 5)
            "how to set up vector search index in MongoDB Atlas step by step guide",
        ]

        for query in test_queries:
            params = optimizer.optimize(query=query)
            total = params.vector_weight + params.text_weight
            assert abs(total - 1.0) < 1e-6, (
                f"Weights must sum to 1.0, got {total} for query '{query[:30]}...'"
            )


# ---- M24: _fallback_regex_search COLLSCAN ----


class TestM24AnchoredRegex:
    """_fallback_regex_search must use anchored regex pattern for index usage."""

    def test_fallback_regex_uses_anchored_pattern(self) -> None:
        """Regex search should use ^pattern for index prefix scan.

        Per antipattern-examples: 'Only a left-anchored, case-sensitive $regex
        can be converted into an efficient index range scan.'
        """
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage._fallback_regex_search)
        # Should use anchored regex (^pattern)
        assert "^" in source, (
            "_fallback_regex_search must use anchored regex (^pattern) for index prefix scan. "
            "Per antipattern-examples: unanchored regex causes COLLSCAN."
        )


# ---- M25: Workspace namespace logic copy-pasted 5 times ----


class TestM25WorkspaceNamespaceHelper:
    """Workspace namespace logic must be extracted to a helper function."""

    def test_get_collection_name_helper_exists(self) -> None:
        """A get_collection_name helper function must exist in mongo_impl."""
        from hybridrag.engine.kg import mongo_impl

        assert hasattr(mongo_impl, "get_collection_name"), (
            "mongo_impl must have a get_collection_name() helper function "
            "to replace the copy-pasted workspace namespace logic"
        )

    def test_get_collection_name_with_workspace(self) -> None:
        """Helper should prefix collection name with workspace when provided."""
        from hybridrag.engine.kg.mongo_impl import get_collection_name

        assert (
            get_collection_name("my_workspace", "entities") == "my_workspace_entities"
        )

    def test_get_collection_name_without_workspace(self) -> None:
        """Helper should return base name when workspace is None or empty."""
        from hybridrag.engine.kg.mongo_impl import get_collection_name

        assert get_collection_name(None, "entities") == "entities"
        assert get_collection_name("", "entities") == "entities"


# ---- M26: Sequential storage initialization ----


class TestM26ParallelStorageInit:
    """Storage initialization should use asyncio.gather for parallelism."""

    def test_initialize_storages_uses_asyncio_gather(self) -> None:
        """initialize_storages must use asyncio.gather instead of sequential await."""
        from hybridrag.engine.base_engine import BaseRAGEngine

        source = inspect.getsource(BaseRAGEngine.initialize_storages)
        assert "asyncio.gather" in source, (
            "initialize_storages must use asyncio.gather for parallel initialization "
            "instead of sequential awaits"
        )


# ---- M27: Migration loads all nodes+edges into memory ----


class TestM27CursorBasedMigration:
    """Migration must use cursor-based batch processing, not load all into memory."""

    def test_migration_uses_batch_processing(self) -> None:
        """_migrate_entity_relation_data should NOT call get_all_nodes()/get_all_edges()
        which loads everything into memory. Should use batched cursor instead."""
        from hybridrag.engine.base_engine import BaseRAGEngine

        source = inspect.getsource(BaseRAGEngine._migrate_entity_relation_data)
        # Should NOT load all nodes/edges into memory via get_all_nodes()/get_all_edges()
        # Instead should use cursor-based processing
        uses_get_all = "get_all_nodes()" in source and "get_all_edges()" in source
        uses_batch = "batch_size" in source or "async for" in source
        assert not uses_get_all or uses_batch, (
            "_migrate_entity_relation_data must NOT call get_all_nodes()/get_all_edges() "
            "to load everything into memory. Use batched cursor processing instead."
        )


# ---- M28: count_documents({}) before every search_labels call ----


class TestM28EstimatedDocumentCount:
    """search_labels must use estimated_document_count instead of count_documents({})."""

    def test_search_labels_uses_estimated_document_count(self) -> None:
        """search_labels should use estimated_document_count for empty check."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage.search_labels)
        assert "count_documents" not in source, (
            "search_labels must NOT use count_documents({}) -- too expensive. "
            "Use estimated_document_count() per query-optimizer skill."
        )
        assert "estimated_document_count" in source, (
            "search_labels must use estimated_document_count() for the empty-check "
            "instead of count_documents({})"
        )


# ---- M48: shared_storage uses time.time as type annotation ----


class TestM48TypeAnnotationFix:
    """shared_storage must use float instead of time.time as type annotation."""

    def test_lock_cleanup_data_uses_float_annotation(self) -> None:
        """_lock_cleanup_data type annotation should be float, not time.time."""
        from hybridrag.engine.kg import shared_storage

        source = inspect.getsource(shared_storage)
        # Should NOT have time.time as a type annotation
        # time.time is a function, not a type
        assert "dict[str, time.time]" not in source, (
            "_lock_cleanup_data must use 'dict[str, float]' not 'dict[str, time.time]'. "
            "time.time is a callable, not a type annotation."
        )

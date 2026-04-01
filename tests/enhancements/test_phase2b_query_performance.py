"""Tests for Phase 2B: Query Performance findings (H10, H11, H13, H14, H15).

Backing skill: mongodb-query-optimizer
- bulk_write preferred over multiple individual operations
- NEVER use $regex for search -- use direct equality or Atlas Search
- count_documents({}) is expensive -- use estimated_document_count() for unfiltered
- Filter early in pipelines with $match + $limit
"""

import inspect

# ---- H10: Unbounded find({}) in get_all_nodes/get_all_edges ----


class TestH10BoundedGetAllNodesEdges:
    """get_all_nodes and get_all_edges must accept limit param and use batch_size."""

    def test_get_all_nodes_accepts_limit_parameter(self) -> None:
        """get_all_nodes should accept an optional limit parameter."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        sig = inspect.signature(MongoGraphStorage.get_all_nodes)
        assert "limit" in sig.parameters, (
            "get_all_nodes must accept a 'limit' parameter"
        )
        # Default should be 0 (meaning no explicit limit applied)
        assert sig.parameters["limit"].default == 0

    def test_get_all_edges_accepts_limit_parameter(self) -> None:
        """get_all_edges should accept an optional limit parameter."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        sig = inspect.signature(MongoGraphStorage.get_all_edges)
        assert "limit" in sig.parameters, (
            "get_all_edges must accept a 'limit' parameter"
        )
        assert sig.parameters["limit"].default == 0

    def test_get_all_nodes_uses_batch_size(self) -> None:
        """get_all_nodes source should call batch_size for cursor control."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage.get_all_nodes)
        assert "batch_size" in source, (
            "get_all_nodes must use batch_size() on cursor for memory control"
        )

    def test_get_all_edges_uses_batch_size(self) -> None:
        """get_all_edges source should call batch_size for cursor control."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage.get_all_edges)
        assert "batch_size" in source, (
            "get_all_edges must use batch_size() on cursor for memory control"
        )


# ---- H11: DocStatus upsert uses N individual update_one calls ----


class TestH11BulkWriteUpsert:
    """DocStatusStorage.upsert must use bulk_write instead of N update_one calls."""

    def test_upsert_uses_bulk_write(self) -> None:
        """upsert should use bulk_write, not individual update_one calls."""
        from hybridrag.engine.kg.mongo_impl import MongoDocStatusStorage

        source = inspect.getsource(MongoDocStatusStorage.upsert)
        assert "bulk_write" in source, (
            "upsert must use bulk_write instead of N individual update_one calls"
        )

    def test_upsert_does_not_use_asyncio_gather_with_update_one(self) -> None:
        """upsert should NOT use asyncio.gather with individual update_one tasks."""
        from hybridrag.engine.kg.mongo_impl import MongoDocStatusStorage

        source = inspect.getsource(MongoDocStatusStorage.upsert)
        # Should not have the pattern: gather(*update_tasks) with update_one
        has_gather = "asyncio.gather" in source
        has_update_one = "update_one" in source
        assert not (has_gather and has_update_one), (
            "upsert must NOT use asyncio.gather with update_one -- use bulk_write"
        )

    def test_upsert_uses_ordered_false(self) -> None:
        """bulk_write should use ordered=False for parallel execution."""
        from hybridrag.engine.kg.mongo_impl import MongoDocStatusStorage

        source = inspect.getsource(MongoDocStatusStorage.upsert)
        assert "ordered=False" in source, (
            "bulk_write should use ordered=False for better performance"
        )


# ---- H13: COLLSCAN from case-insensitive regex in graph_search ----


class TestH13NoRegexInGraphSearch:
    """build_graph_lookup_pipeline must NOT use $regex for entity matching."""

    def test_pipeline_match_has_no_regex(self) -> None:
        """$match stage should NOT use $regex (causes COLLSCAN)."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("MongoDB", config)

        match_stage = pipeline[0]["$match"]
        match_str = str(match_stage)
        # Should not contain $regex or re.compile patterns
        assert "$regex" not in match_str, (
            "Pipeline $match must NOT use $regex (causes COLLSCAN per query-optimizer skill)"
        )
        assert "re.compile" not in match_str, (
            "Pipeline $match must NOT use compiled regex patterns"
        )

    def test_pipeline_uses_equality_on_lowercased_entity(self) -> None:
        """$match should use direct equality on lowercased entity name."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("  MongoDB  ", config)

        match_stage = pipeline[0]["$match"]
        # Should have $or with direct equality on lowercased name
        assert "$or" in match_stage
        conditions = match_stage["$or"]

        # Both conditions should use the lowercased, stripped entity name
        source_val = conditions[0].get(config.source_field)
        target_val = conditions[1].get(config.target_field)

        # Values should be plain strings (lowercased), not regex patterns
        assert isinstance(source_val, str), (
            f"Source match should be a plain string, got {type(source_val)}"
        )
        assert isinstance(target_val, str), (
            f"Target match should be a plain string, got {type(target_val)}"
        )
        assert source_val == "mongodb", (
            f"Source should match lowercased entity 'mongodb', got '{source_val}'"
        )
        assert target_val == "mongodb", (
            f"Target should match lowercased entity 'mongodb', got '{target_val}'"
        )


# ---- H14: No cycle prevention in $graphLookup ----


class TestH14GraphLookupCyclePrevention:
    """$graphLookup must have restrictSearchWithMatch and capped maxDepth."""

    def test_graph_lookup_has_restrict_search_with_match(self) -> None:
        """$graphLookup stage must include restrictSearchWithMatch."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("test_entity", config)

        # Find $graphLookup stage
        graph_lookup = None
        for stage in pipeline:
            if "$graphLookup" in stage:
                graph_lookup = stage["$graphLookup"]
                break

        assert graph_lookup is not None, "$graphLookup stage must exist"
        assert "restrictSearchWithMatch" in graph_lookup, (
            "$graphLookup must include restrictSearchWithMatch to prevent explosion"
        )

    def test_graph_lookup_max_depth_capped_at_5(self) -> None:
        """maxDepth should be capped at 5 even if config asks for more."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        # Config with very high max_depth
        config = GraphTraversalConfig(max_depth=20)
        pipeline = build_graph_lookup_pipeline("test_entity", config)

        # Find $graphLookup stage
        graph_lookup = None
        for stage in pipeline:
            if "$graphLookup" in stage:
                graph_lookup = stage["$graphLookup"]
                break

        assert graph_lookup is not None
        assert graph_lookup["maxDepth"] <= 5, (
            f"maxDepth must be capped at 5, got {graph_lookup['maxDepth']}"
        )

    def test_pipeline_has_limit_after_match(self) -> None:
        """Pipeline should have a $limit stage after $match to cap seed set."""
        from hybridrag.enhancements.graph_search import (
            GraphTraversalConfig,
            build_graph_lookup_pipeline,
        )

        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("test_entity", config)

        # Find the index of $match and $graphLookup
        match_idx = None
        graph_lookup_idx = None
        for i, stage in enumerate(pipeline):
            if "$match" in stage:
                match_idx = i
            if "$graphLookup" in stage:
                graph_lookup_idx = i

        assert match_idx is not None
        assert graph_lookup_idx is not None

        # There should be a $limit between $match and $graphLookup
        has_limit_between = False
        for i in range(match_idx + 1, graph_lookup_idx):
            if "$limit" in pipeline[i]:
                has_limit_between = True
                break

        assert has_limit_between, (
            "Pipeline must have $limit between $match and $graphLookup "
            "to cap the seed set and prevent unbounded fan-out"
        )


# ---- H15: count_documents({}) on large collections ----


class TestH15EstimatedDocumentCount:
    """Stats reporting must use estimated_document_count() not count_documents({})."""

    def test_get_knowledge_base_stats_uses_estimated_document_count(self) -> None:
        """get_knowledge_base_stats should use estimated_document_count for unfiltered counts."""
        from hybridrag.core.rag import HybridRAG

        source = inspect.getsource(HybridRAG.get_knowledge_base_stats)

        # Should NOT use count_documents({})
        assert "count_documents" not in source, (
            "get_knowledge_base_stats must NOT use count_documents({}) -- "
            "use estimated_document_count() per query-optimizer skill"
        )

        # Should use estimated_document_count
        assert "estimated_document_count" in source, (
            "get_knowledge_base_stats must use estimated_document_count() "
            "for unfiltered counts"
        )

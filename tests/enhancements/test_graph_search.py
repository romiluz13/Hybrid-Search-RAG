"""Tests for graph search module."""

import pytest

from hybridrag.enhancements.graph_search import (
    GraphEdge,
    GraphTraversalConfig,
    GraphTraversalResult,
    build_graph_lookup_pipeline,
    normalize_entity_name,
)


class TestNormalizeEntityName:
    """Test entity name normalization."""

    def test_lowercase(self) -> None:
        """Names should be lowercased."""
        assert normalize_entity_name("MongoDB") == "mongodb"
        assert normalize_entity_name("ATLAS") == "atlas"

    def test_strip_whitespace(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        assert normalize_entity_name("  mongodb  ") == "mongodb"
        assert normalize_entity_name("\tAtlas\n") == "atlas"

    def test_empty_string(self) -> None:
        """Empty string should return empty."""
        assert normalize_entity_name("") == ""
        assert normalize_entity_name("   ") == ""


class TestGraphTraversalConfig:
    """Test GraphTraversalConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration should have sensible values."""
        config = GraphTraversalConfig()
        assert config.edges_collection == "kg_edges"
        assert config.chunks_collection == "text_chunks"
        assert config.documents_collection == "documents"
        assert config.max_depth == 2
        assert config.max_nodes == 50
        assert config.workspace == ""

    def test_custom_values(self) -> None:
        """Custom values should override defaults."""
        config = GraphTraversalConfig(
            edges_collection="custom_edges",
            max_depth=5,
            max_nodes=100,
            workspace="test_workspace",
        )
        assert config.edges_collection == "custom_edges"
        assert config.max_depth == 5
        assert config.max_nodes == 100
        assert config.workspace == "test_workspace"

    def test_workspace_affects_collection_names(self) -> None:
        """Workspace prefix doesn't auto-apply (handled at runtime)."""
        config = GraphTraversalConfig(workspace="myspace")
        # Collection names stay as configured
        assert config.edges_collection == "kg_edges"
        assert config.workspace == "myspace"


class TestGraphEdge:
    """Test GraphEdge dataclass."""

    def test_create_edge(self) -> None:
        """Edge should store all fields."""
        edge = GraphEdge(
            source="mongodb",
            target="atlas",
            relationship_type="part_of",
            weight=0.9,
            depth=1,
        )
        assert edge.source == "mongodb"
        assert edge.target == "atlas"
        assert edge.relationship_type == "part_of"
        assert edge.weight == 0.9
        assert edge.depth == 1

    def test_default_depth(self) -> None:
        """Edge should have default depth."""
        edge = GraphEdge(
            source="a",
            target="b",
            relationship_type="related",
            weight=1.0,
        )
        assert edge.depth == 0


class TestGraphTraversalResult:
    """Test GraphTraversalResult dataclass."""

    def test_create_result(self) -> None:
        """Result should store entities, edges, and counts."""
        edges = [
            GraphEdge(source="a", target="b", relationship_type="r1", weight=1.0),
            GraphEdge(source="b", target="c", relationship_type="r2", weight=0.8),
        ]
        result = GraphTraversalResult(
            starting_entity="a",
            related_entities=["b", "c"],
            edges=edges,
            max_depth_reached=1,
            total_edges_traversed=2,
        )
        assert result.starting_entity == "a"
        assert len(result.related_entities) == 2
        assert len(result.edges) == 2
        assert result.max_depth_reached == 1


class TestBuildGraphLookupPipeline:
    """Test $graphLookup pipeline builder."""

    def test_default_pipeline_structure(self) -> None:
        """Pipeline should have correct stages."""
        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("test_entity", config)

        # Should be a list of pipeline stages
        assert isinstance(pipeline, list)
        assert len(pipeline) >= 3

        # First stage should be $match
        assert "$match" in pipeline[0]

        # Should have $graphLookup stage
        graph_lookup_stage = None
        for stage in pipeline:
            if "$graphLookup" in stage:
                graph_lookup_stage = stage
                break
        assert graph_lookup_stage is not None

    def test_graph_lookup_configuration(self) -> None:
        """$graphLookup should use config fields."""
        config = GraphTraversalConfig(
            edges_collection="my_edges",
            source_field="from_node",
            target_field="to_node",
            max_depth=3,
        )
        pipeline = build_graph_lookup_pipeline("test", config)

        # Find $graphLookup stage
        graph_lookup = None
        for stage in pipeline:
            if "$graphLookup" in stage:
                graph_lookup = stage["$graphLookup"]
                break

        assert graph_lookup is not None
        assert graph_lookup["from"] == "my_edges"
        assert graph_lookup["connectFromField"] == "to_node"
        assert graph_lookup["connectToField"] == "from_node"
        # maxDepth is depth - 1 since first hop already matched
        assert graph_lookup["maxDepth"] == 2

    def test_entity_normalization_in_pipeline(self) -> None:
        """Entity name should use lowercased equality in match stage (no regex)."""
        config = GraphTraversalConfig()
        pipeline = build_graph_lookup_pipeline("  MongoDB  ", config)

        # First stage match should have $or with direct equality on lowercased name
        match_stage = pipeline[0]["$match"]
        # Match uses $or for source or target
        assert "$or" in match_stage
        # Check equality match on lowercased, stripped entity name
        conditions = match_stage["$or"]
        source_condition = conditions[0]["source_node_id"]
        target_condition = conditions[1]["target_node_id"]
        # Should be plain string (lowercased), not regex pattern
        assert isinstance(source_condition, str)
        assert source_condition == "mongodb"
        assert isinstance(target_condition, str)
        assert target_condition == "mongodb"


class TestGraphSearchIntegration:
    """Integration tests (require MongoDB connection).

    L22: Removed empty stub bodies with pass/pytest.skip() calls.
    L26: Removed redundant pytest.skip() inside @pytest.mark.skip decorated tests.
    TODO: Implement when MongoDB integration test infrastructure is available.
    Tracked stubs:
    - test_graph_traversal_execution: Full traversal with real graph data
    - test_expand_entities_via_graph: Entity expansion via graph edges
    - test_get_chunks_for_entities: Chunk retrieval for discovered entities
    """

    @pytest.mark.skip(reason="Requires MongoDB connection - TODO: implement")
    async def test_graph_traversal_execution(self) -> None:
        """Test actual graph traversal execution with MongoDB."""

    @pytest.mark.skip(reason="Requires MongoDB connection - TODO: implement")
    async def test_expand_entities_via_graph(self) -> None:
        """Test entity expansion via graph traversal."""

    @pytest.mark.skip(reason="Requires MongoDB connection - TODO: implement")
    async def test_get_chunks_for_entities(self) -> None:
        """Test chunk retrieval for entities from graph."""

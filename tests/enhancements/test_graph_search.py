"""Tests for graph search module."""

from datetime import UTC, datetime, timedelta

import pytest

from hybridrag.enhancements.graph_search import (
    GraphEdge,
    GraphTraversalConfig,
    GraphTraversalResult,
    build_graph_lookup_pipeline,
    expand_entities_via_graph,
    get_chunks_for_entities,
    graph_traversal,
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
    """Integration tests for real MongoDB-backed graph traversal behavior."""

    @pytest.fixture
    async def seeded_graph_db(self, mongodb_test_db):
        """Seed graph edges and chunks into an isolated test database."""
        now = datetime.now(UTC)

        await mongodb_test_db["kg_edges"].insert_many(
            [
                {
                    "source_node_id": "mongodb",
                    "target_node_id": "atlas",
                    "relationship_type": "platform_for",
                    "weight": 0.95,
                },
                {
                    "source_node_id": "atlas",
                    "target_node_id": "vector search",
                    "relationship_type": "supports",
                    "weight": 0.90,
                },
                {
                    "source_node_id": "atlas",
                    "target_node_id": "search indexes",
                    "relationship_type": "uses",
                    "weight": 0.75,
                },
            ]
        )

        await mongodb_test_db["text_chunks"].insert_many(
            [
                {
                    "document_id": "doc-1",
                    "content": "Atlas supports Vector Search with search indexes.",
                    "entities": [
                        {"name": "Atlas"},
                        {"name": "Vector Search"},
                    ],
                    "timestamp": now,
                    "metadata": {"source": "mongodb-docs"},
                },
                {
                    "document_id": "doc-2",
                    "content": "MongoDB Atlas is the platform behind the search stack.",
                    "entities": [
                        {"name": "MongoDB"},
                        {"name": "Atlas"},
                    ],
                    "timestamp": now - timedelta(hours=1),
                    "metadata": {"source": "product-overview"},
                },
            ]
        )

        return mongodb_test_db

    @pytest.mark.asyncio
    async def test_graph_traversal_execution(self, seeded_graph_db) -> None:
        """Test actual graph traversal execution with MongoDB."""
        result = await graph_traversal(
            seeded_graph_db,
            "MongoDB",
            GraphTraversalConfig(max_depth=2, max_nodes=10),
        )

        assert "atlas" in result.related_entities
        assert "vector search" in result.related_entities
        assert result.total_edges_traversed >= 2
        assert any(edge.relationship_type == "supports" for edge in result.edges)

    @pytest.mark.asyncio
    async def test_expand_entities_via_graph(self, seeded_graph_db) -> None:
        """Test entity expansion via graph traversal."""
        entities, edges = await expand_entities_via_graph(
            seeded_graph_db,
            ["MongoDB"],
            GraphTraversalConfig(max_depth=2, max_nodes=10),
        )

        assert "atlas" in entities
        assert "vector search" in entities
        assert len(edges) >= 2

    @pytest.mark.asyncio
    async def test_get_chunks_for_entities(self, seeded_graph_db) -> None:
        """Test chunk retrieval for entities from graph."""
        chunks = await get_chunks_for_entities(
            seeded_graph_db,
            ["Atlas", "Vector Search"],
            limit=10,
            config=GraphTraversalConfig(),
        )

        assert chunks
        assert chunks[0]["matched_entities"] >= 1
        assert any("Atlas" in chunk["content"] for chunk in chunks)

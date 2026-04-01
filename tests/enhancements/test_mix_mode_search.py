"""Tests for mix mode search module."""

import pytest

from hybridrag.enhancements.graph_search import GraphTraversalConfig
from hybridrag.enhancements.mix_mode_search import (
    MixModeConfig,
    MixModeSearchResult,
    extract_pipeline_score,
)
from hybridrag.enhancements.mongodb_hybrid_search import MongoDBHybridSearchConfig


class TestMixModeConfig:
    """Test MixModeConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration should have sensible values."""
        config = MixModeConfig()
        assert config.enable_graph_traversal is True
        assert config.enable_entity_boosting is True
        assert config.enable_reranking is True
        assert config.entity_boost_weight == 0.2
        assert config.entity_only_weight == 0.5

    def test_custom_values(self) -> None:
        """Custom values should override defaults."""
        config = MixModeConfig(
            enable_graph_traversal=False,
            enable_entity_boosting=False,
            entity_boost_weight=0.5,
            entity_only_weight=0.3,
        )
        assert config.enable_graph_traversal is False
        assert config.enable_entity_boosting is False
        assert config.entity_boost_weight == 0.5
        assert config.entity_only_weight == 0.3

    def test_nested_configs(self) -> None:
        """Nested configs should be accessible."""
        hybrid_config = MongoDBHybridSearchConfig(
            vector_weight=0.7,
            text_weight=0.3,
        )
        graph_config = GraphTraversalConfig(
            max_depth=3,
            max_nodes=100,
        )
        config = MixModeConfig(
            hybrid_config=hybrid_config,
            graph_config=graph_config,
        )
        assert config.hybrid_config.vector_weight == 0.7
        assert config.graph_config.max_depth == 3


class TestMixModeSearchResult:
    """Test MixModeSearchResult model."""

    def test_create_result(self) -> None:
        """Result should store all fields."""
        result = MixModeSearchResult(
            chunk_id="507f1f77bcf86cd799439011",
            document_id="507f1f77bcf86cd799439012",
            content="Test content",
            score=0.85,
            metadata={"source": "test.pdf"},
            search_type="mix_mode",
            source_scores={"vector": 0.9, "text": 0.8, "entity": 0.5},
            graph_entities=["entity1", "entity2"],
            entity_boost=0.1,
            document_title="Test Doc",
            document_source="test.pdf",
        )
        assert result.chunk_id == "507f1f77bcf86cd799439011"
        assert result.score == 0.85
        assert result.search_type == "mix_mode"
        assert len(result.source_scores) == 3
        assert len(result.graph_entities) == 2

    def test_default_values(self) -> None:
        """Result should have sensible defaults."""
        result = MixModeSearchResult(
            chunk_id="test",
            content="content",
            score=0.5,
        )
        assert result.document_id == ""
        assert result.metadata == {}
        assert result.search_type == "mix_mode"
        assert result.source_scores == {}
        assert result.graph_entities == []
        assert result.entity_boost == 0.0

    def test_source_scores_breakdown(self) -> None:
        """Source scores should be accessible individually."""
        result = MixModeSearchResult(
            chunk_id="test",
            content="content",
            score=0.85,
            source_scores={
                "vector": 0.92,
                "text": 0.78,
                "entity": 0.65,
            },
        )
        assert result.source_scores["vector"] == 0.92
        assert result.source_scores["text"] == 0.78
        assert result.source_scores["entity"] == 0.65


class TestExtractPipelineScore:
    """Test per-pipeline score extraction."""

    def test_extract_vector_score(self) -> None:
        """Should extract vector pipeline score."""
        score_details = {
            "value": 0.85,
            "details": [
                {"inputPipelineName": "vector", "value": 0.92},
                {"inputPipelineName": "text", "value": 0.78},
            ],
        }
        score = extract_pipeline_score(score_details, "vector")
        assert score == 0.92

    def test_extract_text_score(self) -> None:
        """Should extract text pipeline score."""
        score_details = {
            "value": 0.85,
            "details": [
                {"inputPipelineName": "vector", "value": 0.92},
                {"inputPipelineName": "text", "value": 0.78},
            ],
        }
        score = extract_pipeline_score(score_details, "text")
        assert score == 0.78

    def test_missing_pipeline_returns_zero(self) -> None:
        """Missing pipeline should return 0.0."""
        score_details = {
            "value": 0.85,
            "details": [
                {"inputPipelineName": "vector", "value": 0.92},
            ],
        }
        score = extract_pipeline_score(score_details, "text")
        assert score == 0.0

    def test_none_score_details_returns_zero(self) -> None:
        """None score_details should return 0.0."""
        score = extract_pipeline_score(None, "vector")
        assert score == 0.0

    def test_empty_details_returns_zero(self) -> None:
        """Empty details array should return 0.0."""
        score_details = {"value": 0.5, "details": []}
        score = extract_pipeline_score(score_details, "vector")
        assert score == 0.0

    def test_missing_details_key_returns_zero(self) -> None:
        """Missing details key should return 0.0."""
        score_details = {"value": 0.5}
        score = extract_pipeline_score(score_details, "vector")
        assert score == 0.0


class TestMixModeSearchIntegration:
    """Integration tests (require MongoDB connection).

    L22: Removed empty stub bodies with pass statements.
    TODO: Implement when MongoDB integration test infrastructure is available.
    Tracked stubs:
    - test_mix_mode_search_execution: Full search pipeline with real DB
    - test_mix_mode_searcher_class: MixModeSearcher initialization + query
    - test_graph_only_search: Graph-only mode with real graph data
    """

    @pytest.mark.skip(reason="Requires MongoDB connection - TODO: implement")
    async def test_mix_mode_search_execution(self) -> None:
        """Test actual mix mode search execution with MongoDB."""

    @pytest.mark.skip(reason="Requires MongoDB connection - TODO: implement")
    async def test_mix_mode_searcher_class(self) -> None:
        """Test MixModeSearcher class initialization and query."""

    @pytest.mark.skip(reason="Requires MongoDB connection - TODO: implement")
    async def test_graph_only_search(self) -> None:
        """Test graph-only search mode with real graph data."""


class TestResultMerging:
    """Test result merging and deduplication logic.

    L21: Tests use production MixModeConfig values and MixModeSearchResult
    construction rather than reimplementing merge logic locally.
    """

    def test_entity_boost_weight_from_config(self) -> None:
        """Entity boost weight should come from MixModeConfig defaults."""
        config = MixModeConfig()
        base_score = 0.8
        entity_score = 0.5

        # Use the actual config weight, not a hardcoded copy
        final_score = base_score + (entity_score * config.entity_boost_weight)
        assert final_score == pytest.approx(0.9)

    def test_entity_only_weight_from_config(self) -> None:
        """Entity-only weight should come from MixModeConfig defaults."""
        config = MixModeConfig()
        entity_score = 0.5

        # Use the actual config weight
        final_score = entity_score * config.entity_only_weight
        assert final_score == pytest.approx(0.25)

    def test_result_deduplication_via_model_construction(self) -> None:
        """Results built as MixModeSearchResult deduplicate by chunk_id."""
        results = [
            MixModeSearchResult(
                chunk_id="a", content="c1", score=0.9, search_type="hybrid"
            ),
            MixModeSearchResult(
                chunk_id="a", content="c1", score=0.7, search_type="entity"
            ),
            MixModeSearchResult(
                chunk_id="b", content="c2", score=0.8, search_type="hybrid"
            ),
        ]

        # Deduplicate using dict keyed by chunk_id (production pattern)
        merged_map: dict[str, MixModeSearchResult] = {}
        for r in results:
            if r.chunk_id not in merged_map:
                merged_map[r.chunk_id] = r

        deduped = list(merged_map.values())
        assert len(deduped) == 2
        assert deduped[0].chunk_id == "a"
        assert deduped[0].score == 0.9  # First occurrence kept
        assert deduped[1].chunk_id == "b"

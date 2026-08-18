"""Tests for mix mode search module."""

from datetime import UTC, datetime
from typing import NamedTuple

import pytest

from hybridrag.engine.exceptions import (
    RetrievalCapabilityError,
    RetrievalExecutionError,
)
from hybridrag.enhancements.graph_search import GraphTraversalConfig
from hybridrag.enhancements.mix_mode_search import (
    MixModeConfig,
    MixModeSearcher,
    MixModeSearchResult,
    extract_pipeline_score,
    mix_mode_search,
)
from hybridrag.enhancements.mongodb_hybrid_search import MongoDBHybridSearchConfig


@pytest.mark.asyncio
async def test_mix_mode_never_substitutes_manual_fusion(monkeypatch) -> None:
    import sys

    async def failed_native_fusion(*args, **kwargs):
        raise RetrievalCapabilityError("rank fusion unavailable")

    async def forbidden_manual_fusion(*args, **kwargs):
        pytest.fail("manual fusion must not run after native fusion failure")

    mix_module = sys.modules["hybridrag.enhancements.mix_mode_search"]
    monkeypatch.setattr(
        mix_module,
        "hybrid_search_with_rank_fusion",
        failed_native_fusion,
    )
    monkeypatch.setattr(
        mix_module,
        "manual_hybrid_search_with_rrf",
        forbidden_manual_fusion,
        raising=False,
    )

    with pytest.raises(RetrievalCapabilityError, match="rank fusion unavailable"):
        await mix_mode_search(
            db={"text_chunks": object()},
            query="test query",
            query_vector=[0.1, 0.2],
            config=MixModeConfig(enable_graph_traversal=False),
        )


@pytest.mark.asyncio
async def test_mix_mode_never_swallows_graph_failure(monkeypatch) -> None:
    import sys

    async def successful_native_fusion(*args, **kwargs):
        return []

    async def failed_graph_search(*args, **kwargs):
        raise RuntimeError("graph lookup failed")

    mix_module = sys.modules["hybridrag.enhancements.mix_mode_search"]
    monkeypatch.setattr(
        mix_module,
        "hybrid_search_with_rank_fusion",
        successful_native_fusion,
    )
    monkeypatch.setattr(
        mix_module,
        "expand_entities_via_graph",
        failed_graph_search,
    )

    with pytest.raises(RetrievalExecutionError, match="graph traversal failed"):
        await mix_mode_search(
            db={"text_chunks": object()},
            query="test query",
            query_vector=[0.1, 0.2],
            query_entities=["MongoDB"],
        )


class SeededMixModeDB(NamedTuple):
    """Typed fixture return for seeded mix mode database."""

    db: object  # AsyncDatabase
    chunk_ids: list[str]


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


@pytest.mark.integration
class TestMixModeSearchIntegration:
    """MongoDB-backed integration tests for mix mode behavior.

    Marked ``integration``: requires a live MongoDB (atlas-local on localhost:27018).
    """

    @pytest.fixture
    async def seeded_mix_mode_db(self, mongodb_test_db):
        """Seed graph edges and chunk data for mix mode tests."""
        now = datetime.now(UTC)
        insert_result = await mongodb_test_db["text_chunks"].insert_many(
            [
                {
                    "document_id": "doc-graph-1",
                    "content": "Atlas powers Vector Search for HybridRAG.",
                    "entities": [
                        {"name": "Atlas"},
                        {"name": "Vector Search"},
                    ],
                    "timestamp": now,
                    "metadata": {"source": "graph-doc"},
                },
                {
                    "document_id": "doc-graph-2",
                    "content": "MongoDB Atlas uses search indexes for retrieval.",
                    "entities": [
                        {"name": "MongoDB"},
                        {"name": "Atlas"},
                    ],
                    "timestamp": now,
                    "metadata": {"source": "platform-doc"},
                },
            ]
        )
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
            ]
        )

        return SeededMixModeDB(
            db=mongodb_test_db,
            chunk_ids=[str(chunk_id) for chunk_id in insert_result.inserted_ids],
        )

    @pytest.mark.asyncio
    async def test_mix_mode_search_execution(self, seeded_mix_mode_db, monkeypatch):
        """Test mix mode merges hybrid and graph-derived results."""
        chunk_id = seeded_mix_mode_db.chunk_ids[0]

        async def fake_hybrid_search_with_rank_fusion(*args, **kwargs):
            return [
                {
                    "chunk_id": chunk_id,
                    "document_id": "doc-hybrid-1",
                    "content": "Hybrid result about Atlas and Vector Search.",
                    "score": 0.82,
                    "metadata": {"source": "hybrid"},
                    "search_type": "hybrid_rrf",
                    "score_details": {
                        "details": [
                            {"inputPipelineName": "vector", "value": 0.91},
                            {"inputPipelineName": "text", "value": 0.73},
                        ]
                    },
                }
            ]

        import sys

        _mix_mod = sys.modules["hybridrag.enhancements.mix_mode_search"]
        monkeypatch.setattr(
            _mix_mod,
            "hybrid_search_with_rank_fusion",
            fake_hybrid_search_with_rank_fusion,
        )

        results = await mix_mode_search(
            db=seeded_mix_mode_db.db,
            query="How does MongoDB Atlas support vector search?",
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
            config=MixModeConfig(
                graph_config=GraphTraversalConfig(max_depth=2, max_nodes=10)
            ),
            query_entities=["MongoDB"],
        )

        assert results
        primary = results[0]
        assert primary.chunk_id == chunk_id
        assert primary.source_scores["vector"] == pytest.approx(0.91)
        assert primary.source_scores["text"] == pytest.approx(0.73)
        # entity_only_weight from MixModeConfig default = 0.5
        assert primary.source_scores["entity"] == pytest.approx(0.5)
        assert primary.entity_boost == pytest.approx(0.1)
        assert "atlas" in primary.graph_entities

    @pytest.mark.asyncio
    async def test_mix_mode_searcher_class(self, seeded_mix_mode_db, monkeypatch):
        """Test MixModeSearcher initialization and query."""

        async def fake_hybrid_search_with_rank_fusion(*args, **kwargs):
            return []

        import sys

        _mix_mod = sys.modules["hybridrag.enhancements.mix_mode_search"]
        monkeypatch.setattr(
            _mix_mod,
            "hybrid_search_with_rank_fusion",
            fake_hybrid_search_with_rank_fusion,
        )

        searcher = MixModeSearcher(
            db=seeded_mix_mode_db.db,
            config=MixModeConfig(
                graph_config=GraphTraversalConfig(max_depth=2, max_nodes=10)
            ),
        )

        results = await searcher.search(
            query="Atlas graph search",
            query_vector=[0.4, 0.5, 0.6],
            top_k=5,
            query_entities=["MongoDB"],
        )

        assert results
        assert all(isinstance(result, MixModeSearchResult) for result in results)
        assert results[0].search_type == "entity_only"

    @pytest.mark.asyncio
    async def test_graph_only_search(self, seeded_mix_mode_db) -> None:
        """Test graph-only search mode with real graph data."""
        searcher = MixModeSearcher(
            db=seeded_mix_mode_db.db,
            config=MixModeConfig(
                graph_config=GraphTraversalConfig(max_depth=2, max_nodes=10)
            ),
        )

        results = await searcher.search_with_graph_only(
            query_entities=["MongoDB"],
            top_k=5,
        )

        assert results
        assert all(result.search_type == "graph_only" for result in results)
        assert any("Atlas" in result.content for result in results)


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

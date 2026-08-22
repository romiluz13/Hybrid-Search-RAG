"""Tests for MongoDB hybrid search pipeline correctness.

Validates CRITICAL fixes C5, C6, C7 from the MongoDB audit:
- C5: Mixed $project inclusion/exclusion (server error)
- C6: $scoreFusion wrong $meta field name
- C7: $scoreFusion normalization placement
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pymongo.errors import OperationFailure

from hybridrag.engine.exceptions import (
    RetrievalCapabilityError,
    RetrievalExecutionError,
    RetrievalValidationError,
)


class _FakeCursor:
    """Minimal async cursor stub for pipeline capture."""

    def __init__(self):
        self._results = []

    async def to_list(self, length=None):
        return self._results


def _make_mock_collection():
    """Create a mock collection that captures aggregate pipeline."""
    collection = AsyncMock()
    fake_cursor = _FakeCursor()
    # aggregate returns a coroutine that resolves to a cursor
    collection.aggregate = AsyncMock(return_value=fake_cursor)
    return collection, fake_cursor


def test_ann_candidates_respect_server_maximum():
    from hybridrag.enhancements.mongodb_hybrid_search import (
        MongoDBHybridSearchConfig,
        calculate_num_candidates,
    )

    assert calculate_num_candidates(600) == 10_000
    with pytest.raises(RetrievalValidationError, match="numCandidates"):
        MongoDBHybridSearchConfig(vector_num_candidates=10_001)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_name",
    ["hybrid_search_with_rank_fusion", "hybrid_search_with_score_fusion"],
)
async def test_native_hybrid_search_never_silently_changes_strategy(search_name):
    from hybridrag.enhancements import mongodb_hybrid_search

    collection, _ = _make_mock_collection()
    collection.aggregate.side_effect = OperationFailure(
        "Unrecognized pipeline stage name",
        code=40324,
    )
    search = getattr(mongodb_hybrid_search, search_name)

    with pytest.raises(RetrievalCapabilityError, match="fusion is unavailable"):
        await search(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=5,
        )

    assert collection.aggregate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_name",
    ["hybrid_search_with_rank_fusion", "hybrid_search_with_score_fusion"],
)
async def test_native_hybrid_search_does_not_mislabel_execution_failure(
    search_name,
):
    from hybridrag.enhancements import mongodb_hybrid_search

    collection, _ = _make_mock_collection()
    collection.aggregate.side_effect = OperationFailure("invalid pipeline", code=9)
    search = getattr(mongodb_hybrid_search, search_name)

    with pytest.raises(RetrievalExecutionError, match="fusion failed"):
        await search(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=5,
        )


@pytest.mark.asyncio
async def test_filtered_text_search_never_retries_without_filter():
    from hybridrag.enhancements.filters import AtlasSearchFilterConfig
    from hybridrag.enhancements.mongodb_hybrid_search import text_only_search

    collection, _ = _make_mock_collection()
    collection.aggregate.side_effect = OperationFailure("text search failed")

    with pytest.raises(RetrievalExecutionError, match="text search failed"):
        await text_only_search(
            collection=collection,
            query_text="test query",
            top_k=5,
            filter_config=AtlasSearchFilterConfig(
                equality_filters={"metadata.tenant": "tenant-a"}
            ),
        )

    assert collection.aggregate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_name",
    ["vector_only_search", "vector_search_with_lexical_prefilters"],
)
async def test_exported_vector_search_never_swallows_or_substitutes_failure(
    search_name,
):
    from hybridrag.enhancements import mongodb_hybrid_search

    collection, _ = _make_mock_collection()
    collection.aggregate.side_effect = OperationFailure("vector search failed")
    search = getattr(mongodb_hybrid_search, search_name)

    with pytest.raises(RetrievalExecutionError, match="vector search failed"):
        await search(
            collection=collection,
            query_vector=[0.1] * 1024,
            top_k=5,
        )

    assert collection.aggregate.await_count == 1


@pytest.mark.asyncio
async def test_manual_hybrid_search_never_returns_a_single_surviving_branch(
    monkeypatch,
):
    from hybridrag.enhancements import mongodb_hybrid_search

    async def failed_vector_search(*args, **kwargs):
        raise RetrievalExecutionError("vector branch failed")

    async def successful_text_search(*args, **kwargs):
        return []

    monkeypatch.setattr(
        mongodb_hybrid_search, "vector_only_search", failed_vector_search
    )
    monkeypatch.setattr(
        mongodb_hybrid_search, "text_only_search", successful_text_search
    )

    with pytest.raises(RetrievalExecutionError, match="vector branch failed"):
        await mongodb_hybrid_search.manual_hybrid_search_with_rrf(
            collection=AsyncMock(),
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=5,
        )


@pytest.mark.asyncio
async def test_legacy_fusion_rejects_independent_filter_models():
    from hybridrag.enhancements.filters import AtlasSearchFilterConfig
    from hybridrag.enhancements.mongodb_hybrid_search import (
        hybrid_search_with_rank_fusion,
    )

    collection, _ = _make_mock_collection()

    with pytest.raises(
        RetrievalValidationError,
        match="unified FilterConfig",
    ):
        await hybrid_search_with_rank_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            atlas_filter_config=AtlasSearchFilterConfig(
                equality_filters={"metadata.tenant": "tenant-a"}
            ),
        )

    collection.aggregate.assert_not_awaited()


@pytest.mark.asyncio
async def test_weighted_text_search_never_substitutes_simple_search():
    from hybridrag.enhancements.mongodb_hybrid_search import (
        MongoDBHybridSearchConfig,
        multi_field_text_search,
    )

    collection, _ = _make_mock_collection()
    collection.aggregate.side_effect = OperationFailure("weighted search failed")

    with pytest.raises(RetrievalExecutionError, match="text search failed"):
        await multi_field_text_search(
            collection=collection,
            query_text="test query",
            config=MongoDBHybridSearchConfig(text_search_path_weights={"content": 1.0}),
        )

    assert collection.aggregate.await_count == 1


class TestC5MixedProjection:
    """C5: No mixed $project inclusion/exclusion in search pipelines.

    MongoDB rejects $project stages that mix inclusions (field: 1) with
    exclusions (field: 0), except for _id: 0 which is always allowed.
    """

    @pytest.mark.asyncio
    async def test_text_only_search_no_mixed_projection(self):
        """text_only_search must not have 'vector: 0' mixed with inclusions."""
        from hybridrag.enhancements.mongodb_hybrid_search import text_only_search

        collection, _ = _make_mock_collection()

        await text_only_search(
            collection=collection,
            query_text="test query",
            top_k=5,
        )

        # Get the pipeline passed to aggregate
        pipeline = collection.aggregate.call_args[0][0]

        # Find all $project stages
        project_stages = [s for s in pipeline if "$project" in s]
        assert len(project_stages) > 0, "Expected at least one $project stage"

        for project_stage in project_stages:
            proj = project_stage["$project"]
            # Check for mixed inclusion/exclusion
            has_inclusion = any(
                v == 1 for k, v in proj.items() if isinstance(v, int) and k != "_id"
            )
            has_exclusion_vector = "vector" in proj and proj["vector"] == 0
            assert not (has_inclusion and has_exclusion_vector), (
                f"Mixed $project inclusion/exclusion detected: {proj}. "
                "'vector': 0 must not coexist with field: 1 inclusions."
            )

    @pytest.mark.asyncio
    async def test_vector_only_search_no_mixed_projection(self):
        """vector_only_search must not have 'vector: 0' mixed with inclusions."""
        from hybridrag.enhancements.mongodb_hybrid_search import vector_only_search

        collection, _ = _make_mock_collection()

        await vector_only_search(
            collection=collection,
            query_vector=[0.1] * 1024,
            top_k=5,
        )

        pipeline = collection.aggregate.call_args[0][0]
        project_stages = [s for s in pipeline if "$project" in s]
        assert len(project_stages) > 0

        for project_stage in project_stages:
            proj = project_stage["$project"]
            has_inclusion = any(
                v == 1 for k, v in proj.items() if isinstance(v, int) and k != "_id"
            )
            has_exclusion_vector = "vector" in proj and proj["vector"] == 0
            assert not (has_inclusion and has_exclusion_vector), (
                f"Mixed $project inclusion/exclusion detected: {proj}"
            )

    @pytest.mark.asyncio
    async def test_vector_search_with_lexical_prefilters_no_mixed_projection(self):
        """vector_search_with_lexical_prefilters must not have mixed projection."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            vector_search_with_lexical_prefilters,
        )

        collection, _ = _make_mock_collection()

        await vector_search_with_lexical_prefilters(
            collection=collection,
            query_vector=[0.1] * 1024,
            top_k=5,
        )

        pipeline = collection.aggregate.call_args[0][0]
        project_stages = [s for s in pipeline if "$project" in s]
        assert len(project_stages) > 0

        for project_stage in project_stages:
            proj = project_stage["$project"]
            has_inclusion = any(
                v == 1 for k, v in proj.items() if isinstance(v, int) and k != "_id"
            )
            has_exclusion_vector = "vector" in proj and proj["vector"] == 0
            assert not (has_inclusion and has_exclusion_vector), (
                f"Mixed $project inclusion/exclusion detected: {proj}"
            )


class TestC6ScoreFusionMetaField:
    """C6: $scoreFusion must use correct $meta field name.

    There is no '$meta: scoreFusionScore' in MongoDB.
    Use '$meta: score' for the standard search score.
    """

    @pytest.mark.asyncio
    async def test_score_fusion_uses_correct_meta_field(self):
        """hybrid_search_with_score_fusion must use '$meta: score', not 'scoreFusionScore'."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_score_fusion,
        )

        collection, _ = _make_mock_collection()

        await hybrid_search_with_score_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=5,
        )

        pipeline = collection.aggregate.call_args[0][0]

        # Check no stage uses 'scoreFusionScore'
        pipeline_str = str(pipeline)
        assert "scoreFusionScore" not in pipeline_str, (
            "Pipeline must not contain '$meta: scoreFusionScore'. "
            "Use '$meta: score' instead."
        )

        # Verify correct meta field is used in $addFields
        add_fields_stages = [s for s in pipeline if "$addFields" in s]
        assert len(add_fields_stages) > 0, "Expected at least one $addFields stage"
        found_score_meta = False
        for stage in add_fields_stages:
            for _field_name, field_val in stage["$addFields"].items():
                if isinstance(field_val, dict) and "$meta" in field_val:
                    assert field_val["$meta"] == "score", (
                        f"$meta field should be 'score', got '{field_val['$meta']}'"
                    )
                    found_score_meta = True
        assert found_score_meta, "Expected $addFields with $meta: 'score'"


class TestC7ScoreFusionNormalization:
    """C7: Verify $scoreFusion normalization placement.

    Per official MongoDB docs, 'normalization' goes INSIDE 'input'
    as a sibling of 'pipelines'. This test verifies the correct position.
    """

    @pytest.mark.asyncio
    async def test_normalization_inside_input(self):
        """normalization must be inside 'input' as a sibling of 'pipelines'."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_score_fusion,
        )

        collection, _ = _make_mock_collection()

        await hybrid_search_with_score_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=5,
        )

        pipeline = collection.aggregate.call_args[0][0]

        # Find the $scoreFusion stage
        score_fusion_stages = [s for s in pipeline if "$scoreFusion" in s]
        assert len(score_fusion_stages) == 1, "Expected exactly one $scoreFusion stage"

        sf = score_fusion_stages[0]["$scoreFusion"]

        # normalization MUST be inside input (sibling of pipelines)
        assert "input" in sf, "$scoreFusion must have 'input' key"
        assert "normalization" in sf["input"], (
            "normalization must be inside 'input' as a sibling of 'pipelines'. "
            f"Found keys in input: {list(sf['input'].keys())}"
        )
        assert sf["input"]["normalization"] == "sigmoid"

        # normalization must NOT be at the top level of $scoreFusion
        # (it belongs inside input, not as a sibling of input)
        # Note: normalization IS correct inside input per official docs

    @pytest.mark.asyncio
    async def test_weighted_score_fusion_uses_expression_sum(self):
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
            hybrid_search_with_score_fusion,
        )

        collection, _ = _make_mock_collection()

        await hybrid_search_with_score_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            config=MongoDBHybridSearchConfig(vector_weight=0.7, text_weight=0.3),
        )

        pipeline = collection.aggregate.call_args[0][0]
        combination = pipeline[0]["$scoreFusion"]["combination"]
        assert combination == {
            "method": "expression",
            "expression": {
                "$sum": [
                    {"$multiply": ["$$vector", 0.7]},
                    {"$multiply": ["$$text", 0.3]},
                ]
            },
        }


# ---------------------------------------------------------------------------
# Per-branch over-fetch floor (inspired by Anthropic CMA cookbook)
# ---------------------------------------------------------------------------


class TestBranchOverfetchFloor:
    """Verify that $rankFusion / $scoreFusion input pipelines over-fetch
    using ``max(top_k * branch_overfetch_factor, branch_overfetch_floor)``.

    The previous behavior was ``top_k * 2``, which starved the fusion stage
    for small ``top_k`` values (e.g. top_k=3 → only 6 candidates per branch).
    The new defaults (factor=4, floor=20) match the Anthropic cookbook's
    proven heuristic, ensuring at least 20 candidates per branch.
    """

    @pytest.mark.asyncio
    async def test_rank_fusion_uses_overfetch_floor_for_small_top_k(self):
        """For top_k=3, branch limit should be max(3*4, 20) = 20, not 6."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_rank_fusion,
        )

        collection, _ = _make_mock_collection()
        await hybrid_search_with_rank_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=3,
        )

        pipeline = collection.aggregate.call_args[0][0]
        rank_fusion_stage = next(s for s in pipeline if "$rankFusion" in s)
        branches = rank_fusion_stage["$rankFusion"]["input"]["pipelines"]

        # Check each branch's $limit or $vectorSearch.limit
        for branch_name, branch_pipeline in branches.items():
            limit_val = self._extract_branch_limit(branch_pipeline)
            assert limit_val == 20, (
                f"Branch '{branch_name}' limit should be 20 for top_k=3, "
                f"got {limit_val}"
            )

    @pytest.mark.asyncio
    async def test_score_fusion_uses_overfetch_floor_for_small_top_k(self):
        """For top_k=3, branch limit should be max(3*4, 20) = 20, not 6."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_score_fusion,
        )

        collection, _ = _make_mock_collection()
        await hybrid_search_with_score_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=3,
        )

        pipeline = collection.aggregate.call_args[0][0]
        score_fusion_stage = next(s for s in pipeline if "$scoreFusion" in s)
        branches = score_fusion_stage["$scoreFusion"]["input"]["pipelines"]

        for branch_name, branch_pipeline in branches.items():
            limit_val = self._extract_branch_limit(branch_pipeline)
            assert limit_val == 20, (
                f"Branch '{branch_name}' limit should be 20 for top_k=3, "
                f"got {limit_val}"
            )

    @pytest.mark.asyncio
    async def test_rank_fusion_overfetch_for_large_top_k(self):
        """For top_k=10, branch limit should be max(10*4, 20) = 40."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_rank_fusion,
        )

        collection, _ = _make_mock_collection()
        await hybrid_search_with_rank_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=10,
        )

        pipeline = collection.aggregate.call_args[0][0]
        rank_fusion_stage = next(s for s in pipeline if "$rankFusion" in s)
        branches = rank_fusion_stage["$rankFusion"]["input"]["pipelines"]

        for branch_name, branch_pipeline in branches.items():
            limit_val = self._extract_branch_limit(branch_pipeline)
            assert limit_val == 40, (
                f"Branch '{branch_name}' limit should be 40 for top_k=10, "
                f"got {limit_val}"
            )

    @pytest.mark.asyncio
    async def test_rank_fusion_overfetch_configurable(self):
        """Custom branch_overfetch_factor and floor should propagate."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
            hybrid_search_with_rank_fusion,
        )

        collection, _ = _make_mock_collection()
        config = MongoDBHybridSearchConfig(
            branch_overfetch_factor=3,
            branch_overfetch_floor=15,
        )
        await hybrid_search_with_rank_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=10,
            config=config,
        )

        pipeline = collection.aggregate.call_args[0][0]
        rank_fusion_stage = next(s for s in pipeline if "$rankFusion" in s)
        branches = rank_fusion_stage["$rankFusion"]["input"]["pipelines"]

        # max(10*3, 15) = 30
        for branch_name, branch_pipeline in branches.items():
            limit_val = self._extract_branch_limit(branch_pipeline)
            assert limit_val == 30, (
                f"Branch '{branch_name}' limit should be 30 for "
                f"factor=3, floor=15, top_k=10, got {limit_val}"
            )

    @pytest.mark.asyncio
    async def test_overfetch_floor_zero_preserves_old_behavior(self):
        """Setting factor=2, floor=0 should produce top_k * 2 (old behavior)."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
            hybrid_search_with_rank_fusion,
        )

        collection, _ = _make_mock_collection()
        config = MongoDBHybridSearchConfig(
            branch_overfetch_factor=2,
            branch_overfetch_floor=0,
        )
        await hybrid_search_with_rank_fusion(
            collection=collection,
            query_text="test query",
            query_vector=[0.1] * 1024,
            top_k=10,
            config=config,
        )

        pipeline = collection.aggregate.call_args[0][0]
        rank_fusion_stage = next(s for s in pipeline if "$rankFusion" in s)
        branches = rank_fusion_stage["$rankFusion"]["input"]["pipelines"]

        for branch_name, branch_pipeline in branches.items():
            limit_val = self._extract_branch_limit(branch_pipeline)
            assert limit_val == 20, (
                f"Branch '{branch_name}' limit should be 20 (10*2) for "
                f"factor=2, floor=0, top_k=10, got {limit_val}"
            )

    @pytest.mark.asyncio
    async def test_manual_rrf_uses_branch_limit(self):
        """manual_hybrid_search_with_rrf should use config.branch_limit for fetch_count."""
        from hybridrag.enhancements import mongodb_hybrid_search
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
        )

        # Capture the top_k passed to vector_only_search and text_only_search
        captured_top_ks: list[int] = []

        async def fake_vector_search(
            collection, query_vector, top_k, config=None, db=None, **kw
        ):
            captured_top_ks.append(top_k)
            return []

        async def fake_text_search(
            collection, query_text, top_k, config=None, db=None, **kw
        ):
            captured_top_ks.append(top_k)
            return []

        # Use monkeypatch via direct attribute set (restored by test isolation)
        orig_v = mongodb_hybrid_search.vector_only_search
        orig_t = mongodb_hybrid_search.text_only_search
        mongodb_hybrid_search.vector_only_search = fake_vector_search
        mongodb_hybrid_search.text_only_search = fake_text_search

        try:
            config = MongoDBHybridSearchConfig(
                branch_overfetch_factor=4,
                branch_overfetch_floor=20,
            )
            await mongodb_hybrid_search.manual_hybrid_search_with_rrf(
                collection=AsyncMock(),
                query_text="test query",
                query_vector=[0.1] * 1024,
                top_k=5,
                config=config,
            )
            # fetch_count should be max(5*4, 20) = 20
            assert all(t == 20 for t in captured_top_ks), (
                f"Expected fetch_count=20 for top_k=5 with factor=4, floor=20, "
                f"got {captured_top_ks}"
            )
        finally:
            mongodb_hybrid_search.vector_only_search = orig_v
            mongodb_hybrid_search.text_only_search = orig_t

    def test_config_rejects_invalid_overfetch_factor(self):
        """branch_overfetch_factor < 1 should raise."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
        )

        with pytest.raises(
            RetrievalValidationError, match="branch_overfetch_factor"
        ):
            MongoDBHybridSearchConfig(branch_overfetch_factor=0)

    def test_config_rejects_negative_overfetch_floor(self):
        """branch_overfetch_floor < 0 should raise."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
        )

        with pytest.raises(
            RetrievalValidationError, match="branch_overfetch_floor"
        ):
            MongoDBHybridSearchConfig(branch_overfetch_floor=-1)

    def test_branch_limit_method(self):
        """MongoDBHybridSearchConfig.branch_limit() should compute correctly."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
        )

        config = MongoDBHybridSearchConfig()
        assert config.branch_limit(3) == 20   # max(12, 20) = 20
        assert config.branch_limit(10) == 40  # max(40, 20) = 40
        assert config.branch_limit(5) == 20   # max(20, 20) = 20
        assert config.branch_limit(6) == 24   # max(24, 20) = 24

    @staticmethod
    def _extract_branch_limit(branch_pipeline: list) -> int:
        """Extract the effective $limit from a fusion input branch pipeline."""
        for stage in branch_pipeline:
            if "$limit" in stage:
                return stage["$limit"]
            if "$vectorSearch" in stage:
                return stage["$vectorSearch"].get("limit", 0)
            if "$search" in stage and "vectorSearch" in stage["$search"]:
                return stage["$search"]["vectorSearch"].get("limit", 0)
        return 0

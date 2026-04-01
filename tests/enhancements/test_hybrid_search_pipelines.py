"""Tests for MongoDB hybrid search pipeline correctness.

Validates CRITICAL fixes C5, C6, C7 from the MongoDB audit:
- C5: Mixed $project inclusion/exclusion (server error)
- C6: $scoreFusion wrong $meta field name
- C7: $scoreFusion normalization placement
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


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

"""
Tests for Phase 3C (Search & AI Correctness) and Phase 3D (Filters).

Covers findings: M10, M17, M18, M36, M37, M45, M46, M47, M19, M33.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# M10: Fallback chain catches OperationFailure specifically
# ---------------------------------------------------------------------------
class TestM10FallbackChainExceptionHandling:
    """M10: Fallback chain should catch OperationFailure specifically,
    not bare Exception. Programming errors must re-raise."""

    @pytest.mark.asyncio
    async def test_operation_failure_triggers_fallback(self):
        """OperationFailure from $rankFusion should fall back to manual RRF."""
        from pymongo.errors import OperationFailure

        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_rank_fusion,
        )

        mock_collection = AsyncMock()
        # Make aggregate raise OperationFailure (e.g., $rankFusion not supported)
        mock_collection.aggregate.side_effect = OperationFailure(
            "$rankFusion not supported"
        )
        mock_collection.database = MagicMock()

        with patch(
            "hybridrag.enhancements.mongodb_hybrid_search.manual_hybrid_search_with_rrf",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fallback:
            result = await hybrid_search_with_rank_fusion(
                collection=mock_collection,
                query_text="test query",
                query_vector=[0.1] * 1024,
                top_k=5,
            )
            # Should have called the fallback
            mock_fallback.assert_called_once()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_unexpected_error_reraises(self):
        """Non-OperationFailure exceptions (e.g., TypeError) must re-raise,
        not silently fall back."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            hybrid_search_with_rank_fusion,
        )

        mock_collection = AsyncMock()
        # Programming error -- should NOT be caught by fallback
        mock_collection.aggregate.side_effect = TypeError(
            "unexpected programming error"
        )
        mock_collection.database = MagicMock()

        with pytest.raises(TypeError, match="unexpected programming error"):
            await hybrid_search_with_rank_fusion(
                collection=mock_collection,
                query_text="test query",
                query_vector=[0.1] * 1024,
                top_k=5,
            )


# ---------------------------------------------------------------------------
# M17: search_paths nested list bug
# ---------------------------------------------------------------------------
class TestM17SearchPathsNestedList:
    """M17: When search_paths is None and config.text_search_path is a string,
    it should be wrapped in a list. When it's already a list, don't double-wrap."""

    @pytest.mark.asyncio
    async def test_search_paths_string_becomes_list(self):
        """String text_search_path wraps in single-element list."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
            text_only_search,
        )

        config = MongoDBHybridSearchConfig(text_search_path="content")
        mock_collection = AsyncMock()
        mock_collection.aggregate.return_value = AsyncMock()
        mock_collection.aggregate.return_value.__aiter__ = AsyncMock(
            return_value=iter([])
        )

        # Call with search_paths=None -- should resolve to ["content"]
        result = await text_only_search(
            collection=mock_collection,
            query_text="test",
            top_k=5,
            config=config,
            search_paths=None,
        )
        # Check the pipeline sent to aggregate
        call_args = mock_collection.aggregate.call_args
        assert call_args is not None, "aggregate not called"
        pipeline = call_args[0][0]
        search_stage = pipeline[0]
        search_body = search_stage.get("$search", {})
        compound = search_body.get("compound", {})
        must = compound.get("must", [])
        assert must, "No 'must' clauses in compound"
        text_clause = must[0].get("text", {})
        path = text_clause.get("path", [])
        # Path should be ["content"], not [["content"]]
        assert isinstance(path, list), f"Expected list, got {type(path)}"
        assert all(isinstance(p, str) for p in path), f"Nested list detected: {path}"

    @pytest.mark.asyncio
    async def test_search_paths_list_not_double_wrapped(self):
        """List text_search_path should not be double-wrapped."""
        from hybridrag.enhancements.mongodb_hybrid_search import (
            MongoDBHybridSearchConfig,
            text_only_search,
        )

        config = MongoDBHybridSearchConfig(text_search_path=["content", "title"])
        mock_collection = AsyncMock()
        mock_collection.aggregate.return_value = AsyncMock()
        mock_collection.aggregate.return_value.__aiter__ = AsyncMock(
            return_value=iter([])
        )

        result = await text_only_search(
            collection=mock_collection,
            query_text="test",
            top_k=5,
            config=config,
            search_paths=None,
        )
        call_args = mock_collection.aggregate.call_args
        assert call_args is not None, "aggregate not called"
        pipeline = call_args[0][0]
        search_stage = pipeline[0]
        search_body = search_stage.get("$search", {})
        compound = search_body.get("compound", {})
        must = compound.get("must", [])
        assert must, "No 'must' clauses in compound"
        text_clause = must[0].get("text", {})
        path = text_clause.get("path", [])
        assert isinstance(path, list)
        assert all(isinstance(p, str) for p in path), f"Nested list detected: {path}"
        assert path == ["content", "title"]


# ---------------------------------------------------------------------------
# M18: Atlas Search score boost syntax wrong
# ---------------------------------------------------------------------------
class TestM18AtlasSearchScoreBoost:
    """M18: Score boost should use {"boost": {"value": N}} format."""

    def test_score_boost_correct_format(self):
        """Atlas Search score boost should use simple numeric value format."""
        from hybridrag.enhancements.filters.atlas_search_filters import (
            build_compound_search_stage,
        )

        stage = build_compound_search_stage(
            index_name="test_index",
            query_text="test query",
            search_paths=["content"],
            path_weights={"content": 10},
        )
        text_clause = stage["$search"]["compound"]["must"][0]["text"]
        score = text_clause.get("score", {})
        boost = score.get("boost", {})
        # Must use {"value": N} format, not {"path": ..., "undefined": ...}
        assert "value" in boost, f"Expected 'value' key in boost, got: {boost}"
        assert "path" not in boost, f"Unexpected 'path' key in boost: {boost}"
        assert "undefined" not in boost, f"Unexpected 'undefined' key in boost: {boost}"
        assert isinstance(boost["value"], (int, float))


# ---------------------------------------------------------------------------
# M36: top_k allows 200 (too expensive)
# ---------------------------------------------------------------------------
class TestM36TopKLimit:
    """M36: top_k should be capped at 100, not 200."""

    def test_top_k_rejects_over_100(self):
        """top_k > 100 should be rejected by validation."""
        from pydantic import ValidationError

        from hybridrag.api.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=101)

    def test_top_k_accepts_100(self):
        """top_k = 100 should be accepted."""
        from hybridrag.api.models import QueryRequest

        req = QueryRequest(query="test", top_k=100)
        assert req.top_k == 100


# ---------------------------------------------------------------------------
# M37: 10K char query max exceeds model limits
# ---------------------------------------------------------------------------
class TestM37QueryMaxLength:
    """M37: Query max_length should be 5000, not 10000."""

    def test_query_rejects_over_5000_chars(self):
        """Queries over 5000 chars should be rejected."""
        from pydantic import ValidationError

        from hybridrag.api.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 5001)

    def test_query_accepts_5000_chars(self):
        """Queries up to 5000 chars should be accepted."""
        from hybridrag.api.models import QueryRequest

        req = QueryRequest(query="x" * 5000)
        assert len(req.query) == 5000


# ---------------------------------------------------------------------------
# M45: No rate limit/retry on Voyage AI calls
# ---------------------------------------------------------------------------
class TestM45VoyageRetry:
    """M45: Voyage AI embed calls should have tenacity retry for rate limits."""

    @pytest.mark.asyncio
    async def test_embed_retries_on_rate_limit(self):
        """embed_async should retry on rate limit errors."""
        from hybridrag.integrations.voyage import VoyageEmbedder

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.embedding_model = "voyage-4-large"
        embedder.batch_size = 128
        embedder._async_client = AsyncMock()

        # First call raises rate limit, second succeeds
        call_count = 0

        async def mock_embed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HTTP 429: Too Many Requests")
            mock_result = MagicMock()
            mock_result.embeddings = [[0.1] * 1024]
            return mock_result

        embedder._async_client.embed = mock_embed

        result = await embedder.embed_async(["test text"])
        assert call_count >= 2, f"Expected retry, but only {call_count} call(s)"
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# M46: Empty embedding array wrong shape
# ---------------------------------------------------------------------------
class TestM46EmptyEmbeddingShape:
    """M46: Empty embedding should return (0, 1024) not (0,)."""

    @pytest.mark.asyncio
    async def test_empty_texts_returns_2d_array(self):
        """Empty texts list should return 2D array with correct dimension."""
        from hybridrag.integrations.voyage import VoyageEmbedder

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.embedding_model = "voyage-4-large"
        embedder.batch_size = 128
        embedder._async_client = AsyncMock()

        result = await embedder.embed_async([])
        assert len(result.shape) == 2, f"Expected 2D array, got shape {result.shape}"
        assert result.shape[0] == 0
        assert result.shape[1] == 1024

    def test_empty_texts_sync_returns_2d_array(self):
        """Sync version: empty texts should also return 2D array."""
        from hybridrag.integrations.voyage import VoyageEmbedder

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.embedding_model = "voyage-4-large"
        embedder.batch_size = 128
        embedder._sync_client = MagicMock()

        result = embedder.embed_sync([])
        assert len(result.shape) == 2, f"Expected 2D array, got shape {result.shape}"
        assert result.shape[0] == 0
        assert result.shape[1] == 1024

    @pytest.mark.asyncio
    async def test_contextualized_empty_returns_2d_array(self):
        """Contextualized embed with empty input should return 2D array."""
        from hybridrag.integrations.voyage import VoyageEmbedder

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.context_model = "voyage-context-3"
        embedder._async_client = AsyncMock()

        result = await embedder.embed_contextualized_async([])
        assert len(result.shape) == 2, f"Expected 2D array, got shape {result.shape}"
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# M47: Search presets not wired to $rankFusion
# ---------------------------------------------------------------------------
class TestM47PresetsRankFusionWeights:
    """M47: Search presets should include rank_fusion_weights dict."""

    def test_presets_have_rank_fusion_weights(self):
        """Each preset should have a rank_fusion_weights dict."""
        from hybridrag.presets.search_presets import PRESETS

        for name, preset in PRESETS.items():
            assert hasattr(preset, "rank_fusion_weights"), (
                f"Preset '{name}' missing rank_fusion_weights attribute"
            )
            weights = preset.rank_fusion_weights
            assert isinstance(weights, dict), (
                f"Preset '{name}' rank_fusion_weights should be dict, got {type(weights)}"
            )

    def test_rank_fusion_weights_have_vector_and_text(self):
        """rank_fusion_weights should have vector and text pipeline entries."""
        from hybridrag.presets.search_presets import PRESETS

        for name, preset in PRESETS.items():
            weights = preset.rank_fusion_weights
            assert "vector" in weights, (
                f"Preset '{name}' missing 'vector' key in rank_fusion_weights"
            )
            assert "text" in weights, (
                f"Preset '{name}' missing 'text' key in rank_fusion_weights"
            )

    def test_rank_fusion_weights_match_preset_weights(self):
        """rank_fusion_weights values should match vector_weight/text_weight."""
        from hybridrag.presets.search_presets import PRESETS

        for name, preset in PRESETS.items():
            weights = preset.rank_fusion_weights
            assert weights["vector"] == preset.vector_weight, (
                f"Preset '{name}': vector weight mismatch {weights['vector']} != {preset.vector_weight}"
            )
            assert weights["text"] == preset.text_weight, (
                f"Preset '{name}': text weight mismatch {weights['text']} != {preset.text_weight}"
            )


# ---------------------------------------------------------------------------
# M19: Filter collision -- in_filters overwrites equality_filters
# ---------------------------------------------------------------------------
class TestM19FilterCollision:
    """M19: in_filters should merge with equality_filters, not overwrite."""

    def test_in_filter_merges_with_equality_filter(self):
        """When same field has equality and in_filter, values should merge."""
        from hybridrag.enhancements.filters.vector_search_filters import (
            VectorSearchFilterConfig,
            build_vector_search_filters,
        )

        config = VectorSearchFilterConfig(
            equality_filters={"category": "tech"},
            in_filters={"category": ["science", "health"]},
        )
        result = build_vector_search_filters(config)
        # The "category" filter should be $in with all three values
        assert "category" in result
        cat_filter = result["category"]
        assert "$in" in cat_filter, f"Expected $in operator, got: {cat_filter}"
        values = cat_filter["$in"]
        assert "tech" in values, f"Equality value 'tech' lost in merge: {values}"
        assert "science" in values, f"In-filter value 'science' lost: {values}"
        assert "health" in values, f"In-filter value 'health' lost: {values}"

    def test_in_filter_without_equality_works_normally(self):
        """in_filter without matching equality_filter should work as before."""
        from hybridrag.enhancements.filters.vector_search_filters import (
            VectorSearchFilterConfig,
            build_vector_search_filters,
        )

        config = VectorSearchFilterConfig(
            in_filters={"category": ["tech", "science"]},
        )
        result = build_vector_search_filters(config)
        assert result["category"] == {"$in": ["tech", "science"]}

    def test_equality_filter_without_in_works_normally(self):
        """equality_filter without matching in_filter should work as before."""
        from hybridrag.enhancements.filters.vector_search_filters import (
            VectorSearchFilterConfig,
            build_vector_search_filters,
        )

        config = VectorSearchFilterConfig(
            equality_filters={"category": "tech"},
        )
        result = build_vector_search_filters(config)
        assert result["category"] == {"$eq": "tech"}


# ---------------------------------------------------------------------------
# M33: doc_id not validated as ObjectId
# ---------------------------------------------------------------------------
class TestM33DocIdValidation:
    """M33: doc_id must be validated as a valid ObjectId format."""

    @pytest.mark.asyncio
    async def test_invalid_doc_id_returns_400(self):
        """Invalid ObjectId format should return 400."""
        from fastapi.testclient import TestClient

        from hybridrag.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.delete("/v1/documents/not-a-valid-objectid")
        assert response.status_code == 400, (
            f"Expected 400 for invalid ObjectId, got {response.status_code}"
        )

    @pytest.mark.asyncio
    async def test_valid_objectid_format_passes_validation(self):
        """Valid ObjectId format should pass validation (may fail on other grounds)."""
        from fastapi.testclient import TestClient

        from hybridrag.api.main import create_app

        app = create_app()
        client = TestClient(app)
        # Valid ObjectId format but document won't exist
        response = client.delete("/v1/documents/507f1f77bcf86cd799439011")
        # Should not be 400 (validation passed) -- likely 503 since RAG not initialized
        assert response.status_code != 400, "Valid ObjectId rejected with 400"

"""Tests for vector search filter builder."""

from datetime import datetime

from hybridrag.enhancements.filters.vector_search_filters import (
    VectorSearchFilterConfig,
    build_vector_search_filters,
)


class TestBuildVectorSearchFilters:
    """Test vector search filter builder with standard MongoDB operators."""

    def test_empty_config_returns_empty_dict(self):
        """Empty config should return empty filter dict."""
        config = VectorSearchFilterConfig()
        result = build_vector_search_filters(config)
        assert result == {}

    def test_date_range_filter(self):
        """Date range uses standard $gte/$lte operators."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        config = VectorSearchFilterConfig(
            start_date=start, end_date=end, timestamp_field="timestamp"
        )
        result = build_vector_search_filters(config)

        # Vector search uses STANDARD MongoDB operators
        assert "timestamp" in result
        assert result["timestamp"]["$gte"] == start
        assert result["timestamp"]["$lte"] == end

    def test_equality_filter(self):
        """Equality uses standard $eq operator."""
        config = VectorSearchFilterConfig(
            equality_filters={"senderName": "John", "status": "active"}
        )
        result = build_vector_search_filters(config)

        # Vector search uses STANDARD MongoDB operators
        assert result["senderName"]["$eq"] == "John"
        assert result["status"]["$eq"] == "active"

    def test_in_filter(self):
        """In-list uses standard $in operator."""
        config = VectorSearchFilterConfig(in_filters={"category": ["tech", "science"]})
        result = build_vector_search_filters(config)

        assert result["category"]["$in"] == ["tech", "science"]

    def test_equality_and_membership_on_same_field_remain_conjunctive(self):
        config = VectorSearchFilterConfig(
            equality_filters={"tenant": "tenant-a"},
            in_filters={"tenant": ["tenant-b"]},
        )

        result = build_vector_search_filters(config)

        assert result["tenant"] == {
            "$eq": "tenant-a",
            "$in": ["tenant-b"],
        }

    def test_combined_filters(self):
        """Multiple filter types combine correctly."""
        start = datetime(2024, 1, 1)
        config = VectorSearchFilterConfig(
            start_date=start,
            timestamp_field="created_at",
            equality_filters={"source": "api"},
            in_filters={"tags": ["urgent", "priority"]},
        )
        result = build_vector_search_filters(config)

        assert "created_at" in result
        assert result["created_at"]["$gte"] == start
        assert result["source"]["$eq"] == "api"
        assert result["tags"]["$in"] == ["urgent", "priority"]

    def test_nested_field_path_equality(self):
        """L24: Dotted paths like 'metadata.category' should work in equality filters."""
        config = VectorSearchFilterConfig(
            equality_filters={"metadata.category": "features"}
        )
        result = build_vector_search_filters(config)

        assert "metadata.category" in result
        assert result["metadata.category"]["$eq"] == "features"

    def test_nested_field_path_in_filter(self):
        """L24: Dotted paths should work in $in filters."""
        config = VectorSearchFilterConfig(
            in_filters={"metadata.source": ["docs", "api"]}
        )
        result = build_vector_search_filters(config)

        assert "metadata.source" in result
        assert result["metadata.source"]["$in"] == ["docs", "api"]

    def test_nested_field_path_range(self):
        """L24: Dotted paths should work in range filters via timestamp_field."""
        start = datetime(2024, 1, 1)
        config = VectorSearchFilterConfig(
            start_date=start,
            timestamp_field="metadata.timestamp",
        )
        result = build_vector_search_filters(config)

        assert "metadata.timestamp" in result
        assert result["metadata.timestamp"]["$gte"] == start

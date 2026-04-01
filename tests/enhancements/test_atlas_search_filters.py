"""Tests for Atlas Search filter builder."""

from datetime import datetime

from hybridrag.enhancements.filters.atlas_search_filters import (
    AtlasSearchFilterConfig,
    build_atlas_search_filters,
)


class TestBuildAtlasSearchFilters:
    """Test Atlas Search filter builder with Atlas-specific operators."""

    def test_empty_config_returns_empty_list(self) -> None:
        """Empty config should return empty filter list."""
        config = AtlasSearchFilterConfig()
        result = build_atlas_search_filters(config)
        assert result == []

    def test_date_range_filter(self) -> None:
        """Date range uses Atlas 'range' operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        config = AtlasSearchFilterConfig(
            start_date=start, end_date=end, timestamp_field="timestamp"
        )
        result = build_atlas_search_filters(config)

        # Atlas Search uses 'range' operator (NOT $gte/$lte)
        assert len(result) == 1
        assert "range" in result[0]
        assert result[0]["range"]["path"] == "timestamp"
        assert result[0]["range"]["gte"] == start
        assert result[0]["range"]["lte"] == end

    def test_equality_filter(self) -> None:
        """Equality uses Atlas 'equals' operator."""
        config = AtlasSearchFilterConfig(
            equality_filters={"senderName": "John", "status": "active"}
        )
        result = build_atlas_search_filters(config)

        # Atlas Search uses 'equals' operator (NOT $eq)
        assert len(result) == 2

        sender_filter = next(
            f for f in result if f.get("equals", {}).get("path") == "senderName"
        )
        assert sender_filter["equals"]["value"] == "John"

        status_filter = next(
            f for f in result if f.get("equals", {}).get("path") == "status"
        )
        assert status_filter["equals"]["value"] == "active"

    def test_in_filter(self) -> None:
        """In-list uses multiple 'equals' with compound 'should'."""
        config = AtlasSearchFilterConfig(in_filters={"category": ["tech", "science"]})
        result = build_atlas_search_filters(config)

        # Atlas Search uses compound should for in-list
        assert len(result) == 1
        assert "compound" in result[0]
        assert "should" in result[0]["compound"]

        should_clauses = result[0]["compound"]["should"]
        assert len(should_clauses) == 2
        values = [c["equals"]["value"] for c in should_clauses]
        assert "tech" in values
        assert "science" in values

    def test_nested_field_path_equality(self) -> None:
        """L24: Dotted paths like 'metadata.source' should work in equality filters."""
        config = AtlasSearchFilterConfig(equality_filters={"metadata.source": "docs"})
        result = build_atlas_search_filters(config)

        assert len(result) == 1
        assert "equals" in result[0]
        assert result[0]["equals"]["path"] == "metadata.source"
        assert result[0]["equals"]["value"] == "docs"

    def test_nested_field_path_range(self) -> None:
        """L24: Dotted paths should work in range filters."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        config = AtlasSearchFilterConfig(
            start_date=start,
            end_date=end,
            timestamp_field="metadata.timestamp",
        )
        result = build_atlas_search_filters(config)

        assert len(result) == 1
        assert "range" in result[0]
        assert result[0]["range"]["path"] == "metadata.timestamp"
        assert result[0]["range"]["gte"] == start
        assert result[0]["range"]["lte"] == end

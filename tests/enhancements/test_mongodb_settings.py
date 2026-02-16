"""Tests for MongoDB configuration settings."""

from hybridrag.config.settings import Settings


class TestMongoDBSettings:
    """Test MongoDB-specific settings defaults and validation."""

    def test_default_pool_settings(self):
        """Verify pool defaults match current behavior (no pool config = pymongo defaults)."""
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_max_pool_size == 100
        assert settings.mongodb_min_pool_size == 0
        assert settings.mongodb_max_idle_time_ms == 60000

    def test_default_concern_settings(self):
        """Verify concern defaults match current implicit behavior."""
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_read_concern == "local"
        assert settings.mongodb_write_concern == "1"

    def test_default_query_validation(self):
        """Verify query length limit exists."""
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.max_query_length == 10000

    def test_default_aggregate_timeout(self):
        """Verify aggregate timeout exists."""
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_aggregate_timeout_ms == 30000

    def test_custom_pool_settings(self):
        """Verify custom pool settings are accepted."""
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
            mongodb_max_pool_size=50,
            mongodb_min_pool_size=5,
            mongodb_max_idle_time_ms=30000,
        )
        assert settings.mongodb_max_pool_size == 50
        assert settings.mongodb_min_pool_size == 5
        assert settings.mongodb_max_idle_time_ms == 30000

    def test_custom_concern_settings(self):
        """Verify custom concern settings are accepted."""
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
            mongodb_read_concern="majority",
            mongodb_write_concern="majority",
        )
        assert settings.mongodb_read_concern == "majority"
        assert settings.mongodb_write_concern == "majority"

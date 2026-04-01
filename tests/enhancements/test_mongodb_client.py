"""Tests for unified MongoDB client factory."""

from unittest.mock import MagicMock, patch

from hybridrag.core.mongodb_client import (
    create_motor_client,
    get_database,
)


class TestCreateMotorClient:
    """Test client creation with pool settings."""

    @patch("hybridrag.core.mongodb_client.AsyncMongoClient")
    def test_default_pool_settings(self, mock_client_cls):
        """Verify default pool settings are passed to pymongo.AsyncMongoClient."""
        create_motor_client("mongodb://localhost:27017")
        mock_client_cls.assert_called_once_with(
            "mongodb://localhost:27017",
            maxPoolSize=100,
            minPoolSize=0,
            maxIdleTimeMS=60000,
        )

    @patch("hybridrag.core.mongodb_client.AsyncMongoClient")
    def test_custom_pool_settings(self, mock_client_cls):
        """Verify custom pool settings are forwarded."""
        create_motor_client(
            "mongodb://localhost:27017",
            max_pool_size=50,
            min_pool_size=5,
            max_idle_time_ms=30000,
        )
        mock_client_cls.assert_called_once_with(
            "mongodb://localhost:27017",
            maxPoolSize=50,
            minPoolSize=5,
            maxIdleTimeMS=30000,
        )


class TestGetDatabase:
    """Test database creation with read/write concerns."""

    def test_default_concerns(self):
        """Verify default concerns match current behavior (local, w=1)."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        # Chain mock returns
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        get_database(mock_client, "testdb")

        # Should call with_options twice (read_concern, write_concern)
        assert mock_db.with_options.call_count == 2

    def test_majority_write_concern(self):
        """Verify majority write concern is applied."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        get_database(
            mock_client,
            "testdb",
            read_concern="majority",
            write_concern="majority",
        )

        assert mock_db.with_options.call_count == 2

    def test_default_concerns_verify_values(self):
        """L19: Verify actual ReadConcern/WriteConcern values, not just call counts."""
        from pymongo.read_concern import ReadConcern
        from pymongo.write_concern import WriteConcern

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        get_database(mock_client, "testdb")

        # First call should be read_concern="local"
        first_call = mock_db.with_options.call_args_list[0]
        rc = first_call.kwargs.get("read_concern") or first_call[1].get("read_concern")
        assert isinstance(rc, ReadConcern)
        assert rc.level == "local"

        # Second call should be write_concern=w:1
        second_call = mock_db.with_options.call_args_list[1]
        wc = second_call.kwargs.get("write_concern") or second_call[1].get(
            "write_concern"
        )
        assert isinstance(wc, WriteConcern)
        assert wc.document.get("w") == 1

    def test_majority_concerns_verify_values(self):
        """L19: Verify majority concerns pass correct objects."""
        from pymongo.read_concern import ReadConcern
        from pymongo.write_concern import WriteConcern

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        get_database(
            mock_client,
            "testdb",
            read_concern="majority",
            write_concern="majority",
        )

        # First call: read_concern="majority"
        first_call = mock_db.with_options.call_args_list[0]
        rc = first_call.kwargs.get("read_concern") or first_call[1].get("read_concern")
        assert isinstance(rc, ReadConcern)
        assert rc.level == "majority"

        # Second call: write_concern="majority" with wtimeout
        second_call = mock_db.with_options.call_args_list[1]
        wc = second_call.kwargs.get("write_concern") or second_call[1].get(
            "write_concern"
        )
        assert isinstance(wc, WriteConcern)
        assert wc.document.get("w") == "majority"
        assert wc.document.get("wtimeout") == 10000

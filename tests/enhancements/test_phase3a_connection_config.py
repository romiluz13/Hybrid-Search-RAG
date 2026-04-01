"""Tests for Phase 3A: Connection Config Hardening (M1-M9, M39, M40, M42).

Tests for 12 MEDIUM findings. 4 already done (M5, M8, M39, M40) -- verified only.
8 require new fixes (M1, M2, M3, M4, M6, M7, M9, M42).

Backing Skill: mongodb-connection
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from pymongo.errors import OperationFailure

# ── M1: Default read_concern="majority" + write_concern="majority" ────────


class TestM1DurableDefaults:
    """M1: Default read/write concerns must be 'majority' for durability."""

    def test_default_read_concern_is_majority(self):
        """Read concern default should be 'majority' for production durability."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_read_concern == "majority"

    def test_default_write_concern_is_majority(self):
        """Write concern default should be 'majority' for production durability."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_write_concern == "majority"


# ── M2: URI format validation ─────────────────────────────────────────────


class TestM2URIValidation:
    """M2: MongoDB URI must be validated to start with mongodb:// or mongodb+srv://."""

    def test_valid_mongodb_uri_accepted(self):
        """Standard mongodb:// URI should be accepted."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_uri.get_secret_value() == "mongodb://localhost:27017"

    def test_valid_mongodb_srv_uri_accepted(self):
        """SRV mongodb+srv:// URI should be accepted."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb+srv://cluster.example.net",
            voyage_api_key="test-key",
        )
        assert (
            settings.mongodb_uri.get_secret_value()
            == "mongodb+srv://cluster.example.net"
        )

    def test_invalid_uri_rejected(self):
        """Non-MongoDB URI should be rejected with ValidationError."""
        from hybridrag.config.settings import Settings

        with pytest.raises(ValidationError, match="MongoDB URI must start with"):
            Settings(
                mongodb_uri="http://localhost:27017",
                voyage_api_key="test-key",
            )

    def test_empty_scheme_uri_rejected(self):
        """URI without scheme should be rejected."""
        from hybridrag.config.settings import Settings

        with pytest.raises(ValidationError, match="MongoDB URI must start with"):
            Settings(
                mongodb_uri="localhost:27017",
                voyage_api_key="test-key",
            )


# ── M3: clear_settings_cache() helper ─────────────────────────────────────


class TestM3ClearSettingsCache:
    """M3: A clear_settings_cache() helper must exist for test isolation."""

    def test_clear_settings_cache_exists(self):
        """clear_settings_cache function must be importable."""
        from hybridrag.config.settings import clear_settings_cache

        # Should be callable
        assert callable(clear_settings_cache)

    def test_clear_settings_cache_resets_lru_cache(self):
        """Clearing cache should allow fresh Settings instantiation."""
        from hybridrag.config.settings import clear_settings_cache, get_settings

        # Call get_settings to populate cache
        try:
            get_settings()
        except Exception:
            pass  # May fail without .env, that's ok

        # Should not raise
        clear_settings_cache()


# ── M4: TLS settings exposed ──────────────────────────────────────────────


class TestM4TLSSettings:
    """M4: TLS configuration fields must exist on Settings."""

    def test_tls_field_exists(self):
        """mongodb_tls field should exist on Settings."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert hasattr(settings, "mongodb_tls")

    def test_tls_default_is_false(self):
        """TLS should default to False (most dev environments)."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_tls is False

    def test_tls_allow_invalid_certs_field_exists(self):
        """mongodb_tls_allow_invalid_certificates field should exist."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert hasattr(settings, "mongodb_tls_allow_invalid_certificates")

    def test_tls_allow_invalid_certs_default_is_false(self):
        """Allow invalid certs should default to False (secure default)."""
        from hybridrag.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            voyage_api_key="test-key",
        )
        assert settings.mongodb_tls_allow_invalid_certificates is False


# ── M5: appName parameter (ALREADY DONE -- verify) ────────────────────────


class TestM5AppNameVerify:
    """M5: Verify appName='hybridrag' is already configured (done in Phase 1A)."""

    @patch("hybridrag.core.mongodb_client.AsyncMongoClient")
    def test_shared_client_has_app_name(self, mock_client_cls):
        """get_shared_client must pass appName='hybridrag'."""
        from hybridrag.core.mongodb_client import (
            _reset_shared_client,
            get_shared_client,
        )

        _reset_shared_client()
        mock_settings = MagicMock()
        mock_settings.mongodb_uri.get_secret_value.return_value = (
            "mongodb://localhost:27017"
        )
        mock_settings.mongodb_max_pool_size = 100
        mock_settings.mongodb_min_pool_size = 0
        mock_settings.mongodb_max_idle_time_ms = 60000
        mock_settings.mongodb_server_selection_timeout_ms = 5000
        mock_settings.mongodb_connect_timeout_ms = 10000

        get_shared_client(mock_settings)

        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["appName"] == "hybridrag"
        _reset_shared_client()


# ── M6: write_concern validation in get_database ──────────────────────────


class TestM6WriteConcernValidation:
    """M6: get_database must validate write_concern parameter."""

    def test_valid_write_concern_accepted(self):
        """Valid write concern values should work."""
        from hybridrag.core.mongodb_client import get_database

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        # These should not raise
        get_database(mock_client, "testdb", write_concern="0")
        get_database(mock_client, "testdb", write_concern="1")
        get_database(mock_client, "testdb", write_concern="majority")

    def test_invalid_write_concern_rejected(self):
        """Invalid write concern value should raise ValueError."""
        from hybridrag.core.mongodb_client import get_database

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        with pytest.raises(ValueError, match="Invalid write_concern"):
            get_database(mock_client, "testdb", write_concern="invalid")

    def test_numeric_write_concern_rejected(self):
        """Numeric string like '2' should be rejected."""
        from hybridrag.core.mongodb_client import get_database

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        with pytest.raises(ValueError, match="Invalid write_concern"):
            get_database(mock_client, "testdb", write_concern="2")


# ── M7: wtimeout on WriteConcern(w="majority") ───────────────────────────


class TestM7WtimeoutOnMajority:
    """M7: WriteConcern(w='majority') must include wtimeout=10000."""

    def test_majority_write_concern_has_wtimeout(self):
        """When write_concern='majority', WriteConcern must have wtimeout=10000."""
        from hybridrag.core.mongodb_client import get_database

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        get_database(mock_client, "testdb", write_concern="majority")

        # Find the write_concern call -- it's the second with_options call
        calls = mock_db.with_options.call_args_list
        # The write concern call should contain wtimeout
        wc_found = False
        for call in calls:
            _, kwargs = call
            if "write_concern" in kwargs:
                wc = kwargs["write_concern"]
                assert wc.document.get("w") == "majority"
                assert wc.document.get("wtimeout") == 10000
                wc_found = True
        assert wc_found, (
            "WriteConcern with w='majority' not found in with_options calls"
        )

    def test_non_majority_write_concern_no_wtimeout(self):
        """When write_concern='1', no wtimeout should be set."""
        from hybridrag.core.mongodb_client import get_database

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.with_options = MagicMock(return_value=mock_db)

        get_database(mock_client, "testdb", write_concern="1")

        calls = mock_db.with_options.call_args_list
        for call in calls:
            _, kwargs = call
            if "write_concern" in kwargs:
                wc = kwargs["write_concern"]
                # w=1 should not have wtimeout
                assert wc.document.get("wtimeout") is None


# ── M8: Error code checking (ALREADY DONE -- verify) ─────────────────────


class TestM8ErrorCodeCheckingVerify:
    """M8: Verify error code checking is in place (done in Phase 2A)."""

    def test_transaction_not_supported_codes_defined(self):
        """TRANSACTION_NOT_SUPPORTED_CODES must be defined with expected error codes."""
        from hybridrag.core.transaction_helper import TRANSACTION_NOT_SUPPORTED_CODES

        assert 20 in TRANSACTION_NOT_SUPPORTED_CODES
        assert 263 in TRANSACTION_NOT_SUPPORTED_CODES
        assert 13435 in TRANSACTION_NOT_SUPPORTED_CODES

    @pytest.mark.asyncio
    async def test_fallback_uses_error_code(self):
        """Transaction fallback should check error codes, not just strings."""
        from hybridrag.core.transaction_helper import run_with_transaction

        call_count = 0

        async def my_callback(session=None):
            nonlocal call_count
            call_count += 1
            return "success"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # Create OperationFailure with error code 20 (IllegalOperation)
        error = OperationFailure("test error", code=20)
        mock_session.with_transaction = AsyncMock(side_effect=error)
        mock_client.start_session = AsyncMock(return_value=mock_session)

        result = await run_with_transaction(mock_client, my_callback)
        assert result == "success"


# ── M9: Read/write concern on transaction sessions ────────────────────────


class TestM9TransactionConcerns:
    """M9: Transactions must pass read_concern='snapshot' and write_concern='majority'."""

    @pytest.mark.asyncio
    async def test_with_transaction_passes_read_concern(self):
        """with_transaction must receive read_concern=ReadConcern('snapshot')."""
        from hybridrag.core.transaction_helper import run_with_transaction

        async def my_callback(session=None):
            return "success"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # Capture the actual call to with_transaction
        actual_kwargs = {}

        async def capture_with_transaction(callback, **kwargs):
            actual_kwargs.update(kwargs)
            await callback(mock_session)

        mock_session.with_transaction = capture_with_transaction
        mock_client.start_session = AsyncMock(return_value=mock_session)

        await run_with_transaction(mock_client, my_callback)

        assert "read_concern" in actual_kwargs, (
            "read_concern not passed to with_transaction"
        )
        assert actual_kwargs["read_concern"].level == "snapshot"

    @pytest.mark.asyncio
    async def test_with_transaction_passes_write_concern(self):
        """with_transaction must receive write_concern=WriteConcern('majority')."""
        from hybridrag.core.transaction_helper import run_with_transaction

        async def my_callback(session=None):
            return "success"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        actual_kwargs = {}

        async def capture_with_transaction(callback, **kwargs):
            actual_kwargs.update(kwargs)
            await callback(mock_session)

        mock_session.with_transaction = capture_with_transaction
        mock_client.start_session = AsyncMock(return_value=mock_session)

        await run_with_transaction(mock_client, my_callback)

        assert "write_concern" in actual_kwargs, (
            "write_concern not passed to with_transaction"
        )
        assert actual_kwargs["write_concern"].document.get("w") == "majority"


# ── M39: Motor version floor (ALREADY DONE -- verify) ────────────────────


class TestM39MotorVersionVerify:
    """M39: Verify motor>=3.4.0 is specified in pyproject.toml."""

    def test_motor_version_floor(self):
        """pyproject.toml must have motor>=3.4.0."""
        with open("pyproject.toml") as f:
            content = f.read()
        assert "motor>=3.4.0" in content


# ── M40: Upper bounds on deps (ALREADY DONE -- verify) ───────────────────


class TestM40DepUpperBoundsVerify:
    """M40: Verify pymongo and motor have upper version bounds."""

    def test_pymongo_has_upper_bound(self):
        """pyproject.toml must have pymongo<5.0."""
        with open("pyproject.toml") as f:
            content = f.read()
        assert "pymongo>=4.7.0,<5.0" in content

    def test_motor_has_upper_bound(self):
        """pyproject.toml must have motor<4.0."""
        with open("pyproject.toml") as f:
            content = f.read()
        assert "motor>=3.4.0,<4.0" in content


# ── M42: Makefile atlas-check with timeouts ───────────────────────────────


class TestM42MakefileTimeouts:
    """M42: Makefile atlas-check must include serverSelectionTimeoutMS and connectTimeoutMS."""

    def test_makefile_has_server_selection_timeout(self):
        """atlas-check command must include serverSelectionTimeoutMS."""
        with open("Makefile") as f:
            content = f.read()
        assert "serverSelectionTimeoutMS" in content

    def test_makefile_has_connect_timeout(self):
        """atlas-check command must include connectTimeoutMS."""
        with open("Makefile") as f:
            content = f.read()
        assert "connectTimeoutMS" in content

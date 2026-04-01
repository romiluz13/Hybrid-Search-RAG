"""Tests for Phase 1A: Connection Layer Remediation (C1, C2, C3, C4, C13).

Validates:
- C1: No URI leak to os.environ
- C2: Singleton shared client (pymongo.AsyncMongoClient, not Motor)
- C3: No client.close() in ingest methods (shared client lifecycle)
- C4: ClientManager stores + closes client ref on ref_count=0
- C13: pymongo version floor bumped
- Motor migration: No AsyncIOMotorClient in production source
"""

import inspect
from unittest.mock import MagicMock, patch

# ── C1: No URI leak to os.environ ──────────────────────────────────────────


class TestC1NoURILeak:
    """C1: os.environ['MONGO_URI'] must not appear in rag.py initialize()."""

    def test_no_os_environ_mongo_uri_in_initialize(self):
        """initialize() must not set os.environ['MONGO_URI']."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.initialize)
        # Should NOT contain os.environ["MONGO_URI"] assignment
        assert 'os.environ["MONGO_URI"]' not in source, (
            "C1 VIOLATION: os.environ['MONGO_URI'] still set in initialize(). "
            "URI must not leak to environment variables."
        )

    def test_no_os_environ_mongo_database_in_initialize(self):
        """initialize() must not set os.environ['MONGO_DATABASE']."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.initialize)
        assert 'os.environ["MONGO_DATABASE"]' not in source, (
            "C1 VIOLATION: os.environ['MONGO_DATABASE'] still set in initialize(). "
            "Database name must not leak to environment variables."
        )


# ── C2: Singleton shared client ────────────────────────────────────────────


class TestC2SingletonClient:
    """C2: All code paths must use get_shared_client, not create_motor_client."""

    def test_get_shared_client_exists(self):
        """mongodb_client.py must export get_shared_client()."""
        from hybridrag.core.mongodb_client import get_shared_client

        assert callable(get_shared_client), (
            "C2 VIOLATION: get_shared_client() not found in mongodb_client.py"
        )

    def test_close_shared_client_exists(self):
        """mongodb_client.py must export close_shared_client()."""
        from hybridrag.core.mongodb_client import close_shared_client

        assert callable(close_shared_client), (
            "C2 VIOLATION: close_shared_client() not found in mongodb_client.py"
        )

    def test_shared_client_returns_pymongo_async(self):
        """get_shared_client must return a pymongo.AsyncMongoClient, not Motor."""
        from hybridrag.core import mongodb_client as mc_module

        source = inspect.getsource(mc_module)
        # The module should import from pymongo, not motor
        assert "pymongo" in source, (
            "mongodb_client.py must use pymongo.AsyncMongoClient"
        )

    def test_ingest_files_uses_shared_client(self):
        """rag.py ingest_files must use get_shared_client, not create_motor_client_from_settings."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_files)
        assert "create_motor_client_from_settings" not in source, (
            "C2 VIOLATION: ingest_files still creates its own client. "
            "Must use get_shared_client() instead."
        )

    def test_ingest_url_uses_shared_client(self):
        """rag.py ingest_url must use get_shared_client, not create_motor_client_from_settings."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_url)
        assert "create_motor_client_from_settings" not in source, (
            "C2 VIOLATION: ingest_url still creates its own client. "
            "Must use get_shared_client() instead."
        )

    def test_ingest_website_uses_shared_client(self):
        """rag.py ingest_website must use get_shared_client, not create_motor_client_from_settings."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_website)
        assert "create_motor_client_from_settings" not in source, (
            "C2 VIOLATION: ingest_website still creates its own client. "
            "Must use get_shared_client() instead."
        )

    def test_singleton_is_reused(self):
        """Calling get_shared_client twice with same settings returns same instance."""
        from hybridrag.core.mongodb_client import (
            _reset_shared_client,
            get_shared_client,
        )

        # Reset state
        _reset_shared_client()

        settings = MagicMock()
        settings.mongodb_uri.get_secret_value.return_value = "mongodb://localhost:27017"
        settings.mongodb_max_pool_size = 100
        settings.mongodb_min_pool_size = 0
        settings.mongodb_max_idle_time_ms = 60000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client1 = get_shared_client(settings)
            client2 = get_shared_client(settings)

            assert client1 is client2, (
                "C2 VIOLATION: get_shared_client() returned different instances. "
                "Must return the same singleton."
            )
            # Constructor called only once
            assert mock_cls.call_count == 1, (
                "C2 VIOLATION: AsyncMongoClient created more than once"
            )

        _reset_shared_client()

    def test_shared_client_uses_pymongo_async_mongo_client(self):
        """get_shared_client must use pymongo.AsyncMongoClient, NOT Motor."""
        from hybridrag.core import mongodb_client as mc_module

        source = inspect.getsource(mc_module.get_shared_client)
        # Should not reference Motor
        assert "AsyncIOMotorClient" not in source, (
            "C2 VIOLATION: get_shared_client still uses Motor AsyncIOMotorClient"
        )


# ── C3: No client.close() in ingest methods ───────────────────────────────


class TestC3NoClientCloseInIngest:
    """C3: Ingest methods must not close the shared client."""

    def test_ingest_url_no_client_close(self):
        """ingest_url must not call client.close()."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_url)
        assert "client.close()" not in source, (
            "C3 VIOLATION: ingest_url still calls client.close(). "
            "Shared client lifecycle is managed globally."
        )

    def test_ingest_website_no_client_close(self):
        """ingest_website must not call client.close()."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_website)
        assert "client.close()" not in source, (
            "C3 VIOLATION: ingest_website still calls client.close(). "
            "Shared client lifecycle is managed globally."
        )


# ── C4: ClientManager stores + closes client ref ──────────────────────────


class TestC4ClientManagerClosesClient:
    """C4: ClientManager must store client ref and close it on ref_count=0."""

    def test_client_manager_stores_client_ref(self):
        """ClientManager._instances must have a 'client' key."""
        from hybridrag.engine.kg.mongo_impl import ClientManager

        assert "client" in ClientManager._instances, (
            "C4 VIOLATION: ClientManager._instances missing 'client' key. "
            "Must store client reference for proper cleanup."
        )

    def test_release_client_closes_on_zero_refs(self):
        """release_client must close client when ref_count drops to 0."""
        from hybridrag.engine.kg.mongo_impl import ClientManager

        source = inspect.getsource(ClientManager.release_client)
        # Must contain a close() call
        assert ".close()" in source, (
            "C4 VIOLATION: release_client does not call client.close(). "
            "Client must be closed when ref_count reaches 0."
        )

    def test_get_client_accepts_uri_parameter(self):
        """ClientManager.get_client must accept a uri parameter (for C1 fix)."""
        from hybridrag.engine.kg.mongo_impl import ClientManager

        sig = inspect.signature(ClientManager.get_client)
        assert "uri" in sig.parameters, (
            "C4/C1: ClientManager.get_client must accept a uri parameter "
            "so the URI is not read from os.environ."
        )


# ── C13: Version floor bump ───────────────────────────────────────────────


class TestC13VersionFloor:
    """C13: pymongo >= 4.7.0,<5.0 in pyproject.toml and requirements.txt."""

    def test_pyproject_pymongo_version(self):
        """pyproject.toml must have pymongo>=4.7.0,<5.0."""
        with open("pyproject.toml") as f:
            content = f.read()

        # Must have the bumped version
        assert "pymongo>=4.7.0" in content, (
            "C13 VIOLATION: pyproject.toml pymongo version not bumped to >=4.7.0"
        )

    def test_requirements_pymongo_version(self):
        """requirements.txt must have pymongo>=4.7.0,<5.0."""
        with open("requirements.txt") as f:
            content = f.read()

        assert "pymongo>=4.7.0" in content, (
            "C13 VIOLATION: requirements.txt pymongo version not bumped to >=4.7.0"
        )


# ── Motor Migration ───────────────────────────────────────────────────────


class TestMotorMigration:
    """Motor -> PyMongo Async migration: no AsyncIOMotorClient in production src."""

    def test_mongodb_client_no_motor_import(self):
        """mongodb_client.py must not import from motor.motor_asyncio."""
        from hybridrag.core import mongodb_client as mc_module

        source = inspect.getsource(mc_module)
        # Allow in comments/docstrings but not as active imports
        import_lines = [
            line.strip()
            for line in source.split("\n")
            if line.strip().startswith("from motor")
            or line.strip().startswith("import motor")
        ]
        assert len(import_lines) == 0, (
            f"Motor MIGRATION VIOLATION: mongodb_client.py still imports Motor: "
            f"{import_lines}"
        )

    def test_mongodb_client_uses_pymongo_async_mongo_client(self):
        """mongodb_client.py must import and use pymongo.AsyncMongoClient."""
        from hybridrag.core import mongodb_client as mc_module

        source = inspect.getsource(mc_module)
        assert "AsyncMongoClient" in source, (
            "Motor MIGRATION: mongodb_client.py must use pymongo.AsyncMongoClient"
        )

    def test_conversation_memory_no_motor(self):
        """conversation.py must not import from motor.motor_asyncio."""
        from hybridrag.memory import conversation as conv_module

        source = inspect.getsource(conv_module)
        import_lines = [
            line.strip()
            for line in source.split("\n")
            if line.strip().startswith("from motor")
            or line.strip().startswith("import motor")
        ]
        assert len(import_lines) == 0, (
            f"Motor MIGRATION VIOLATION: conversation.py still imports Motor: "
            f"{import_lines}"
        )

    def test_conversation_uses_shared_client(self):
        """conversation.py must use get_shared_client or accept a client, not create its own."""
        from hybridrag.memory import conversation as conv_module

        source = inspect.getsource(conv_module.ConversationMemory.initialize)
        assert "AsyncIOMotorClient(" not in source, (
            "Motor MIGRATION VIOLATION: ConversationMemory.initialize still creates "
            "an AsyncIOMotorClient. Must use shared client."
        )


# ── Singleton lifecycle ───────────────────────────────────────────────────


class TestSharedClientLifecycle:
    """Test shared client lifecycle: create, reuse, close."""

    def test_close_shared_client_resets_singleton(self):
        """close_shared_client must reset the singleton to None."""
        from hybridrag.core.mongodb_client import (
            _reset_shared_client,
            close_shared_client,
            get_shared_client,
        )

        _reset_shared_client()

        settings = MagicMock()
        settings.mongodb_uri.get_secret_value.return_value = "mongodb://localhost:27017"
        settings.mongodb_max_pool_size = 100
        settings.mongodb_min_pool_size = 0
        settings.mongodb_max_idle_time_ms = 60000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # Create singleton
            client1 = get_shared_client(settings)
            assert client1 is mock_client

            # Close it
            close_shared_client()

            # Next call should create a new one
            mock_client2 = MagicMock()
            mock_cls.return_value = mock_client2
            client2 = get_shared_client(settings)
            assert client2 is mock_client2

        _reset_shared_client()

    def test_shared_client_has_app_name(self):
        """Shared client must be created with appName='hybridrag'."""
        from hybridrag.core.mongodb_client import (
            _reset_shared_client,
            get_shared_client,
        )

        _reset_shared_client()

        settings = MagicMock()
        settings.mongodb_uri.get_secret_value.return_value = "mongodb://localhost:27017"
        settings.mongodb_max_pool_size = 100
        settings.mongodb_min_pool_size = 0
        settings.mongodb_max_idle_time_ms = 60000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            get_shared_client(settings)

            call_kwargs = mock_cls.call_args
            # appName should be in kwargs
            assert call_kwargs.kwargs.get("appName") == "hybridrag" or (
                len(call_kwargs.args) > 0 and any("appName" in str(call_kwargs))
            ), "Shared client must set appName='hybridrag' for monitoring"

        _reset_shared_client()

    def test_shared_client_has_retry_writes(self):
        """Shared client must have retryWrites=True."""
        from hybridrag.core.mongodb_client import (
            _reset_shared_client,
            get_shared_client,
        )

        _reset_shared_client()

        settings = MagicMock()
        settings.mongodb_uri.get_secret_value.return_value = "mongodb://localhost:27017"
        settings.mongodb_max_pool_size = 100
        settings.mongodb_min_pool_size = 0
        settings.mongodb_max_idle_time_ms = 60000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            get_shared_client(settings)

            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("retryWrites") is True, (
                "Shared client must set retryWrites=True per mongodb-connection skill"
            )

        _reset_shared_client()

    def test_shared_client_has_retry_reads(self):
        """Shared client must have retryReads=True."""
        from hybridrag.core.mongodb_client import (
            _reset_shared_client,
            get_shared_client,
        )

        _reset_shared_client()

        settings = MagicMock()
        settings.mongodb_uri.get_secret_value.return_value = "mongodb://localhost:27017"
        settings.mongodb_max_pool_size = 100
        settings.mongodb_min_pool_size = 0
        settings.mongodb_max_idle_time_ms = 60000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            get_shared_client(settings)

            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("retryReads") is True, (
                "Shared client must set retryReads=True per mongodb-connection skill"
            )

        _reset_shared_client()

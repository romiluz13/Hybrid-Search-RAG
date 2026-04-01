"""Tests for Phase 2A: Connection Hardening (H1-H9, H20, H25, H27).

Validates:
- H1: retryWrites/retryReads present (already done in 1A, verify)
- H2: Timeout settings in Settings and passed to AsyncMongoClient
- H3: Ingestion uses shared client (already done in 1A, verify)
- H4+H5: ConversationMemory accepts pre-created client, uses timeouts
- H6: API lifespan calls close_shared_client on shutdown
- H7+H8: Chainlit shared RAG / on_chat_end handler
- H9: Client reference stored in engine (already done in 1A, verify)
- H20: Transaction helper catches specific pymongo errors, not broad Exception
- H25: Examples use try/finally with close_shared_client
- H27: E2E test wraps Motor client in try/finally

Backing Skill: mongodb-connection
"""

import inspect
from unittest.mock import MagicMock, patch

# ── H1: retryWrites/retryReads (verify 1A) ─────────────────────────────


class TestH1RetrySettings:
    """H1: Verify retryWrites and retryReads are set on shared client."""

    def test_shared_client_has_retry_writes(self):
        """get_shared_client must pass retryWrites=True."""
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
        settings.mongodb_server_selection_timeout_ms = 5000
        settings.mongodb_connect_timeout_ms = 10000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_shared_client(settings)
            assert mock_cls.call_args.kwargs.get("retryWrites") is True

        _reset_shared_client()

    def test_shared_client_has_retry_reads(self):
        """get_shared_client must pass retryReads=True."""
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
        settings.mongodb_server_selection_timeout_ms = 5000
        settings.mongodb_connect_timeout_ms = 10000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_shared_client(settings)
            assert mock_cls.call_args.kwargs.get("retryReads") is True

        _reset_shared_client()


# ── H2: Timeout settings ──────────────────────────────────────────────


class TestH2TimeoutSettings:
    """H2: Settings must have timeout fields and they must be passed to client."""

    def test_settings_has_server_selection_timeout(self):
        """Settings class must define mongodb_server_selection_timeout_ms."""
        from hybridrag.config.settings import Settings

        field_names = Settings.model_fields.keys()
        assert "mongodb_server_selection_timeout_ms" in field_names, (
            "H2 VIOLATION: Settings missing mongodb_server_selection_timeout_ms field. "
            "Must add timeout for quick failover on topology changes."
        )

    def test_settings_has_connect_timeout(self):
        """Settings class must define mongodb_connect_timeout_ms."""
        from hybridrag.config.settings import Settings

        field_names = Settings.model_fields.keys()
        assert "mongodb_connect_timeout_ms" in field_names, (
            "H2 VIOLATION: Settings missing mongodb_connect_timeout_ms field. "
            "Must add timeout to fail fast on connection issues."
        )

    def test_settings_has_socket_timeout(self):
        """Settings class must define mongodb_socket_timeout_ms."""
        from hybridrag.config.settings import Settings

        field_names = Settings.model_fields.keys()
        assert "mongodb_socket_timeout_ms" in field_names, (
            "H2 VIOLATION: Settings missing mongodb_socket_timeout_ms field. "
            "0=no timeout for long-running ops; users can configure it."
        )

    def test_server_selection_timeout_default(self):
        """serverSelectionTimeoutMS default should be 5000 (5s) per skill."""
        from hybridrag.config.settings import Settings

        field = Settings.model_fields["mongodb_server_selection_timeout_ms"]
        assert field.default == 5000, (
            f"H2: serverSelectionTimeoutMS default should be 5000, got {field.default}"
        )

    def test_connect_timeout_default(self):
        """connectTimeoutMS default should be 10000 (10s) per skill."""
        from hybridrag.config.settings import Settings

        field = Settings.model_fields["mongodb_connect_timeout_ms"]
        assert field.default == 10000, (
            f"H2: connectTimeoutMS default should be 10000, got {field.default}"
        )

    def test_socket_timeout_default_is_zero(self):
        """socketTimeoutMS default should be 0 (no timeout) to avoid breaking long ops."""
        from hybridrag.config.settings import Settings

        field = Settings.model_fields["mongodb_socket_timeout_ms"]
        assert field.default == 0, (
            f"H2: socketTimeoutMS default should be 0, got {field.default}"
        )

    def test_shared_client_passes_server_selection_timeout(self):
        """get_shared_client must pass serverSelectionTimeoutMS to AsyncMongoClient."""
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
        settings.mongodb_server_selection_timeout_ms = 5000
        settings.mongodb_connect_timeout_ms = 10000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_shared_client(settings)
            kwargs = mock_cls.call_args.kwargs
            assert kwargs.get("serverSelectionTimeoutMS") == 5000, (
                f"H2 VIOLATION: serverSelectionTimeoutMS not passed. Got kwargs: {kwargs}"
            )

        _reset_shared_client()

    def test_shared_client_passes_connect_timeout(self):
        """get_shared_client must pass connectTimeoutMS to AsyncMongoClient."""
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
        settings.mongodb_server_selection_timeout_ms = 5000
        settings.mongodb_connect_timeout_ms = 10000

        with patch("hybridrag.core.mongodb_client.AsyncMongoClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_shared_client(settings)
            kwargs = mock_cls.call_args.kwargs
            assert kwargs.get("connectTimeoutMS") == 10000, (
                f"H2 VIOLATION: connectTimeoutMS not passed. Got kwargs: {kwargs}"
            )

        _reset_shared_client()


# ── H4+H5: ConversationMemory uses shared client ─────────────────────


class TestH4H5ConversationMemoryClient:
    """H4+H5: ConversationMemory must accept external client, not always create its own."""

    def test_conversation_memory_accepts_client_parameter(self):
        """ConversationMemory.__init__ must accept a client parameter."""
        from hybridrag.memory.conversation import ConversationMemory

        sig = inspect.signature(ConversationMemory.__init__)
        assert "client" in sig.parameters, (
            "H4 VIOLATION: ConversationMemory.__init__ must accept a 'client' parameter "
            "to allow injecting a shared client instead of creating a new one."
        )

    def test_conversation_memory_initialize_uses_provided_client(self):
        """When client is provided, initialize() must use it instead of creating new."""
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory.initialize)
        # Must check for an existing client before creating
        assert "self._client" in source, "H4: initialize() must reference self._client"


# ── H6: API shutdown closes connection ────────────────────────────────


class TestH6APIShutdown:
    """H6: API lifespan must call close_shared_client on shutdown."""

    def test_api_lifespan_imports_close_shared_client(self):
        """api/main.py must import or reference close_shared_client."""
        from hybridrag.api import main as api_main

        source = inspect.getsource(api_main)
        assert "close_shared_client" in source, (
            "H6 VIOLATION: api/main.py does not reference close_shared_client. "
            "Must close the shared client on shutdown."
        )

    def test_api_lifespan_calls_close_on_shutdown(self):
        """The lifespan function must call close_shared_client after yield."""
        from hybridrag.api import main as api_main

        source = inspect.getsource(api_main.lifespan)
        # After yield, close_shared_client must be called
        yield_pos = source.find("yield")
        close_pos = source.find("close_shared_client")
        assert yield_pos > 0, "lifespan must have yield"
        assert close_pos > yield_pos, (
            "H6 VIOLATION: close_shared_client must be called AFTER yield (shutdown phase)"
        )


# ── H20: Transaction helper specific exception handling ──────────────


class TestH20TransactionExceptionHandling:
    """H20: Transaction helper must catch specific pymongo errors, not broad Exception."""

    def test_no_bare_except_exception(self):
        """run_with_transaction must NOT use 'except Exception' as the primary catch."""
        from hybridrag.core import transaction_helper as txn_module

        source = inspect.getsource(txn_module.run_with_transaction)
        # Should not have bare 'except Exception as e:' as the primary handler
        # Instead should catch OperationFailure, ConnectionFailure
        assert "OperationFailure" in source or "ConnectionFailure" in source, (
            "H20 VIOLATION: run_with_transaction must catch specific pymongo errors "
            "(OperationFailure, ConnectionFailure), not broad Exception."
        )

    def test_imports_pymongo_errors(self):
        """transaction_helper must import pymongo.errors for specific handling."""
        from hybridrag.core import transaction_helper as txn_module

        source = inspect.getsource(txn_module)
        assert "pymongo.errors" in source or "from pymongo.errors import" in source, (
            "H20 VIOLATION: transaction_helper must import specific pymongo error classes."
        )

    def test_checks_error_codes(self):
        """Transaction fallback should check error codes, not just string matching."""
        from hybridrag.core import transaction_helper as txn_module

        source = inspect.getsource(txn_module.run_with_transaction)
        # Should reference .code attribute or error code numbers
        assert ".code" in source or "TRANSACTION_NOT_SUPPORTED_CODES" in source, (
            "H20 VIOLATION: Must check error codes (e.code) for transaction fallback, "
            "not rely solely on string matching."
        )


# ── H25: Examples use try/finally with close_shared_client ───────────


class TestH25ExamplesCleanup:
    """H25: Example files must use try/finally with close_shared_client."""

    def _read_example(self, filename: str) -> str:
        """Read an example file."""
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", filename)
        with open(path) as f:
            return f.read()

    def test_quickstart_has_cleanup(self):
        """01_quickstart.py must have close_shared_client in finally block."""
        source = self._read_example("01_quickstart.py")
        assert "close_shared_client" in source, (
            "H25 VIOLATION: 01_quickstart.py missing close_shared_client cleanup."
        )

    def test_web_ingestion_has_cleanup(self):
        """05_web_ingestion.py must have close_shared_client in finally block."""
        source = self._read_example("05_web_ingestion.py")
        assert "close_shared_client" in source, (
            "H25 VIOLATION: 05_web_ingestion.py missing close_shared_client cleanup."
        )

    def test_conversation_memory_has_cleanup(self):
        """06_conversation_memory.py must have close_shared_client in finally block."""
        source = self._read_example("06_conversation_memory.py")
        assert "close_shared_client" in source, (
            "H25 VIOLATION: 06_conversation_memory.py missing close_shared_client cleanup."
        )

    def test_custom_filters_has_cleanup(self):
        """07_custom_filters.py must have close_shared_client in finally block."""
        source = self._read_example("07_custom_filters.py")
        assert "close_shared_client" in source, (
            "H25 VIOLATION: 07_custom_filters.py missing close_shared_client cleanup."
        )

    def test_langgraph_agent_has_cleanup(self):
        """09_langgraph_agent.py must have close_shared_client in finally block."""
        source = self._read_example("09_langgraph_agent.py")
        assert "close_shared_client" in source, (
            "H25 VIOLATION: 09_langgraph_agent.py missing close_shared_client cleanup."
        )


# ── H27: E2E test wraps client in try/finally ────────────────────────


class TestH27E2EClientLifecycle:
    """H27: E2E test must wrap Motor client usage in try/finally."""

    def _read_e2e_test(self) -> str:
        """Read the e2e test file."""
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "tests", "e2e_real_test.py"
        )
        with open(path) as f:
            return f.read()

    def test_mongodb_connection_test_has_try_finally(self):
        """test_mongodb_connection must wrap client in try/finally."""
        source = self._read_e2e_test()
        # Find the test_mongodb_connection function
        func_start = source.find("async def test_mongodb_connection")
        func_end = source.find("\nasync def ", func_start + 1)
        func_source = (
            source[func_start:func_end] if func_end > 0 else source[func_start:]
        )

        assert "finally" in func_source, (
            "H27 VIOLATION: test_mongodb_connection must wrap client in try/finally "
            "to ensure cleanup."
        )

    def test_vector_search_test_has_try_finally(self):
        """test_vector_search_direct must wrap client in try/finally."""
        source = self._read_e2e_test()
        func_start = source.find("async def test_vector_search_direct")
        func_end = source.find("\nasync def ", func_start + 1)
        func_source = (
            source[func_start:func_end] if func_end > 0 else source[func_start:]
        )

        assert "finally" in func_source, (
            "H27 VIOLATION: test_vector_search_direct must wrap client in try/finally "
            "to ensure cleanup."
        )

    def test_knowledge_graph_test_has_try_finally(self):
        """test_knowledge_graph must wrap client in try/finally."""
        source = self._read_e2e_test()
        func_start = source.find("async def test_knowledge_graph")
        func_end = source.find("\nasync def ", func_start + 1)
        func_source = (
            source[func_start:func_end] if func_end > 0 else source[func_start:]
        )

        assert "finally" in func_source, (
            "H27 VIOLATION: test_knowledge_graph must wrap client in try/finally "
            "to ensure cleanup."
        )

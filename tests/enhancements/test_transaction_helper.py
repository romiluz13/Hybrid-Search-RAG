"""Tests for transaction helper with fallback."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import ConnectionFailure, OperationFailure

from hybridrag.core.transaction_helper import run_with_transaction


class TestRunWithTransaction:
    """Test transaction helper fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_transaction_not_supported(self):
        """When transactions are not supported, falls back to non-transactional."""
        call_count = 0

        async def my_callback(session=None):
            nonlocal call_count
            call_count += 1
            return "success"

        # Mock client that raises "transaction numbers" error
        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.with_transaction = AsyncMock(
            side_effect=OperationFailure("transaction numbers are not allowed")
        )
        mock_client.start_session = MagicMock(return_value=mock_session)

        result = await run_with_transaction(
            mock_client,
            my_callback,
            fallback_without_transaction=True,
        )

        assert result == "success"
        assert call_count == 1  # Called once in fallback mode

    @pytest.mark.asyncio
    async def test_no_fallback_raises_error(self):
        """When fallback is disabled, transaction errors propagate."""

        async def my_callback(session=None):
            return "success"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.with_transaction = AsyncMock(
            side_effect=OperationFailure("transaction numbers are not allowed")
        )
        mock_client.start_session = MagicMock(return_value=mock_session)

        with pytest.raises(OperationFailure, match="transaction numbers"):
            await run_with_transaction(
                mock_client,
                my_callback,
                fallback_without_transaction=False,
            )

    @pytest.mark.asyncio
    async def test_non_transaction_error_propagates(self):
        """Non-transaction errors always propagate."""

        async def my_callback(session=None):
            return "success"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.with_transaction = AsyncMock(
            side_effect=OperationFailure("some other error")
        )
        mock_client.start_session = MagicMock(return_value=mock_session)

        with pytest.raises(OperationFailure, match="some other error"):
            await run_with_transaction(
                mock_client,
                my_callback,
                fallback_without_transaction=True,
            )

    @pytest.mark.asyncio
    async def test_callback_receives_session_none_on_fallback(self):
        """In fallback mode, callback receives session=None."""
        received_session = "not_set"

        async def my_callback(session=None):
            nonlocal received_session
            received_session = session
            return "done"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.with_transaction = AsyncMock(
            side_effect=ConnectionFailure("transactions are not supported")
        )
        mock_client.start_session = MagicMock(return_value=mock_session)

        result = await run_with_transaction(mock_client, my_callback)

        assert result == "done"
        assert received_session is None

    @pytest.mark.asyncio
    async def test_generic_exception_not_caught(self):
        """L25: Generic Exception should NOT be caught -- only pymongo errors.

        run_with_transaction should only catch OperationFailure and
        ConnectionFailure, not generic Exception types.
        """

        async def my_callback(session=None):
            return "success"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        # Use a non-pymongo exception -- should propagate, not be caught
        mock_session.with_transaction = AsyncMock(
            side_effect=ValueError("unexpected non-pymongo error")
        )
        mock_client.start_session = MagicMock(return_value=mock_session)

        with pytest.raises(ValueError, match="unexpected non-pymongo error"):
            await run_with_transaction(
                mock_client,
                my_callback,
                fallback_without_transaction=True,
            )

    @pytest.mark.asyncio
    async def test_operation_failure_with_error_code(self):
        """L25: OperationFailure with transaction-not-supported code triggers fallback."""
        call_count = 0

        async def my_callback(session=None):
            nonlocal call_count
            call_count += 1
            return "result"

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        # Error code 263 = OperationNotSupportedInTransaction
        error = OperationFailure("not supported", code=263)
        mock_session.with_transaction = AsyncMock(side_effect=error)
        mock_client.start_session = MagicMock(return_value=mock_session)

        result = await run_with_transaction(
            mock_client,
            my_callback,
            fallback_without_transaction=True,
        )

        assert result == "result"
        assert call_count == 1

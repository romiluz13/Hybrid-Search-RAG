"""
Transaction Helper with Fallback for M0/Standalone Compatibility.

Provides a safe transaction wrapper that falls back to non-transactional
execution when transactions are not supported (M0 free tier, standalone).

[Rule: fundamental-use-transactions-when-required]
[Rule: retry-transient-transaction-error]
[Rule: pattern-withtransaction-vs-core-api]
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger("hybridrag.transactions")

T = TypeVar("T")


async def run_with_transaction(
    client: Any,
    callback: Callable[..., Coroutine[Any, Any, T]],
    *,
    fallback_without_transaction: bool = True,
) -> T:
    """Execute a callback within a MongoDB transaction with fallback.

    Uses the callback API (with_transaction) which handles retries for
    TransientTransactionError and UnknownTransactionCommitResult internally.

    If the deployment does not support transactions (M0 free tier,
    standalone mongod), falls back to running without a transaction.

    Args:
        client: Motor AsyncIOMotorClient instance
        callback: Async function that receives a session parameter.
                  Must be idempotent for retry safety.
        fallback_without_transaction: If True, run without transaction
                                      when transactions are not supported.
                                      Default True for M0/standalone compatibility.

    Returns:
        Return value from the callback.

    Raises:
        Exception: Re-raises if fallback is disabled and transactions fail.

    Example:
        ```python
        async def my_operation(session=None):
            await collection.delete_many({"x": 1}, session=session)
            await collection.insert_one({"x": 2}, session=session)

        await run_with_transaction(client, my_operation)
        ```
    """
    # Try with transaction first
    try:
        async with await client.start_session() as session:
            result = None

            async def _txn_body(s):
                nonlocal result
                result = await callback(session=s)

            await session.with_transaction(_txn_body)
            return result

    except Exception as e:
        error_str = str(e).lower()
        # Check for transaction-not-supported errors
        transaction_not_supported = any(
            marker in error_str
            for marker in [
                "transaction numbers",
                "transactions are not supported",
                "command not supported",
                "no such command",
                "illegal_operation",
                "transaction is not supported",
            ]
        )

        if transaction_not_supported and fallback_without_transaction:
            logger.warning(
                "[TRANSACTION] Transactions not supported on this deployment. "
                "Falling back to non-transactional execution."
            )
            return await callback(session=None)

        # Re-raise for actual errors
        raise

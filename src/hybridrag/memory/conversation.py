"""
MongoDB-backed Conversation Memory for HybridRAG.

Inspired by langchain-ai/langchain-mongodb MongoDBChatMessageHistory.

Key Features:
- Session-based conversation storage
- Configurable history size limit
- Async MongoDB operations
- Compatible with multi-turn conversation workflows

Schema (MongoDB Best Practice - Rule 1.1: No Unbounded Arrays):
    Collection: conversation_sessions
    Document:
    {
        "_id": ObjectId,
        "session_id": "unique-session-id",
        "message_count": 42,
        "summary": "...",
        "created_at": ISODate,
        "updated_at": ISODate,
        "metadata": {}
    }

    Collection: conversation_messages
    Document:
    {
        "_id": ObjectId,
        "session_id": "unique-session-id",
        "role": "user" | "assistant",
        "content": "...",
        "timestamp": ISODate,
        "message_index": 0
    }
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


logger = logging.getLogger("hybridrag.memory")

# Default configuration
DEFAULT_SESSIONS_COLLECTION = "conversation_sessions"
DEFAULT_MESSAGES_COLLECTION = "conversation_messages"
DEFAULT_HISTORY_SIZE = 50  # Keep last N message pairs (100 messages total)
DEFAULT_MAX_TOKEN_LIMIT = 32000  # Token limit before compaction (models support 200K+)
DEFAULT_SESSION_ID_KEY = "session_id"

# Summarization prompt template
SUMMARY_PROMPT = """Progressively summarize the conversation below, adding onto the previous summary.
Return a new summary that incorporates the key points from both the previous summary and new messages.
Focus on important facts, decisions, and context that would be useful for future questions.

CURRENT SUMMARY:
{summary}

NEW MESSAGES:
{new_lines}

NEW SUMMARY:"""


@dataclass
class ConversationSession:
    """Represents a conversation session with its messages."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""  # Running summary of compacted (old) messages
    summary_token_count: int = 0  # Estimated token count of summary
    message_count: int = 0  # Denormalized count for quick access
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_history_format(
        self, max_messages: int | None = None
    ) -> list[dict[str, str]]:
        """
        Convert messages to conversation history format.

        Args:
            max_messages: Optional limit on number of messages to include

        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        messages = self.messages
        if max_messages is not None:
            messages = messages[-max_messages:]

        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def to_context_string(self, max_messages: int | None = None) -> str:
        """
        Convert messages to a context string for retrieval augmentation.

        Includes running summary (if exists) plus recent messages.

        Args:
            max_messages: Optional limit on number of messages to include

        Returns:
            Formatted string of conversation history
        """
        parts = []

        # Include summary of older messages if exists
        if self.summary:
            parts.append(f"Summary of earlier conversation:\n{self.summary}")

        # Include recent messages
        messages = self.messages
        if max_messages is not None:
            messages = messages[-max_messages:]

        if messages:
            lines = ["Recent conversation:"]
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                # Truncate very long messages for context
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"{role}: {content}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else ""


class ConversationMemory:
    """
    MongoDB-backed conversation memory for multi-turn RAG conversations.

    Uses separate collections for sessions and messages to avoid unbounded
    array growth (MongoDB Schema Design Rule 1.1).

    Stores conversation sessions in MongoDB and provides methods to:
    - Create/retrieve sessions
    - Add messages to sessions
    - Get conversation history for RAG queries
    - Clear or delete sessions

    Example:
        ```python
        memory = ConversationMemory(
            mongodb_uri="mongodb+srv://...",
            database="hybridrag"
        )
        await memory.initialize()

        # Start conversation
        session_id = await memory.create_session()

        # Add user message
        await memory.add_message(session_id, "user", "What is this book about?")

        # Get history for RAG query
        history = await memory.get_history(session_id)

        # Add assistant response
        await memory.add_message(session_id, "assistant", "This book is about...")
        ```
    """

    def __init__(
        self,
        mongodb_uri: str | None = None,
        database: str | None = None,
        collection_name: str = DEFAULT_SESSIONS_COLLECTION,
        messages_collection_name: str = DEFAULT_MESSAGES_COLLECTION,
        history_size: int | None = DEFAULT_HISTORY_SIZE,
        max_token_limit: int = DEFAULT_MAX_TOKEN_LIMIT,
        llm_func: Callable | None = None,
        session_id_key: str = DEFAULT_SESSION_ID_KEY,
        client: AsyncMongoClient | None = None,
    ):
        """
        Initialize ConversationMemory with self-compaction support.

        Args:
            mongodb_uri: MongoDB connection string (defaults to MONGODB_URI env var)
            database: Database name (defaults to MONGODB_DATABASE env var)
            collection_name: Collection name for sessions
            messages_collection_name: Collection name for messages (separate collection)
            history_size: Max number of message pairs to keep (None for unlimited)
            max_token_limit: Token limit before compaction (summarization)
            llm_func: LLM function for summarization (required for compaction)
            session_id_key: Field name for session ID
            client: Pre-created AsyncMongoClient to reuse (avoids creating a new one).
                    When provided, mongodb_uri is not required.
        """
        self._mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        self._database_name = database or os.environ.get(
            "MONGODB_DATABASE", "hybridrag"
        )
        self._sessions_collection_name = collection_name
        self._messages_collection_name = messages_collection_name
        self._history_size = history_size
        self._max_token_limit = max_token_limit
        self._llm_func = llm_func
        self._session_id_key = session_id_key

        self._client: AsyncMongoClient | None = client
        self._owns_client = client is None  # Only close if we created it
        self._db: AsyncDatabase | None = None
        self._initialized = False

        if not self._mongodb_uri and client is None:
            raise ValueError(
                "MongoDB URI or client required. Set MONGODB_URI environment variable, "
                "pass mongodb_uri parameter, or provide a pre-created client."
            )

    async def initialize(self) -> None:
        """Initialize MongoDB connection and create indexes."""
        if self._initialized:
            return

        logger.info(
            f"[MEMORY] Initializing ConversationMemory: "
            f"sessions={self._sessions_collection_name}, "
            f"messages={self._messages_collection_name}"
        )

        # Use injected client if available, otherwise create new one
        # [Rule: mongodb-connection] Create client once and reuse
        if self._client is None:
            self._client = AsyncMongoClient(self._mongodb_uri)
        self._db = self._client[self._database_name]
        self._sessions_collection = self._db[self._sessions_collection_name]
        self._messages_collection = self._db[self._messages_collection_name]

        # Create indexes on sessions collection
        await self._sessions_collection.create_index(self._session_id_key, unique=True)
        await self._sessions_collection.create_index("created_at")
        await self._sessions_collection.create_index("updated_at")

        # M15: TTL index for automatic session cleanup (90 days)
        # [Rule: mongodb-schema-design] TTL indexes automatically remove stale documents
        await self._sessions_collection.create_index(
            "updated_at",
            expireAfterSeconds=7776000,  # 60 * 60 * 24 * 90 = 90 days
            name="session_ttl",
        )

        # Create indexes on messages collection (Rule 1.1 compliant)
        # NOTE: Single-field session_id_key index removed (M14) -- the compound indexes
        # below cover this prefix per core-indexing-principles.
        await self._messages_collection.create_index(
            [(self._session_id_key, 1), ("timestamp", 1)]
        )
        await self._messages_collection.create_index(
            [(self._session_id_key, 1), ("message_index", 1)]
        )

        self._initialized = True
        logger.info("[MEMORY] ConversationMemory initialized")

    def _ensure_initialized(self) -> None:
        """Ensure memory is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "ConversationMemory not initialized. Call 'await memory.initialize()' first."
            )

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional custom session ID (generated if not provided)
            metadata: Optional metadata to store with session

        Returns:
            Session ID
        """
        self._ensure_initialized()

        session_id = session_id or str(uuid.uuid4())
        now = _utcnow()

        doc = {
            self._session_id_key: session_id,
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }

        try:
            await self._sessions_collection.insert_one(doc)
            logger.info(f"[MEMORY] Created session: {session_id}")
        except Exception as e:
            # Session might already exist
            if "duplicate key" in str(e).lower():
                logger.debug(f"[MEMORY] Session already exists: {session_id}")
            else:
                raise

        return session_id

    async def get_session(self, session_id: str) -> ConversationSession | None:
        """
        Get a conversation session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ConversationSession or None if not found
        """
        self._ensure_initialized()

        doc = await self._sessions_collection.find_one(
            {self._session_id_key: session_id}
        )

        if not doc:
            return None

        # Fetch messages from separate collection
        messages = await self._get_messages_from_collection(session_id)

        return ConversationSession(
            session_id=doc[self._session_id_key],
            messages=messages,
            summary=doc.get("summary", ""),
            summary_token_count=doc.get("summary_token_count", 0),
            message_count=doc.get("message_count", len(messages)),
            created_at=doc.get("created_at", _utcnow()),
            updated_at=doc.get("updated_at", _utcnow()),
            metadata=doc.get("metadata", {}),
        )

    async def _get_messages_from_collection(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch messages from the messages collection.

        Args:
            session_id: Session ID
            limit: Optional limit on number of messages

        Returns:
            List of message dicts sorted by message_index
        """
        cursor = self._messages_collection.find(
            {self._session_id_key: session_id},
            {
                "embedding": 0,
                "vector": 0,
            },  # M13: Exclude large fields not needed for messages
        ).sort("message_index", 1)

        if limit is not None:
            # Get total count to calculate skip for "last N" messages
            total = await self._messages_collection.count_documents(
                {self._session_id_key: session_id}
            )
            if total > limit:
                cursor = cursor.skip(total - limit)

        messages = []
        async for msg in cursor:
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp"),
                    "message_index": msg.get("message_index", 0),
                    "metadata": msg.get("metadata"),
                }
            )

        return messages

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional message metadata
        """
        self._ensure_initialized()

        now = _utcnow()

        # Get current message count for index
        session_doc = await self._sessions_collection.find_one(
            {self._session_id_key: session_id}, {"message_count": 1}
        )

        if session_doc:
            message_index = session_doc.get("message_count", 0)
        else:
            # Session doesn't exist, create it first
            await self.create_session(session_id)
            message_index = 0

        # Insert message and update session atomically
        # [Rule: fundamental-use-transactions-when-required]
        message_doc = {
            self._session_id_key: session_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "message_index": message_index,
        }
        if metadata:
            message_doc["metadata"] = metadata

        try:
            from hybridrag.core.transaction_helper import run_with_transaction

            async def _add_message_txn(session=None):
                await self._messages_collection.insert_one(message_doc, session=session)
                await self._sessions_collection.update_one(
                    {self._session_id_key: session_id},
                    {
                        "$inc": {"message_count": 1},
                        "$set": {"updated_at": now},
                    },
                    session=session,
                )

            await run_with_transaction(
                self._client, _add_message_txn, fallback_without_transaction=True
            )
        except ImportError:
            # Fallback if transaction_helper not available
            await self._messages_collection.insert_one(message_doc)
            await self._sessions_collection.update_one(
                {self._session_id_key: session_id},
                {
                    "$inc": {"message_count": 1},
                    "$set": {"updated_at": now},
                },
            )

        logger.debug(f"[MEMORY] Added {role} message to session {session_id}")

        # Trim history if needed
        if self._history_size is not None:
            await self._trim_history(session_id)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 4 chars = 1 token)."""
        return len(text) // 4

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages for summarization prompt."""
        lines = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    async def _summarize_messages(
        self,
        messages: list[dict],
        existing_summary: str,
    ) -> str:
        """
        Use LLM to summarize messages into a running summary.

        Args:
            messages: Messages to summarize
            existing_summary: Existing summary to build upon

        Returns:
            New summary incorporating old summary + new messages
        """
        if not self._llm_func:
            logger.warning("[MEMORY] No LLM function provided, cannot summarize")
            return existing_summary

        new_lines = self._format_messages_for_summary(messages)
        prompt = SUMMARY_PROMPT.format(
            summary=existing_summary or "(No previous summary)",
            new_lines=new_lines,
        )

        try:
            new_summary = await self._llm_func(prompt)
            logger.info(f"[MEMORY] Generated summary: {len(new_summary)} chars")
            return new_summary.strip()
        except Exception as e:
            logger.error(f"[MEMORY] Summarization failed: {e}")
            return existing_summary

    async def _compact_if_needed(self, session_id: str) -> None:
        """
        Compact conversation if it exceeds token limit.

        Instead of deleting old messages, summarizes them and keeps:
        [Summary] + [Recent Messages]
        """
        session = await self.get_session(session_id)
        if not session:
            return

        # Calculate current token usage
        total_tokens = self._estimate_tokens(session.summary)
        for msg in session.messages:
            total_tokens += self._estimate_tokens(msg.get("content", ""))

        if total_tokens <= self._max_token_limit:
            return  # Under limit, no compaction needed

        logger.info(
            f"[MEMORY] Session {session_id[:8]}... exceeds token limit "
            f"({total_tokens} > {self._max_token_limit}), compacting..."
        )

        # Prune oldest messages until under limit (keep at least 4 messages)
        messages = session.messages.copy()
        pruned = []
        min_keep = 4  # Always keep at least 4 recent messages

        while (
            self._estimate_tokens(session.summary)
            + sum(self._estimate_tokens(m.get("content", "")) for m in messages)
            > self._max_token_limit
            and len(messages) > min_keep
        ):
            pruned.append(messages.pop(0))

        if not pruned:
            return  # Nothing to prune

        # Summarize pruned messages and apply atomically
        # [Rule: fundamental-use-transactions-when-required]
        if self._llm_func:
            new_summary = await self._summarize_messages(pruned, session.summary)
            summary_tokens = self._estimate_tokens(new_summary)
            pruned_indexes = [m.get("message_index", 0) for m in pruned]

            try:
                from hybridrag.core.transaction_helper import run_with_transaction

                async def _compact_with_summary(session_param=None):
                    await self._messages_collection.delete_many(
                        {
                            self._session_id_key: session_id,
                            "message_index": {"$in": pruned_indexes},
                        },
                        session=session_param,
                    )
                    await self._sessions_collection.update_one(
                        {self._session_id_key: session_id},
                        {
                            "$set": {
                                "summary": new_summary,
                                "summary_token_count": summary_tokens,
                                "summary_updated_at": _utcnow(),
                                "updated_at": _utcnow(),
                                "message_count": len(messages),
                            }
                        },
                        session=session_param,
                    )

                await run_with_transaction(
                    self._client,
                    _compact_with_summary,
                    fallback_without_transaction=True,
                )
            except ImportError:
                await self._messages_collection.delete_many(
                    {
                        self._session_id_key: session_id,
                        "message_index": {"$in": pruned_indexes},
                    }
                )
                await self._sessions_collection.update_one(
                    {self._session_id_key: session_id},
                    {
                        "$set": {
                            "summary": new_summary,
                            "summary_token_count": summary_tokens,
                            "summary_updated_at": _utcnow(),
                            "updated_at": _utcnow(),
                            "message_count": len(messages),
                        }
                    },
                )
            logger.info(
                f"[MEMORY] Compacted session {session_id[:8]}...: "
                f"pruned {len(pruned)} messages, kept {len(messages)}"
            )
        else:
            # No LLM - fall back to simple truncation (delete oldest messages)
            pruned_indexes = [m.get("message_index", 0) for m in pruned]

            try:
                from hybridrag.core.transaction_helper import run_with_transaction

                async def _compact_truncate(session_param=None):
                    await self._messages_collection.delete_many(
                        {
                            self._session_id_key: session_id,
                            "message_index": {"$in": pruned_indexes},
                        },
                        session=session_param,
                    )
                    await self._sessions_collection.update_one(
                        {self._session_id_key: session_id},
                        {
                            "$set": {
                                "updated_at": _utcnow(),
                                "message_count": len(messages),
                            }
                        },
                        session=session_param,
                    )

                await run_with_transaction(
                    self._client,
                    _compact_truncate,
                    fallback_without_transaction=True,
                )
            except ImportError:
                await self._messages_collection.delete_many(
                    {
                        self._session_id_key: session_id,
                        "message_index": {"$in": pruned_indexes},
                    }
                )
                await self._sessions_collection.update_one(
                    {self._session_id_key: session_id},
                    {
                        "$set": {
                            "updated_at": _utcnow(),
                            "message_count": len(messages),
                        }
                    },
                )
            logger.warning(
                f"[MEMORY] No LLM for summarization, truncated {len(pruned)} messages"
            )

    async def _trim_history(self, session_id: str) -> None:
        """
        Trim/compact history if it exceeds limits.

        Uses intelligent compaction (summarization) if LLM is available,
        otherwise falls back to simple truncation.
        """
        # Use compaction if LLM is available
        if self._llm_func:
            await self._compact_if_needed(session_id)
        elif self._history_size is not None:
            # Legacy truncation behavior
            max_messages = self._history_size * 2  # Pairs of user/assistant

            message_count = await self._messages_collection.count_documents(
                {self._session_id_key: session_id}
            )

            if message_count > max_messages:
                # Find messages to delete (oldest ones)
                to_delete = message_count - max_messages
                cursor = (
                    self._messages_collection.find(
                        {self._session_id_key: session_id}, {"_id": 1}
                    )
                    .sort("message_index", 1)
                    .limit(to_delete)
                )

                ids_to_delete = [doc["_id"] async for doc in cursor]

                if ids_to_delete:
                    await self._messages_collection.delete_many(
                        {"_id": {"$in": ids_to_delete}}
                    )

                    # Update message count
                    await self._sessions_collection.update_one(
                        {self._session_id_key: session_id},
                        {"$set": {"message_count": max_messages}},
                    )

                    logger.debug(
                        f"[MEMORY] Trimmed history for session {session_id} "
                        f"to {max_messages} messages"
                    )

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages from a session.

        Args:
            session_id: Session ID
            limit: Optional limit on number of messages to return

        Returns:
            List of message dicts
        """
        self._ensure_initialized()
        return await self._get_messages_from_collection(session_id, limit)

    async def get_history(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Get conversation history in standard format.

        Args:
            session_id: Session ID
            max_messages: Optional limit on number of messages

        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        messages = await self.get_messages(session_id, limit=max_messages)
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    # Alias for backwards compatibility
    async def get_lightrag_history(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, str]]:
        """Alias for get_history (backwards compatibility)."""
        return await self.get_history(session_id, max_messages)

    async def get_context_string(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> str:
        """
        Get conversation history as a context string for retrieval.

        This is useful for augmenting queries with conversation context
        to improve follow-up question handling.

        Args:
            session_id: Session ID
            max_messages: Max messages to include

        Returns:
            Formatted string of conversation history
        """
        session = await self.get_session(session_id)
        if not session:
            return ""
        return session.to_context_string(max_messages)

    async def clear_session(self, session_id: str) -> None:
        """
        Clear all messages from a session (keeps the session).

        Uses run_with_transaction to ensure message deletion and session
        update happen atomically. Falls back to non-transactional on
        standalone/M0 deployments.

        Args:
            session_id: Session ID
        """
        self._ensure_initialized()

        async def _do_clear(session=None):
            # Delete all messages for this session
            await self._messages_collection.delete_many(
                {self._session_id_key: session_id}, session=session
            )
            # Reset session's message count and summary
            await self._sessions_collection.update_one(
                {self._session_id_key: session_id},
                {
                    "$set": {
                        "message_count": 0,
                        "summary": "",
                        "summary_token_count": 0,
                        "updated_at": _utcnow(),
                    }
                },
                session=session,
            )

        from hybridrag.core.transaction_helper import run_with_transaction

        await run_with_transaction(self._client, _do_clear)
        logger.info(f"[MEMORY] Cleared session: {session_id}")

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.

        Uses run_with_transaction to ensure message deletion and session
        deletion happen atomically. Falls back to non-transactional on
        standalone/M0 deployments.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        self._ensure_initialized()

        result_holder: dict = {"deleted": False}

        async def _do_delete(session=None):
            # Delete all messages for this session
            await self._messages_collection.delete_many(
                {self._session_id_key: session_id}, session=session
            )
            # Delete the session document
            result = await self._sessions_collection.delete_one(
                {self._session_id_key: session_id}, session=session
            )
            result_holder["deleted"] = result.deleted_count > 0

        from hybridrag.core.transaction_helper import run_with_transaction

        await run_with_transaction(self._client, _do_delete)
        deleted = result_holder["deleted"]

        if deleted:
            logger.info(f"[MEMORY] Deleted session: {session_id}")
        else:
            logger.debug(f"[MEMORY] Session not found for deletion: {session_id}")

        return deleted

    async def list_sessions(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List conversation sessions with last message via $lookup.

        Uses a single aggregation pipeline with $lookup instead of N+1 queries.
        [Rule: query-batch-operations] [Rule: agg-lookup-index]

        Args:
            limit: Max sessions to return
            skip: Number of sessions to skip

        Returns:
            List of session metadata dicts
        """
        self._ensure_initialized()

        # Single aggregation with $lookup replaces N+1 pattern
        pipeline = [
            {"$sort": {"updated_at": -1}},
            {"$skip": skip},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": self._messages_collection_name,
                    "let": {"sid": f"${self._session_id_key}"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {"$eq": [f"${self._session_id_key}", "$$sid"]}
                            }
                        },
                        {"$sort": {"message_index": -1}},
                        {"$limit": 1},
                        {
                            "$project": {
                                "role": 1,
                                "content": 1,
                                "timestamp": 1,
                                "_id": 0,
                            }
                        },
                    ],
                    "as": "_last_messages",
                }
            },
            {
                "$project": {
                    self._session_id_key: 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "metadata": 1,
                    "message_count": 1,
                    "_last_messages": 1,
                }
            },
        ]

        cursor = self._sessions_collection.aggregate(pipeline, maxTimeMS=30000)

        sessions = []
        async for doc in cursor:
            last_messages = doc.get("_last_messages", [])
            last_message = last_messages[0] if last_messages else None

            sessions.append(
                {
                    "session_id": doc[self._session_id_key],
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "metadata": doc.get("metadata", {}),
                    "message_count": doc.get("message_count", 0),
                    "last_message": last_message,
                }
            )

        return sessions

    async def close(self) -> None:
        """Close MongoDB connection (only if we created it).

        If a shared client was injected via the ``client`` parameter,
        we do NOT close it -- the caller manages its lifecycle.
        """
        if self._client and self._owns_client:
            self._client.close()
            logger.info("[MEMORY] ConversationMemory closed (own client)")
        self._client = None
        self._db = None
        self._initialized = False

"""
Integration tests for ConversationMemory with MongoDB.

Tests the new schema design following MongoDB Best Practice Rule 1.1:
- Sessions stored in conversation_sessions (no unbounded arrays)
- Messages stored in conversation_messages (separate collection)

Run with: pytest tests/integration/test_conversation_memory.py -v
"""

import uuid

import pytest
from pymongo import AsyncMongoClient

from hybridrag.memory import ConversationMemory
from tests.conftest import _resolve_test_mongodb_uri

pytestmark = pytest.mark.integration


@pytest.fixture
async def memory(require_mongodb_uri):
    """
    Create ConversationMemory instance for integration tests.

    Uses separate test collections to avoid polluting production data.
    """
    test_uri = _resolve_test_mongodb_uri()
    database_name = f"hybridrag_memory_{uuid.uuid4().hex[:8]}"
    memory_instance = ConversationMemory(
        mongodb_uri=test_uri,
        database=database_name,
        collection_name="test_conversation_sessions",
        messages_collection_name="test_conversation_messages",
    )
    await memory_instance.initialize()
    yield memory_instance
    await memory_instance.close()
    # Drop the test database to prevent leak (C8)
    client = AsyncMongoClient(test_uri)
    await client.drop_database(database_name)
    await client.close()


@pytest.fixture
def unique_session_id():
    """Generate a unique session ID for each test."""
    return f"test-{uuid.uuid4()}"


@pytest.mark.integration
class TestConversationMemorySchema:
    """Test the new schema design (Rule 1.1 compliant)."""

    @pytest.mark.asyncio
    async def test_create_session(self, memory, unique_session_id):
        """Test session creation creates document without messages array."""
        session_id = await memory.create_session(
            session_id=unique_session_id, metadata={"test": "create_session"}
        )

        assert session_id == unique_session_id

        # Verify session document structure
        session = await memory.get_session(session_id)
        assert session is not None
        assert session.session_id == unique_session_id
        assert session.message_count == 0
        assert session.messages == []

        # Cleanup
        await memory.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_add_message_creates_separate_document(
        self, memory, unique_session_id
    ):
        """Test that messages are stored in separate collection."""
        session_id = await memory.create_session(session_id=unique_session_id)

        # Add messages
        await memory.add_message(session_id, "user", "Hello, world!")
        await memory.add_message(session_id, "assistant", "Hello! How can I help?")

        # Get session and verify message_count
        session = await memory.get_session(session_id)
        assert session is not None
        assert session.message_count == 2

        # Verify messages are fetched from separate collection
        messages = await memory.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello! How can I help?"

        # Cleanup
        await memory.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_message_ordering(self, memory, unique_session_id):
        """Test that messages maintain correct order via message_index."""
        session_id = await memory.create_session(session_id=unique_session_id)

        # Add multiple messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            await memory.add_message(session_id, role, f"Message {i}")

        messages = await memory.get_messages(session_id)
        assert len(messages) == 5

        # Verify ordering
        for i, msg in enumerate(messages):
            assert msg["message_index"] == i
            assert msg["content"] == f"Message {i}"

        # Cleanup
        await memory.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, memory, unique_session_id):
        """Test getting last N messages."""
        session_id = await memory.create_session(session_id=unique_session_id)

        # Add 10 messages
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            await memory.add_message(session_id, role, f"Message {i}")

        # Get last 3 messages
        messages = await memory.get_messages(session_id, limit=3)
        assert len(messages) == 3
        assert messages[0]["content"] == "Message 7"
        assert messages[1]["content"] == "Message 8"
        assert messages[2]["content"] == "Message 9"

        # Cleanup
        await memory.delete_session(session_id)


@pytest.mark.integration
class TestConversationMemoryOperations:
    """Test CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_history(self, memory, unique_session_id):
        """Test getting conversation history in standard format."""
        session_id = await memory.create_session(session_id=unique_session_id)

        await memory.add_message(session_id, "user", "What is MongoDB?")
        await memory.add_message(
            session_id, "assistant", "MongoDB is a document database."
        )

        history = await memory.get_history(session_id)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is MongoDB?"}
        assert history[1] == {
            "role": "assistant",
            "content": "MongoDB is a document database.",
        }

        # Cleanup
        await memory.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_get_context_string(self, memory, unique_session_id):
        """Test context string generation."""
        session_id = await memory.create_session(session_id=unique_session_id)

        await memory.add_message(session_id, "user", "Hello")
        await memory.add_message(session_id, "assistant", "Hi there!")

        context = await memory.get_context_string(session_id)
        assert "User: Hello" in context
        assert "Assistant: Hi there!" in context

        # Cleanup
        await memory.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_clear_session(self, memory, unique_session_id):
        """Test clearing messages from a session."""
        session_id = await memory.create_session(session_id=unique_session_id)

        await memory.add_message(session_id, "user", "Message 1")
        await memory.add_message(session_id, "assistant", "Response 1")

        # Verify messages exist
        messages = await memory.get_messages(session_id)
        assert len(messages) == 2

        # Clear session
        await memory.clear_session(session_id)

        # Verify messages are gone but session exists
        messages = await memory.get_messages(session_id)
        assert len(messages) == 0

        session = await memory.get_session(session_id)
        assert session is not None
        assert session.message_count == 0

        # Cleanup
        await memory.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_delete_session(self, memory, unique_session_id):
        """Test deleting a session also deletes its messages."""
        session_id = await memory.create_session(session_id=unique_session_id)

        await memory.add_message(session_id, "user", "Message 1")
        await memory.add_message(session_id, "assistant", "Response 1")

        # Delete session
        deleted = await memory.delete_session(session_id)
        assert deleted is True

        # Verify session is gone
        session = await memory.get_session(session_id)
        assert session is None

        # Verify messages are also gone
        messages = await memory.get_messages(session_id)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_list_sessions(self, memory):
        """Test listing sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            sid = f"list-test-{uuid.uuid4()}"
            await memory.create_session(session_id=sid, metadata={"index": i})
            await memory.add_message(sid, "user", f"Message for session {i}")
            session_ids.append(sid)

        # List sessions
        sessions = await memory.list_sessions(limit=10)

        # Verify our test sessions are included
        listed_ids = [s["session_id"] for s in sessions]
        for sid in session_ids:
            assert sid in listed_ids

        # Verify session info includes message count and last message
        for s in sessions:
            if s["session_id"] in session_ids:
                assert s["message_count"] == 1
                assert s["last_message"] is not None
                assert s["last_message"]["role"] == "user"

        # Cleanup
        for sid in session_ids:
            await memory.delete_session(sid)


class TestConversationMemoryAutoCreate:
    """Test auto-creation behavior."""

    @pytest.mark.asyncio
    async def test_add_message_auto_creates_session(self, memory):
        """Test that add_message creates session if it doesn't exist."""
        session_id = f"auto-create-{uuid.uuid4()}"

        # Add message to non-existent session
        await memory.add_message(session_id, "user", "Auto-created session")

        # Verify session was created
        session = await memory.get_session(session_id)
        assert session is not None
        assert session.message_count == 1

        messages = await memory.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Auto-created session"

        # Cleanup
        await memory.delete_session(session_id)


class TestConversationMemoryEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, memory):
        """Test getting a session that doesn't exist."""
        session = await memory.get_session("nonexistent-session-id")
        assert session is None

    @pytest.mark.asyncio
    async def test_get_messages_nonexistent_session(self, memory):
        """Test getting messages from a session that doesn't exist."""
        messages = await memory.get_messages("nonexistent-session-id")
        assert messages == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, memory):
        """Test deleting a session that doesn't exist."""
        deleted = await memory.delete_session("nonexistent-session-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_duplicate_session_creation(self, memory, unique_session_id):
        """Test creating a session with duplicate ID doesn't error."""
        # Create session first time
        await memory.create_session(session_id=unique_session_id)

        # Create again - should not raise
        session_id = await memory.create_session(session_id=unique_session_id)
        assert session_id == unique_session_id

        # Cleanup
        await memory.delete_session(session_id)

"""
Tests for Phase 3E: Schema Design & Data findings (M15, M29, M30, M31, M38).

Backing Skill: mongodb-schema-design
"""

import inspect


class TestM15TTLIndexOnConversationData:
    """M15: No TTL index on conversation data.

    Conversation sessions should have a TTL index on `updated_at`
    for automatic cleanup of stale sessions (90 days).
    """

    def test_initialize_creates_ttl_index_on_sessions(self):
        """The initialize() method must call create_index with expireAfterSeconds."""
        source = inspect.getsource(
            __import__(
                "hybridrag.memory.conversation",
                fromlist=["ConversationMemory"],
            ).ConversationMemory.initialize
        )
        # Must contain expireAfterSeconds for TTL
        assert "expireAfterSeconds" in source, (
            "conversation.py initialize() must create a TTL index with expireAfterSeconds"
        )

    def test_ttl_index_on_updated_at_field(self):
        """TTL index must be on the updated_at field."""
        source = inspect.getsource(
            __import__(
                "hybridrag.memory.conversation",
                fromlist=["ConversationMemory"],
            ).ConversationMemory.initialize
        )
        # Must reference updated_at and expireAfterSeconds in a TTL index call
        assert "updated_at" in source and "expireAfterSeconds" in source, (
            "TTL index must be on updated_at field"
        )

    def test_ttl_seconds_is_90_days(self):
        """TTL should be 90 days (7776000 seconds)."""
        source = inspect.getsource(
            __import__(
                "hybridrag.memory.conversation",
                fromlist=["ConversationMemory"],
            ).ConversationMemory.initialize
        )
        # 90 days = 60 * 60 * 24 * 90 = 7776000
        assert "7776000" in source or "60 * 60 * 24 * 90" in source, (
            "TTL should be 90 days (7776000 seconds)"
        )

    def test_ttl_index_has_name(self):
        """TTL index should have a descriptive name."""
        source = inspect.getsource(
            __import__(
                "hybridrag.memory.conversation",
                fromlist=["ConversationMemory"],
            ).ConversationMemory.initialize
        )
        assert "session_ttl" in source, "TTL index should be named 'session_ttl'"


class TestM29WriteConcrernOnInsertMany:
    """M29: No write concern on insert_many.

    ALREADY ADDRESSED by M1 (default write concern is 'majority').
    Verify the default is in place.
    """

    def test_default_write_concern_is_majority(self):
        """Settings default write concern must be 'majority'."""
        from hybridrag.config.settings import Settings

        # Need to check the field default, not instantiate (requires env)
        field_info = Settings.model_fields.get("mongodb_write_concern")
        assert field_info is not None, "mongodb_write_concern field must exist"
        assert field_info.default == "majority", (
            "Default write concern must be 'majority' (M1 fix)"
        )


class TestM30DuplicateDetectionOnReIngestion:
    """M30: No duplicate detection on re-ingestion.

    Pipeline must check content_hash before inserting a new document.
    """

    def test_store_document_computes_content_hash(self):
        """_store_document must include a content_hash in the document dict."""
        source = inspect.getsource(
            __import__(
                "hybridrag.ingestion.pipeline",
                fromlist=["DocumentIngestionPipeline"],
            ).DocumentIngestionPipeline._store_document
        )
        assert "content_hash" in source, (
            "_store_document must compute and store content_hash for duplicate detection"
        )

    def test_store_document_uses_sha256(self):
        """Content hash must use SHA-256."""
        source = inspect.getsource(
            __import__(
                "hybridrag.ingestion.pipeline",
                fromlist=["DocumentIngestionPipeline"],
            ).DocumentIngestionPipeline._store_document
        )
        assert "sha256" in source, "Content hash must use SHA-256"

    def test_store_document_checks_existing_hash(self):
        """Must check for existing document with same hash before inserting."""
        source = inspect.getsource(
            __import__(
                "hybridrag.ingestion.pipeline",
                fromlist=["DocumentIngestionPipeline"],
            ).DocumentIngestionPipeline._store_document
        )
        assert "find_one" in source, (
            "Must check for existing document with same content_hash via find_one"
        )

    def test_hashlib_imported_in_pipeline(self):
        """hashlib must be imported in pipeline module."""
        mod = __import__(
            "hybridrag.ingestion.pipeline",
            fromlist=["hashlib"],
        )
        assert hasattr(mod, "hashlib") or "hashlib" in dir(mod), (
            "hashlib must be imported in pipeline.py"
        )


class TestM31AtomicCleanCollections:
    """M31: _clean_collections deletes in two steps -- not atomic.

    Must wrap both delete_many calls in run_with_transaction.
    """

    def test_clean_collections_uses_transaction(self):
        """_clean_collections must use run_with_transaction."""
        source = inspect.getsource(
            __import__(
                "hybridrag.ingestion.pipeline",
                fromlist=["DocumentIngestionPipeline"],
            ).DocumentIngestionPipeline._clean_collections
        )
        assert "run_with_transaction" in source, (
            "_clean_collections must use run_with_transaction for atomicity"
        )

    def test_clean_collections_passes_session(self):
        """Delete operations must pass session parameter."""
        source = inspect.getsource(
            __import__(
                "hybridrag.ingestion.pipeline",
                fromlist=["DocumentIngestionPipeline"],
            ).DocumentIngestionPipeline._clean_collections
        )
        assert "session=" in source, (
            "delete_many calls must pass session= for transactional support"
        )


class TestM38AdditionalPropertiesFalseRemoved:
    """M38: additionalProperties:false on sessions validator blocks extensibility.

    The sessions validator must NOT have additionalProperties: False.
    """

    def test_sessions_validator_no_additional_properties_false(self):
        """conversation_sessions validator must not have additionalProperties: False."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        sessions_validator = VALIDATORS.get("conversation_sessions", {})
        schema = sessions_validator.get("$jsonSchema", {})
        assert schema.get("additionalProperties") is not False, (
            "Sessions validator must not have additionalProperties: False "
            "to allow flexible metadata fields"
        )

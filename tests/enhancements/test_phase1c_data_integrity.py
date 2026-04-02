"""Tests for Phase 1C: Data Integrity Fixes (C9, C10, C15).

C9: Non-atomic ingestion -- _store_document must wrap insert_one + insert_many
    in run_with_transaction for atomicity.
C10: Validator collection name alignment -- migrate_schema_validation.py must
     use 'ingested_documents' and 'ingested_chunks' matching rag.py pipeline defaults.
C15: Integration test cleanup -- conftest.py rag fixture must call drop_database()
     after yielding for thorough cleanup.

Backing Skill: mongodb-schema-design
"""

import ast
import inspect


class TestC9AtomicIngestion:
    """C9: _store_document must use run_with_transaction for atomic parent+chunks insert."""

    def test_store_document_imports_run_with_transaction(self):
        """pipeline.py must import run_with_transaction from transaction_helper."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(pipeline_module)
        assert "run_with_transaction" in source, (
            "pipeline.py must import run_with_transaction for atomic ingestion"
        )

    def test_store_document_calls_run_with_transaction(self):
        """_store_document must call run_with_transaction, not bare insert_one."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(
            pipeline_module.DocumentIngestionPipeline._store_document
        )
        assert "run_with_transaction" in source, (
            "_store_document must use run_with_transaction to wrap "
            "insert_one + insert_many in a transaction"
        )

    def test_store_document_passes_session_to_insert_one(self):
        """insert_one inside _store_document must receive session= parameter."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(
            pipeline_module.DocumentIngestionPipeline._store_document
        )
        # The inner callback function should pass session to insert_one
        assert "insert_one(" in source and "session=" in source, (
            "insert_one must receive session= parameter for transaction safety"
        )

    def test_store_document_passes_session_to_insert_many(self):
        """insert_many inside _store_document must receive session= parameter."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(
            pipeline_module.DocumentIngestionPipeline._store_document
        )
        # The inner callback function should pass session to insert_many
        assert "insert_many(" in source and "session=" in source, (
            "insert_many must receive session= parameter for transaction safety"
        )

    def test_store_document_gets_client_from_db(self):
        """_store_document must obtain client from self.db.client for transaction."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(
            pipeline_module.DocumentIngestionPipeline._store_document
        )
        # Must get client reference for run_with_transaction
        assert "self.db.client" in source or "db.client" in source, (
            "_store_document must get client from db for run_with_transaction"
        )


class TestC10ValidatorCollectionNames:
    """C10: Validator collection names must match pipeline defaults in rag.py."""

    def test_validators_include_ingested_documents(self):
        """VALIDATORS dict must have 'ingested_documents' key."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        assert "ingested_documents" in VALIDATORS, (
            "VALIDATORS must include 'ingested_documents' to match rag.py defaults"
        )

    def test_validators_include_ingested_chunks(self):
        """VALIDATORS dict must have 'ingested_chunks' key."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        assert "ingested_chunks" in VALIDATORS, (
            "VALIDATORS must include 'ingested_chunks' to match rag.py defaults"
        )

    def test_validators_do_not_use_bare_documents(self):
        """VALIDATORS must NOT use bare 'documents' (without prefix)."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        # The key should be 'ingested_documents', not 'documents'
        # (unless 'documents' is a completely different collection like conversations)
        document_keys = [k for k in VALIDATORS if k.endswith("documents")]
        for key in document_keys:
            assert key != "documents", (
                "VALIDATORS should use 'ingested_documents', not bare 'documents'"
            )

    def test_validators_do_not_use_bare_chunks(self):
        """VALIDATORS must NOT use bare 'chunks' (without prefix)."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        chunk_keys = [k for k in VALIDATORS if k.endswith("chunks")]
        for key in chunk_keys:
            assert key != "chunks", (
                "VALIDATORS should use 'ingested_chunks', not bare 'chunks'"
            )

    def test_ingested_documents_validator_has_required_fields(self):
        """ingested_documents validator must require title, source, content, created_at."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        validator = VALIDATORS["ingested_documents"]
        schema = validator["$jsonSchema"]
        required = schema.get("required", [])
        for field in ["title", "source", "content", "created_at"]:
            assert field in required, (
                f"ingested_documents validator must require '{field}'"
            )

    def test_ingested_chunks_validator_has_required_fields(self):
        """ingested_chunks validator must require document_id, content, embedding, chunk_index."""
        from hybridrag.migrations.migrate_schema_validation import VALIDATORS

        validator = VALIDATORS["ingested_chunks"]
        schema = validator["$jsonSchema"]
        required = schema.get("required", [])
        for field in ["document_id", "content", "embedding", "chunk_index"]:
            assert field in required, (
                f"ingested_chunks validator must require '{field}'"
            )


class TestC15IntegrationTestCleanup:
    """C15: Integration test conftest.py rag fixture must perform database cleanup."""

    def test_rag_fixture_drops_database(self):
        """The rag fixture must call drop_database() for thorough test cleanup."""
        import tests.integration.conftest as conftest_module

        source = inspect.getsource(conftest_module)

        # Find the rag fixture function and check it has drop_database call
        rag_fixture_source = self._extract_rag_fixture_source(source)

        # Check for uncommented drop_database call
        active_lines = [
            line.strip()
            for line in rag_fixture_source.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        has_drop_database = any("drop_database" in line for line in active_lines)
        assert has_drop_database, (
            "rag fixture must call drop_database() (uncommented) for test cleanup"
        )

    def test_rag_fixture_closes_cleanup_client(self):
        """Cleanup client must be closed after dropping the database."""
        import tests.integration.conftest as conftest_module

        source = inspect.getsource(conftest_module)
        rag_fixture_source = self._extract_rag_fixture_source(source)

        # Must close the client used for cleanup
        assert "close()" in rag_fixture_source, (
            "rag fixture must close the cleanup client after drop_database"
        )

    def _extract_rag_fixture_source(self, module_source: str) -> str:
        """Extract the source of the rag fixture function."""
        tree = ast.parse(module_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == "rag":
                    lines = module_source.split("\n")
                    start = node.lineno - 1
                    end = node.end_lineno
                    return "\n".join(lines[start:end])
        raise AssertionError("Could not find 'rag' fixture function in conftest.py")

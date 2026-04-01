"""Tests for Phase 2C: Data Integrity & Atomicity (H12, H16, H17).

H12: remove_nodes non-atomic -- must wrap edge+node delete_many in transaction.
H16: clear_session and delete_session non-atomic -- must wrap in transaction.
H17: Naive datetime mixed with UTC -- pipeline.py must use datetime.now(timezone.utc).

Backing Skill: mongodb-schema-design
"""

import inspect
import re


class TestH12RemoveNodesAtomic:
    """H12: remove_nodes must use run_with_transaction for atomic edge+node deletion."""

    def test_remove_nodes_uses_run_with_transaction(self):
        """remove_nodes must call run_with_transaction to wrap both deletes."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage.remove_nodes)
        assert "run_with_transaction" in source, (
            "remove_nodes must use run_with_transaction for atomic edge+node deletion"
        )

    def test_remove_nodes_passes_session_to_edge_delete(self):
        """Edge delete_many inside remove_nodes must receive session= parameter."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage.remove_nodes)
        # Must pass session to both delete_many calls
        assert source.count("session=session") >= 2 or (
            source.count("session=") >= 2
        ), "Both delete_many calls must receive session= for transaction safety"

    def test_remove_nodes_gets_client_for_transaction(self):
        """remove_nodes must obtain a client reference for run_with_transaction."""
        from hybridrag.engine.kg.mongo_impl import MongoGraphStorage

        source = inspect.getsource(MongoGraphStorage.remove_nodes)
        assert "client" in source and (
            "collection.database.client" in source or "self.db.client" in source
        ), "remove_nodes must get client reference for run_with_transaction"


class TestH16SessionOperationsAtomic:
    """H16: clear_session and delete_session must use transactions."""

    def test_clear_session_uses_run_with_transaction(self):
        """clear_session must call run_with_transaction for atomicity."""
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory.clear_session)
        assert "run_with_transaction" in source, (
            "clear_session must use run_with_transaction for atomic "
            "message deletion + session update"
        )

    def test_clear_session_passes_session_to_operations(self):
        """Both delete_many and update_one in clear_session must receive session=."""
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory.clear_session)
        # At least 2 session= occurrences: delete_many and update_one
        assert source.count("session=") >= 2, (
            "clear_session must pass session= to both delete_many and update_one"
        )

    def test_delete_session_uses_run_with_transaction(self):
        """delete_session must call run_with_transaction for atomicity."""
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory.delete_session)
        assert "run_with_transaction" in source, (
            "delete_session must use run_with_transaction for atomic "
            "message deletion + session deletion"
        )

    def test_delete_session_passes_session_to_operations(self):
        """Both delete_many and delete_one in delete_session must receive session=."""
        from hybridrag.memory.conversation import ConversationMemory

        source = inspect.getsource(ConversationMemory.delete_session)
        # At least 2 session= occurrences: delete_many and delete_one
        assert source.count("session=") >= 2, (
            "delete_session must pass session= to both delete_many and delete_one"
        )


class TestH17TimezoneAwareDatetime:
    """H17: pipeline.py must use timezone-aware datetime.now(timezone.utc), not naive datetime.now()."""

    def test_pipeline_imports_timezone(self):
        """pipeline.py must import timezone from datetime module."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(pipeline_module)
        assert "timezone" in source or "UTC" in source, (
            "pipeline.py must import timezone (or UTC) from datetime for aware datetimes"
        )

    def test_pipeline_no_naive_datetime_now(self):
        """pipeline.py must not contain bare datetime.now() -- must use timezone.utc or UTC."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(pipeline_module)
        # Find all datetime.now() calls that do NOT have timezone.utc or UTC argument
        naive_calls = re.findall(r"datetime\.now\(\)", source)
        assert len(naive_calls) == 0, (
            f"pipeline.py contains {len(naive_calls)} naive datetime.now() calls. "
            "All must use datetime.now(timezone.utc) or datetime.now(UTC)"
        )

    def test_store_document_uses_utc_datetime(self):
        """_store_document created_at fields must use timezone-aware datetime."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(
            pipeline_module.DocumentIngestionPipeline._store_document
        )
        # Must not contain bare datetime.now()
        naive_calls = re.findall(r"datetime\.now\(\)", source)
        assert len(naive_calls) == 0, (
            f"_store_document contains {len(naive_calls)} naive datetime.now() calls. "
            "Must use datetime.now(timezone.utc) or datetime.now(UTC)"
        )

    def test_calc_time_ms_uses_utc_datetime(self):
        """_calc_time_ms must use timezone-aware datetime for time calculation."""
        from hybridrag.ingestion import pipeline as pipeline_module

        source = inspect.getsource(
            pipeline_module.DocumentIngestionPipeline._calc_time_ms
        )
        naive_calls = re.findall(r"datetime\.now\(\)", source)
        assert len(naive_calls) == 0, (
            "_calc_time_ms contains naive datetime.now(). "
            "Must use datetime.now(timezone.utc) or datetime.now(UTC)"
        )

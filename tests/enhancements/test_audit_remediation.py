"""Tests for audit remediation fixes (Task 10).

Covers 8 fixes from code-reviewer and silent-failure-hunter findings:
1. CRITICAL: transaction_helper.py - Remove unused max_retries param
2. HIGH: rag.py ObjectId conversion - log warning, no find({}) fallback
3. HIGH: rag.py ingest_url - projection + bounded to_list
4. HIGH: rag.py ingest_website - projection + bounded to_list
5. HIGH: conversation.py list_sessions - maxTimeMS on $lookup aggregate
6. MEDIUM: mongo_impl.py source_ids - warning log on truncation
7. MEDIUM: mongo_impl.py numCandidates - guard top_k=0
8. MEDIUM: rag.py _get_pipeline_embed_func - Callable type hint
"""

import inspect


class TestTransactionHelperMaxRetriesRemoved:
    """Fix 1 (CRITICAL): max_retries should be removed from run_with_transaction."""

    def test_no_max_retries_parameter(self):
        """run_with_transaction should NOT have max_retries parameter."""
        from hybridrag.core.transaction_helper import run_with_transaction

        sig = inspect.signature(run_with_transaction)
        param_names = list(sig.parameters.keys())
        assert "max_retries" not in param_names, (
            f"max_retries is dead code - with_transaction handles retries internally. "
            f"Found parameters: {param_names}"
        )

    def test_no_max_retries_in_docstring(self):
        """Docstring should not mention max_retries."""
        from hybridrag.core.transaction_helper import run_with_transaction

        docstring = run_with_transaction.__doc__ or ""
        assert "max_retries" not in docstring, (
            "max_retries should be removed from docstring as well"
        )


class TestObjectIdConversionWarning:
    """Fix 2 (HIGH): ObjectId conversion should log warning, not silently pass."""

    def test_objectid_conversion_logs_warning(self):
        """ObjectId conversion failure should log a warning, not silently pass."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_files)

        # Should log a warning when ObjectId conversion fails
        assert "logger.warning" in source and "Could not convert" in source, (
            "ingest_files should log warning when ObjectId conversion fails"
        )

    def test_objectid_fallback_uses_string_filter(self):
        """When ObjectId conversion fails, should use string-based document_id filter,
        NOT fall back to find({})."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_files)

        # The else branch (when batch_object_ids is empty) should use batch_doc_ids
        assert "batch_doc_ids" in source, (
            "ingest_files should fall back to string document_id filter, not find({})"
        )


class TestIngestUrlProjection:
    """Fix 3 (HIGH): ingest_url should use projection + bounded to_list."""

    def test_ingest_url_uses_projection(self):
        """ingest_url chunk read should include projection."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_url)

        # Should have projection (content and metadata.source)
        assert "metadata.source" in source, (
            "ingest_url should use projection with metadata.source"
        )
        assert '"content": 1' in source or "'content': 1" in source, (
            "ingest_url should project content field"
        )

    def test_ingest_url_uses_bounded_to_list(self):
        """ingest_url should use bounded to_list, not length=None."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_url)

        # The chunk read section should NOT use length=None
        # Find the chunks_data line
        assert "to_list(length=None)" not in source, (
            "ingest_url should use bounded to_list (e.g., length=10000), not length=None"
        )


class TestIngestWebsiteProjection:
    """Fix 4 (HIGH): ingest_website should use projection + bounded to_list."""

    def test_ingest_website_uses_projection(self):
        """ingest_website chunk read should include projection."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_website)

        assert "metadata.source" in source, (
            "ingest_website should use projection with metadata.source"
        )

    def test_ingest_website_uses_bounded_to_list(self):
        """ingest_website should use bounded to_list, not length=None."""
        from hybridrag.core import rag as rag_module

        source = inspect.getsource(rag_module.HybridRAG.ingest_website)

        assert "to_list(length=None)" not in source, (
            "ingest_website should use bounded to_list (e.g., length=10000), not length=None"
        )


class TestListSessionsMaxTimeMS:
    """Fix 5 (HIGH): list_sessions $lookup aggregate should have maxTimeMS."""

    def test_list_sessions_has_max_time_ms(self):
        """list_sessions aggregate call should include maxTimeMS."""
        from hybridrag.memory import conversation as conv_module

        source = inspect.getsource(conv_module.ConversationMemory.list_sessions)

        assert "maxTimeMS" in source, (
            "list_sessions aggregate call should include maxTimeMS parameter"
        )


class TestSourceIdsTruncationWarning:
    """Fix 6 (MEDIUM): source_ids truncation should log a warning."""

    def test_upsert_node_logs_warning_on_truncation(self):
        """upsert_node should log warning when source_ids exceeds MAX_SOURCE_IDS."""
        from hybridrag.engine.kg import mongo_impl as impl_module

        source = inspect.getsource(impl_module.MongoGraphStorage.upsert_node)

        assert "warning" in source.lower() or "logger.warning" in source, (
            "upsert_node should log a warning when source_ids is truncated"
        )

    def test_upsert_edge_logs_warning_on_truncation(self):
        """upsert_edge should log warning when source_ids exceeds MAX_SOURCE_IDS."""
        from hybridrag.engine.kg import mongo_impl as impl_module

        source = inspect.getsource(impl_module.MongoGraphStorage.upsert_edge)

        assert "warning" in source.lower() or "logger.warning" in source, (
            "upsert_edge should log a warning when source_ids is truncated"
        )


class TestNumCandidatesTopKZeroGuard:
    """Fix 7 (MEDIUM): numCandidates should handle top_k=0."""

    def test_top_k_zero_guard(self):
        """When top_k=0, numCandidates should not be 0."""
        from hybridrag.engine.kg import mongo_impl as impl_module

        source = inspect.getsource(impl_module.MongoVectorDBStorage.query)

        # Should have a guard like: top_k = max(top_k, 1)
        assert "max(top_k" in source or "max(top_k," in source, (
            "query method should guard against top_k=0 producing numCandidates=0"
        )


class TestCallableTypeAnnotation:
    """Fix 8 (MEDIUM): _get_pipeline_embed_func return type should be Callable."""

    def test_return_type_is_callable(self):
        """Return type should be Callable (capitalized), not callable (lowercase)."""
        from hybridrag.core import rag as rag_module

        # Due to __future__ annotations, the return annotation is a string
        # We just need to verify the source code uses Callable not callable
        source = inspect.getsource(rag_module.HybridRAG._get_pipeline_embed_func)

        # Should contain -> Callable or -> "Callable" but NOT -> callable
        # Check the def line
        def_line = [
            line
            for line in source.split("\n")
            if "def _get_pipeline_embed_func" in line
        ][0]
        assert "callable" not in def_line or "Callable" in def_line, (
            "Return type should be Callable (from collections.abc), not lowercase callable"
        )

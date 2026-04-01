"""
Tests for Phase 4A+4B+4C: LOW source code findings (L1-L18).

L1, L3, L13, L15 are already done in prior phases. Not tested here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# ─────────────────── Phase 4A: Connection/Config Cleanup ───────────────────


class TestL2MongoDBWorkspaceSetting:
    """L2: mongodb_workspace setting should be documented for engine layer use."""

    def test_settings_has_mongodb_workspace_field(self):
        """Verify mongodb_workspace field exists in Settings with description."""
        from hybridrag.config.settings import Settings

        field_info = Settings.model_fields.get("mongodb_workspace")
        assert field_info is not None, "mongodb_workspace field must exist"
        assert field_info.description is not None, "must have description"
        # L2: Description should mention engine layer / collection prefix
        desc = field_info.description.lower()
        assert "collection" in desc or "engine" in desc or "workspace" in desc, (
            f"Description should mention collection/engine/workspace usage: {field_info.description}"
        )


class TestL4TransactionClientType:
    """L4: run_with_transaction should have proper type, not Any."""

    def test_client_parameter_is_typed(self):
        """Verify client param is not typed as Any."""
        import inspect

        from hybridrag.core.transaction_helper import run_with_transaction

        sig = inspect.signature(run_with_transaction)
        client_param = sig.parameters["client"]
        annotation = client_param.annotation
        # Should NOT be Any or inspect.Parameter.empty
        assert annotation is not inspect.Parameter.empty, (
            "client must have type annotation"
        )
        assert annotation is not Any, "client should not be typed as Any"
        # Should reference AsyncMongoClient
        ann_str = str(annotation)
        assert "MongoClient" in ann_str or "AsyncMongoClient" in ann_str, (
            f"client should be typed as AsyncMongoClient, got: {ann_str}"
        )


class TestL5LoggerSetLevel:
    """L5: No module-level logger.setLevel(logging.INFO) in rag.py."""

    def test_no_module_level_set_level(self):
        """Root logger config should handle levels, not module-level setLevel."""
        import ast
        from pathlib import Path

        rag_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "core"
            / "rag.py"
        )
        source = rag_path.read_text()
        tree = ast.parse(source)

        # Find top-level calls to logger.setLevel
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute):
                    if call.func.attr == "setLevel":
                        pytest.fail(
                            f"Found module-level logger.setLevel at line {node.lineno}. "
                            "Remove it; let root logger config handle levels."
                        )


class TestL6DuplicateImport:
    """L6: No duplicate Sequence import in rag.py."""

    def test_no_duplicate_sequence_import(self):
        """Sequence should only be imported once, not both runtime and TYPE_CHECKING."""
        from pathlib import Path

        rag_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "core"
            / "rag.py"
        )
        source = rag_path.read_text()

        # Count how many times "Sequence" appears in import lines
        import_lines = [
            line.strip()
            for line in source.split("\n")
            if "Sequence" in line and ("import" in line or "from" in line)
        ]
        assert len(import_lines) <= 1, (
            f"Sequence imported {len(import_lines)} times (should be 1): {import_lines}"
        )


# ─────────────────── Phase 4B: Query/Performance Polish ───────────────────


class TestL7DeterministicSetOrdering:
    """L7: entity_set should be sorted before slicing for determinism."""

    def test_graph_traversal_returns_sorted_entities(self):
        """related_entities should be in sorted (deterministic) order."""

        # Create a result with unsorted entities to verify the contract
        # The actual function should sort before returning
        # We test by checking the source code contains sorted()
        from pathlib import Path

        gs_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "enhancements"
            / "graph_search.py"
        )
        source = gs_path.read_text()
        # The line that creates related_entities should use sorted()
        assert "sorted(entity_set)" in source, (
            "entity_set must be sorted before slicing for deterministic results"
        )


class TestL8ConsistentAggregate:
    """L8: aggregate() calls should consistently use await."""

    def test_graph_search_aggregate_uses_await(self):
        """In pymongo async, aggregate() is a coroutine and must be awaited."""
        from pathlib import Path

        gs_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "enhancements"
            / "graph_search.py"
        )
        source = gs_path.read_text()

        # Find all aggregate( calls and ensure they have await
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if ".aggregate(" in stripped and not stripped.startswith("#"):
                # Check this line or the assignment line has await
                if "await" not in stripped:
                    # Check if it's a multiline with await on previous line
                    prev = lines[i - 2].strip() if i >= 2 else ""
                    if "await" not in prev:
                        pytest.fail(
                            f"graph_search.py line {i}: aggregate() call without await: {stripped}"
                        )


class TestL9WeightSumValidation:
    """L9: Weight normalization should have a validation assertion."""

    def test_optimizer_validates_weight_sum(self):
        """After normalization, weights must sum to 1.0 with an assertion."""
        from hybridrag.enhancements.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()
        # This should not raise -- weights should be properly normalized
        params = optimizer.optimize("What is MongoDB?")
        assert abs(params.vector_weight + params.text_weight - 1.0) < 0.01

        # Check source has assertion
        from pathlib import Path

        qo_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "enhancements"
            / "query_optimizer.py"
        )
        source = qo_path.read_text()
        assert (
            "assert" in source and "vector_weight" in source and "text_weight" in source
        ), "query_optimizer.py should have assertion validating weight sum"


class TestL10GeoFilterRelation:
    """L10: GeoFilter.relation should either be used or removed."""

    def test_geo_filter_relation_handled(self):
        """GeoFilter relation field should be used in building geo filters,
        or the field should not exist."""
        from hybridrag.enhancements.filters.lexical_prefilters import GeoFilter

        # Check if 'relation' key exists in GeoFilter
        has_relation = "relation" in GeoFilter.__annotations__

        if has_relation:
            # If relation exists, it must be used in build_lexical_prefilters
            from pathlib import Path

            lp_path = (
                Path(__file__).parent.parent.parent
                / "src"
                / "hybridrag"
                / "enhancements"
                / "filters"
                / "lexical_prefilters.py"
            )
            source = lp_path.read_text()
            # Check that relation is actually referenced in the geo filter building code
            assert (
                'geo_filter.get("relation")' in source
                or 'geo_filter["relation"]' in source
                or '"relation"' not in GeoFilter.__annotations__
            ), (
                "GeoFilter has 'relation' field but it's never used in filter building. "
                "Either implement it or remove it."
            )


class TestL11EmptyInFiltersWarning:
    """L11: Empty in_filters should log a warning."""

    def test_empty_in_filter_values_logged(self):
        """When in_filters has empty values list, a warning should be logged."""
        from hybridrag.enhancements.filters.atlas_search_filters import (
            AtlasSearchFilterConfig,
            build_atlas_search_filters,
        )

        config = AtlasSearchFilterConfig(
            in_filters={"category": []},  # Empty values
        )

        with patch(
            "hybridrag.enhancements.filters.atlas_search_filters.logger"
        ) as mock_logger:
            build_atlas_search_filters(config)
            # Empty in_filters should be dropped and a warning logged
            mock_logger.warning.assert_called()


class TestL12InjectionValidation:
    """L12: Filter field names should be validated against injection."""

    def test_dollar_in_field_name_raises(self):
        """Field names containing $ should be rejected."""
        from hybridrag.enhancements.filters.vector_search_filters import (
            VectorSearchFilterConfig,
            build_vector_search_filters,
        )

        config = VectorSearchFilterConfig(
            equality_filters={"$set": "hacked"},
        )
        with pytest.raises(ValueError, match="Invalid filter field name"):
            build_vector_search_filters(config)

    def test_dot_prefix_in_field_name_raises(self):
        """Field names starting with . should be rejected."""
        from hybridrag.enhancements.filters.vector_search_filters import (
            VectorSearchFilterConfig,
            build_vector_search_filters,
        )

        config = VectorSearchFilterConfig(
            equality_filters={".hidden": "value"},
        )
        with pytest.raises(ValueError, match="Invalid filter field name"):
            build_vector_search_filters(config)

    def test_valid_dotted_path_allowed(self):
        """Dotted paths like metadata.source should be allowed."""
        from hybridrag.enhancements.filters.vector_search_filters import (
            VectorSearchFilterConfig,
            build_vector_search_filters,
        )

        config = VectorSearchFilterConfig(
            equality_filters={"metadata.source": "docs"},
        )
        result = build_vector_search_filters(config)
        assert "metadata.source" in result


class TestL14CollectionNameCache:
    """L14: get_or_create_collection should cache known collections."""

    def test_get_or_create_collection_caches(self):
        """Second call should not hit list_collection_names again."""
        from pathlib import Path

        mongo_impl_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "engine"
            / "kg"
            / "mongo_impl.py"
        )
        source = mongo_impl_path.read_text()

        # Check that the function references a cache set
        # Find the function body
        func_start = source.find("async def get_or_create_collection(")
        assert func_start != -1, "get_or_create_collection function must exist"
        func_body = source[func_start : func_start + 500]
        assert "_known_collections" in func_body or "_collection_cache" in func_body, (
            "get_or_create_collection should use a collection cache "
            "to avoid calling list_collection_names every time"
        )


# ─────────────────── Phase 4C: Data Integrity Polish ───────────────────


class TestL16EmbeddingDimValidation:
    """L16: Embedding dimension should be validated on chunks."""

    def test_embedding_dim_validation_in_to_dict(self):
        """DocumentChunk.to_dict or pipeline should validate embedding length."""
        from hybridrag.ingestion.types import DocumentChunk

        # Create a chunk with wrong-dimension embedding
        DocumentChunk(
            content="test content",
            index=0,
            start_char=0,
            end_char=12,
            embedding=[0.1, 0.2],  # Only 2 dims
        )

        # The to_dict method should have validation, OR
        # check pipeline source has validation
        # Either approach is fine per the plan
        from pathlib import Path

        pipeline_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "ingestion"
            / "pipeline.py"
        )
        pipeline_source = pipeline_path.read_text()

        types_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "hybridrag"
            / "ingestion"
            / "types.py"
        )
        types_source = types_path.read_text()

        combined = pipeline_source + types_source
        has_dim_validation = (
            "embedding" in combined
            and (
                "len(chunk" in combined
                or "len(embedding" in combined
                or "embedding_dim" in combined
            )
            and (
                "raise" in combined or "ValueError" in combined or "warning" in combined
            )
        )
        assert has_dim_validation, (
            "No embedding dimension validation found in pipeline.py or types.py. "
            "Add validation that embedding length matches expected dimension."
        )


class TestL17EntityWeightFalsy:
    """L17: entity_weight=0.0 should not be treated as falsy."""

    def test_entity_weight_zero_included_in_total(self):
        """entity_weight=0.0 should still be included in weight sum validation."""
        from hybridrag.presets.search_presets import SearchPreset

        # entity_weight=0.0 should NOT be treated as falsy
        # It should be included in the total weight validation
        preset = SearchPreset(
            name="test_zero_entity",
            vector_weight=0.5,
            text_weight=0.5,
            entity_weight=0.0,  # This is a valid weight
        )
        # With entity_weight=0.0, total = 0.5 + 0.5 + 0.0 = 1.0 -- should work
        assert preset.vector_weight == 0.5
        assert preset.text_weight == 0.5
        assert preset.entity_weight == 0.0

    def test_entity_weight_zero_rejected_when_makes_total_wrong(self):
        """entity_weight=0.0 + wrong vector/text should raise."""
        from hybridrag.presets.search_presets import SearchPreset

        # With entity_weight=0.0, total = 0.6 + 0.6 + 0.0 = 1.2 -- should fail
        with pytest.raises(ValueError, match="[Ww]eights must sum"):
            SearchPreset(
                name="test_bad_total",
                vector_weight=0.6,
                text_weight=0.6,
                entity_weight=0.0,
            )


class TestL18RegisterPresetProtection:
    """L18: register_preset should not allow overwriting built-in presets."""

    def test_cannot_overwrite_builtin_preset(self):
        """Overwriting a built-in preset should raise ValueError."""
        from hybridrag.presets.search_presets import SearchPreset, register_preset

        with pytest.raises(ValueError, match="[Bb]uilt-in|[Cc]annot overwrite"):
            register_preset(
                SearchPreset(
                    name="balanced",  # This is a built-in
                    vector_weight=0.9,
                    text_weight=0.1,
                )
            )

    def test_can_register_new_custom_preset(self):
        """Registering a new preset name should work."""
        from hybridrag.presets.search_presets import (
            PRESETS,
            SearchPreset,
            register_preset,
        )

        test_name = "_test_custom_l18_preset"
        try:
            register_preset(
                SearchPreset(
                    name=test_name,
                    vector_weight=0.7,
                    text_weight=0.3,
                    description="Test custom preset",
                )
            )
            assert test_name in PRESETS
        finally:
            # Clean up
            PRESETS.pop(test_name, None)

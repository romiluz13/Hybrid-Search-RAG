"""Tests for inline source handling and reference synthesis."""

from hybridrag.core.rag import _build_inline_file_paths, _extract_references_from_query_data


def test_build_inline_file_paths_uses_stable_hashes_without_ids():
    """Inline inserts should receive deterministic synthetic sources."""
    paths = _build_inline_file_paths(["hello world", "vector search"])

    assert paths == [
        "inline://5eb63bbbe01e",
        "inline://333591ffe034",
    ]


def test_extract_references_falls_back_to_entities_when_chunks_absent():
    """Entity provenance should surface references when chunk refs are unavailable."""
    query_data = {
        "data": {
            "references": [],
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Vector Search",
                    "source_id": "chunk-inline-1",
                    "file_path": "inline://chunk-inline-1",
                }
            ],
            "relationships": [],
        }
    }

    assert _extract_references_from_query_data(query_data) == [
        {
            "reference_id": "chunk-inline-1",
            "file_path": "inline://chunk-inline-1",
        }
    ]

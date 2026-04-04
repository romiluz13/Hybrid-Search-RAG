from hybridrag.core.rag import _extract_retrieval_diagnostics


def test_extract_retrieval_diagnostics_reports_manual_fallback() -> None:
    diagnostics = _extract_retrieval_diagnostics(
        {
            "data": {
                "chunks": [
                    {"search_type": "hybrid_rrf_manual"},
                    {"search_type": "vector_only"},
                ]
            }
        }
    )

    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_search_types"] == [
        "hybrid_rrf_manual",
        "vector_only",
    ]


def test_extract_retrieval_diagnostics_reports_native_path() -> None:
    diagnostics = _extract_retrieval_diagnostics(
        {"data": {"chunks": [{"search_type": "hybrid_rrf"}]}}
    )

    assert diagnostics["fallback_used"] is False
    assert diagnostics["search_types"] == ["hybrid_rrf"]


def test_extract_retrieval_diagnostics_defaults_to_primary_path_when_results_exist() -> (
    None
):
    diagnostics = _extract_retrieval_diagnostics(
        {
            "data": {
                "chunks": [{"content": "result without explicit search_type"}],
                "entities": [{"entity_name": "MongoDB"}],
            }
        }
    )

    assert diagnostics["fallback_used"] is False

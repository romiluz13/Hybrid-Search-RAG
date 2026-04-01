"""Tests for lexical prefilter builder."""

from hybridrag.enhancements.filters.lexical_prefilters import (
    LexicalPrefilterConfig,
    build_lexical_prefilters,
)


class TestLexicalPrefilterConfig:
    """Test lexical prefilter configuration."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = LexicalPrefilterConfig()
        assert config.text_filters == []
        assert config.fuzzy_filters == []
        assert config.phrase_filters == []
        assert config.wildcard_filters == []
        assert config.range_filters == {}
        assert config.geo_filters == []
        assert config.query_string_filter is None


class TestBuildLexicalPrefilters:
    """Test lexical prefilter builder for $search.vectorSearch."""

    def test_empty_config_returns_empty_dict(self):
        """Empty config should return empty filter dict."""
        config = LexicalPrefilterConfig()
        result = build_lexical_prefilters(config)
        assert result == {}

    def test_text_filter(self):
        """Text filter uses Atlas 'text' operator."""
        config = LexicalPrefilterConfig(
            text_filters=[{"path": "content", "query": "mongodb"}]
        )
        result = build_lexical_prefilters(config)

        assert "compound" in result
        assert "filter" in result["compound"]
        text_filter = result["compound"]["filter"][0]
        assert text_filter["text"]["query"] == "mongodb"
        assert text_filter["text"]["path"] == "content"

    def test_fuzzy_filter(self):
        """Fuzzy filter provides typo-tolerant matching."""
        config = LexicalPrefilterConfig(
            fuzzy_filters=[
                {
                    "path": "title",
                    "query": "mongoDB",
                    "maxEdits": 2,
                    "prefixLength": 3,
                }
            ]
        )
        result = build_lexical_prefilters(config)

        assert "compound" in result
        fuzzy = result["compound"]["filter"][0]["text"]
        assert fuzzy["query"] == "mongoDB"
        assert fuzzy["fuzzy"]["maxEdits"] == 2
        assert fuzzy["fuzzy"]["prefixLength"] == 3

    def test_phrase_filter(self):
        """Phrase filter matches exact phrases."""
        config = LexicalPrefilterConfig(
            phrase_filters=[
                {
                    "path": "content",
                    "query": "vector search",
                    "slop": 1,
                }
            ]
        )
        result = build_lexical_prefilters(config)

        phrase = result["compound"]["filter"][0]["phrase"]
        assert phrase["query"] == "vector search"
        assert phrase["slop"] == 1

    def test_wildcard_filter(self):
        """Wildcard filter supports * and ? patterns."""
        config = LexicalPrefilterConfig(
            wildcard_filters=[
                {
                    "path": "filename",
                    "query": "*.pdf",
                }
            ]
        )
        result = build_lexical_prefilters(config)

        wildcard = result["compound"]["filter"][0]["wildcard"]
        assert wildcard["query"] == "*.pdf"

    def test_range_filter(self):
        """Range filter uses Atlas range operator."""
        config = LexicalPrefilterConfig(
            range_filters={"timestamp": {"gte": "2024-01-01", "lte": "2024-12-31"}}
        )
        result = build_lexical_prefilters(config)

        range_filter = result["compound"]["filter"][0]["range"]
        assert range_filter["path"] == "timestamp"
        assert range_filter["gte"] == "2024-01-01"
        assert range_filter["lte"] == "2024-12-31"

    def test_geo_filter(self):
        """Geo filter supports geoWithin queries."""
        config = LexicalPrefilterConfig(
            geo_filters=[
                {
                    "path": "location",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-73.97, 40.77],
                                [-73.97, 40.73],
                                [-73.93, 40.73],
                                [-73.93, 40.77],
                                [-73.97, 40.77],
                            ]
                        ],
                    },
                }
            ]
        )
        result = build_lexical_prefilters(config)

        geo = result["compound"]["filter"][0]["geoWithin"]
        assert geo["path"] == "location"
        assert geo["geometry"]["type"] == "Polygon"

    def test_query_string_filter(self):
        """QueryString filter supports Lucene syntax."""
        config = LexicalPrefilterConfig(
            query_string_filter={
                "defaultPath": "content",
                "query": "mongodb AND (vector OR search)",
            }
        )
        result = build_lexical_prefilters(config)

        qs = result["compound"]["filter"][0]["queryString"]
        assert qs["defaultPath"] == "content"
        assert "mongodb AND" in qs["query"]

    def test_equality_filter(self):
        """Equality filter uses equals operator."""
        config = LexicalPrefilterConfig(
            equality_filters={"status": "published", "category": "tech"}
        )
        result = build_lexical_prefilters(config)

        filters = result["compound"]["filter"]
        assert len(filters) == 2
        values = [f["equals"]["value"] for f in filters if "equals" in f]
        assert "published" in values
        assert "tech" in values

    def test_combined_filters(self):
        """Multiple filter types combine correctly."""
        config = LexicalPrefilterConfig(
            text_filters=[{"path": "content", "query": "mongodb"}],
            range_filters={"score": {"gte": 0.5}},
            equality_filters={"active": True},
        )
        result = build_lexical_prefilters(config)

        filters = result["compound"]["filter"]
        assert len(filters) == 3


class TestBuildSearchVectorSearchStage:
    """Test $search.vectorSearch stage builder."""

    def test_basic_stage(self):
        """Basic stage without filters."""
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
        )

        assert "$search" in stage
        assert stage["$search"]["index"] == "my_index"
        vs = stage["$search"]["vectorSearch"]
        assert vs["queryVector"] == [0.1, 0.2, 0.3]
        assert vs["limit"] == 10
        assert vs["numCandidates"] == 200  # 10 * 20

    def test_stage_with_lexical_prefilters(self):
        """Stage with lexical prefilters."""
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        filter_config = LexicalPrefilterConfig(
            text_filters=[{"path": "category", "query": "tech"}],
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            filter_config=filter_config,
        )

        assert "filter" in stage["$search"]["vectorSearch"]
        assert "compound" in stage["$search"]["vectorSearch"]["filter"]

    def test_exact_search_flag(self):
        """Exact search disables ANN."""
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            exact=True,
        )

        assert stage["$search"]["vectorSearch"]["exact"] is True

    def test_exact_search_excludes_num_candidates(self):
        """C8: exact=True and numCandidates are mutually exclusive.

        Per MongoDB docs: numCandidates is required if exact is false or omitted.
        When exact=True, numCandidates MUST NOT be present.
        """
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            exact=True,
        )

        vs = stage["$search"]["vectorSearch"]
        assert vs["exact"] is True
        assert "numCandidates" not in vs, (
            "numCandidates must NOT be present when exact=True"
        )

    def test_non_exact_search_includes_num_candidates(self):
        """When exact is False (default), numCandidates MUST be present."""
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
        )

        vs = stage["$search"]["vectorSearch"]
        assert "numCandidates" in vs
        assert "exact" not in vs

    def test_num_candidates_boundary_limit_1(self):
        """L23: limit=1 should produce numCandidates=20 (1 * 20)."""
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=1,
        )

        vs = stage["$search"]["vectorSearch"]
        assert vs["limit"] == 1
        assert vs["numCandidates"] == 20  # 1 * 20

    def test_num_candidates_boundary_limit_100(self):
        """L23: limit=100 should produce numCandidates=2000 (100 * 20)."""
        from hybridrag.enhancements.filters.lexical_prefilters import (
            build_search_vector_search_stage,
        )

        stage = build_search_vector_search_stage(
            index_name="my_index",
            query_vector=[0.1, 0.2, 0.3],
            limit=100,
        )

        vs = stage["$search"]["vectorSearch"]
        assert vs["limit"] == 100
        assert vs["numCandidates"] == 2000  # 100 * 20

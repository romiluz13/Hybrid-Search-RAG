"""Tests for bson_to_jsonable — recursive BSON-to-JSON sanitizer.

Inspired by the Anthropic CMA cookbook's ``_jsonable`` helper, this utility
ensures MongoDB documents can be safely serialized to JSON for API responses
and tool outputs without ``TypeError: Object of type ObjectId is not JSON
serializable`` crashes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from bson import ObjectId

from hybridrag.engine.utils import bson_to_jsonable

# ---------------------------------------------------------------------------
# Plain value passthrough
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_none_passthrough() -> None:
    """None should pass through unchanged."""
    assert bson_to_jsonable(None) is None


@pytest.mark.p1
def test_str_passthrough() -> None:
    """Strings should pass through unchanged."""
    assert bson_to_jsonable("hello") == "hello"


@pytest.mark.p1
def test_int_passthrough() -> None:
    """Integers should pass through unchanged."""
    assert bson_to_jsonable(42) == 42


@pytest.mark.p1
def test_float_passthrough() -> None:
    """Floats should pass through unchanged."""
    assert bson_to_jsonable(3.14) == 3.14


@pytest.mark.p1
def test_bool_passthrough() -> None:
    """Booleans should pass through unchanged."""
    assert bson_to_jsonable(True) is True


# ---------------------------------------------------------------------------
# BSON type conversions
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_objectid_converted_to_string() -> None:
    """ObjectId should be converted to its string representation."""
    oid = ObjectId()
    result = bson_to_jsonable(oid)
    assert result == str(oid)
    assert isinstance(result, str)


@pytest.mark.p1
def test_datetime_converted_to_isoformat() -> None:
    """datetime should be converted to ISO 8601 string."""
    dt = datetime(2025, 1, 15, 12, 30, 45, tzinfo=UTC)
    result = bson_to_jsonable(dt)
    assert result == dt.isoformat()
    assert isinstance(result, str)


@pytest.mark.p1
def test_bytes_converted_to_string() -> None:
    """bytes should be decoded to a UTF-8 string."""
    result = bson_to_jsonable(b"hello world")
    assert result == "hello world"
    assert isinstance(result, str)


@pytest.mark.p2
def test_bytes_with_invalid_utf8_replaced() -> None:
    """Invalid UTF-8 bytes should be replaced, not crash."""
    result = bson_to_jsonable(b"\xff\xfe")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_nested_dict_with_bson_types() -> None:
    """ObjectId and datetime inside dicts should be converted recursively."""
    oid = ObjectId()
    dt = datetime(2025, 6, 1, tzinfo=UTC)
    data = {"_id": oid, "created_at": dt, "name": "test"}
    result = bson_to_jsonable(data)
    assert result == {
        "_id": str(oid),
        "created_at": dt.isoformat(),
        "name": "test",
    }


@pytest.mark.p1
def test_list_with_bson_types() -> None:
    """ObjectId and datetime inside lists should be converted recursively."""
    oid = ObjectId()
    dt = datetime(2025, 6, 1, tzinfo=UTC)
    data = [oid, dt, "plain", 42]
    result = bson_to_jsonable(data)
    assert result == [str(oid), dt.isoformat(), "plain", 42]


@pytest.mark.p2
def test_set_with_bson_types() -> None:
    """Sets should be converted to lists with BSON types sanitized."""
    oid = ObjectId()
    result = bson_to_jsonable({oid, "text"})
    assert str(oid) in result
    assert "text" in result
    assert len(result) == 2


@pytest.mark.p2
def test_tuple_with_bson_types() -> None:
    """Tuples should be converted to lists with BSON types sanitized."""
    oid = ObjectId()
    result = bson_to_jsonable((oid, "text"))
    assert result == [str(oid), "text"]


@pytest.mark.p2
def test_deeply_nested_structure() -> None:
    """Deeply nested dicts/lists should be fully sanitized."""
    oid = ObjectId()
    dt = datetime(2025, 3, 15, 8, 0, tzinfo=UTC)
    data = {
        "outer": [
            {"inner_id": oid, "inner_list": [dt, b"bytes", {"deep": oid}]},
        ],
    }
    result = bson_to_jsonable(data)
    assert result == {
        "outer": [
            {
                "inner_id": str(oid),
                "inner_list": [
                    dt.isoformat(),
                    "bytes",
                    {"deep": str(oid)},
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# JSON serializability guarantee
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_result_is_json_serializable() -> None:
    """The sanitized output must be serializable by json.dumps without error."""
    import json

    oid = ObjectId()
    dt = datetime(2025, 1, 1, tzinfo=UTC)
    data = {
        "_id": oid,
        "created_at": dt,
        "tags": [oid, "tag1", dt],
        "nested": {"id": oid, "time": dt},
    }
    result = bson_to_jsonable(data)
    # This must not raise
    json_str = json.dumps(result)
    assert isinstance(json_str, str)


@pytest.mark.p1
def test_raw_bson_not_json_serializable_without_sanitizer() -> None:
    """Sanity check: raw ObjectId in a dict should fail json.dumps."""
    import json

    oid = ObjectId()
    raw = {"_id": oid}
    with pytest.raises(TypeError):
        json.dumps(raw)

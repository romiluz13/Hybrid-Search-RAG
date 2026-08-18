"""Validated public document metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from .vector_search_filters import (
    FilterOperator,
    FilterPredicate,
    normalize_filter_value,
    validate_mapped_filter_value,
)

DocumentMetadata: TypeAlias = dict[str, Any]


def normalize_document_metadata(
    metadata: Mapping[str, Any],
    field_types: Mapping[str, str] | None = None,
) -> DocumentMetadata:
    """Validate and normalize one document's public metadata.

    Args:
        metadata: Caller-provided metadata fields.
        field_types: Configured Search mapping types keyed by metadata path.

    Returns:
        Metadata normalized to stable BSON-compatible values.

    Raises:
        ValueError: If a key, value, or configured mapping is invalid.
    """

    if len(metadata) > 32:
        raise ValueError("Document metadata supports at most 32 fields")

    normalized: DocumentMetadata = {}
    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 128
            or "." in key
            or "$" in key
            or "\x00" in key
        ):
            raise ValueError("Metadata keys must be safe top-level field names")
        operator: FilterOperator = (
            "in" if isinstance(value, list | tuple | set) else "eq"
        )
        predicate = FilterPredicate(
            field=f"metadata.{key}",
            operator=operator,
            value=value,
        )
        field = f"metadata.{key}"
        if field_types is not None and field in field_types:
            validate_mapped_filter_value(
                field,
                field_types[field],
                operator,
                predicate.value,
            )
        normalized[key] = normalize_filter_value(predicate.value)
    return normalized


def normalize_metadata_batch(
    metadata: Sequence[Mapping[str, Any]] | None,
    document_count: int,
    field_types: Mapping[str, str] | None = None,
) -> list[DocumentMetadata] | None:
    """Validate metadata cardinality and each document metadata object.

    Args:
        metadata: One metadata mapping per document, or ``None``.
        document_count: Number of documents in the ingestion request.
        field_types: Configured Search mapping types keyed by metadata path.

    Returns:
        Normalized metadata in document order, or ``None``.

    Raises:
        ValueError: If cardinality or a metadata value is invalid.
    """

    if metadata is None:
        return None
    if len(metadata) != document_count:
        raise ValueError("Number of metadata objects must match number of documents")
    return [normalize_document_metadata(item, field_types) for item in metadata]

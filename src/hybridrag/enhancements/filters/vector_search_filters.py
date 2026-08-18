"""
Vector Search Filter Builder for MongoDB 8.0+.

CRITICAL: Vector search filters use STANDARD MongoDB operators.
This is DIFFERENT from Atlas Search filters which use Atlas-specific operators.

Reference: ai-agents-meetup/src/lib/search/vector-search.ts

Example $vectorSearch with filters:
{
    "$vectorSearch": {
        "index": "vector_knn_index",
        "path": "embedding",
        "queryVector": [...],
        "numCandidates": 200,
        "filter": {
            "timestamp": {"$gte": start_date, "$lte": end_date},
            "senderName": {"$eq": "John"},
            "category": {"$in": ["tech", "science"]}
        },
        "limit": 10
    }
}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from math import isfinite
from typing import Annotated, Any, Literal, TypeAlias
from uuid import UUID

from bson import ObjectId
from bson.binary import Binary, UuidRepresentation
from pydantic import (
    BaseModel,
    ConfigDict,
    WithJsonSchema,
    field_validator,
    model_validator,
)

FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "exists"]
FilterLogic = Literal["and", "or"]

BsonObjectId = Annotated[
    ObjectId,
    WithJsonSchema({"type": "string", "format": "objectid"}),
]


class ObjectIdFilterValue(BaseModel):
    """Tagged ObjectId accepted by JSON API filter requests."""

    type: Literal["objectId"]
    value: str

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        if not ObjectId.is_valid(value):
            raise ValueError("Tagged ObjectId filter value is invalid")
        return value


class UUIDFilterValue(BaseModel):
    """Tagged UUID accepted by JSON API filter requests."""

    type: Literal["uuid"]
    value: str

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        try:
            UUID(value)
        except ValueError as error:
            raise ValueError("Tagged UUID filter value is invalid") from error
        return value


class DateFilterValue(BaseModel):
    """Tagged ISO date accepted by JSON API filter requests."""

    type: Literal["date"]
    value: str

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Tagged date filter value is invalid") from error
        return value


class DatetimeFilterValue(BaseModel):
    """Tagged timezone-aware ISO datetime accepted by JSON API requests."""

    type: Literal["datetime"]
    value: str

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Tagged datetime filter value is invalid") from error
        if parsed.utcoffset() is None:
            raise ValueError("Tagged datetime filter value must include a timezone")
        return value


FilterScalar: TypeAlias = (
    str
    | int
    | float
    | bool
    | date
    | datetime
    | BsonObjectId
    | UUID
    | ObjectIdFilterValue
    | UUIDFilterValue
    | DateFilterValue
    | DatetimeFilterValue
)
FilterValue: TypeAlias = (
    FilterScalar | list[FilterScalar] | tuple[FilterScalar, ...] | set[FilterScalar]
)


def _bson_value_kind(value: Any) -> str:
    normalized = normalize_filter_value(value)
    if type(normalized) is bool:
        return "boolean"
    if isinstance(normalized, str):
        return "token"
    if isinstance(normalized, int | float) and not isinstance(normalized, bool):
        return "number"
    if isinstance(normalized, datetime):
        return "date"
    if isinstance(normalized, ObjectId):
        return "objectId"
    if isinstance(normalized, Binary) and normalized.subtype == 4:
        return "uuid"
    return "unsupported"


def validate_mapped_filter_value(
    field: str,
    field_type: str,
    operator: FilterOperator,
    value: FilterValue,
) -> None:
    """Validate a filter or ingested value against its Search mapping."""
    if operator in {"gt", "gte", "lt", "lte"} and field_type not in {
        "number",
        "date",
    }:
        raise ValueError(
            f"Filter range operators are not supported for {field_type} field {field}"
        )
    if operator == "exists":
        return

    values = value if isinstance(value, list | tuple | set) else [value]
    for item in values:
        normalized = normalize_filter_value(item)
        kind = _bson_value_kind(normalized)
        if kind != field_type:
            raise ValueError(
                f"{field} requires values matching its {field_type} mapping"
            )
        if field_type == "token" and len(normalized.encode("utf-8")) > 8181:
            raise ValueError(f"{field} token values cannot exceed 8181 UTF-8 bytes")
        if (
            field_type == "number"
            and isinstance(normalized, int)
            and not (-(2**63) <= normalized <= 2**63 - 1)
        ):
            raise ValueError(f"{field} integers must fit BSON int64")


def validate_filter_config_for_mappings(
    config: FilterConfig,
    field_types: Mapping[str, str],
) -> None:
    """Validate every predicate against the configured Search field type."""
    for predicate in config.predicates:
        validate_mapped_filter_value(
            predicate.field,
            field_types[predicate.field],
            predicate.operator,
            predicate.value,
        )


def _validate_filter_scalar(value: Any) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Numeric filter values must be finite")
    if isinstance(value, datetime) and value.utcoffset() is None:
        raise ValueError("Datetime filter values must be timezone-aware")
    if not isinstance(
        value,
        str
        | int
        | float
        | bool
        | date
        | ObjectId
        | UUID
        | ObjectIdFilterValue
        | UUIDFilterValue
        | DateFilterValue
        | DatetimeFilterValue,
    ):
        raise ValueError("Filter values must be supported scalar values")


def normalize_filter_value(value: FilterValue) -> Any:
    """Convert accepted filter values to stable BSON-compatible values.

    Args:
        value: Accepted scalar or scalar collection.

    Returns:
        A BSON-encodable scalar or list with stable UUID and date encoding.
    """

    if isinstance(value, ObjectIdFilterValue):
        return ObjectId(value.value)
    if isinstance(value, UUIDFilterValue):
        return Binary.from_uuid(
            UUID(value.value),
            uuid_representation=UuidRepresentation.STANDARD,
        )
    if isinstance(value, DateFilterValue):
        parsed = date.fromisoformat(value.value)
        return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
    if isinstance(value, DatetimeFilterValue):
        return datetime.fromisoformat(value.value.replace("Z", "+00:00"))
    if isinstance(value, list | tuple | set):
        return [normalize_filter_value(item) for item in value]
    if isinstance(value, UUID):
        return Binary.from_uuid(
            value,
            uuid_representation=UuidRepresentation.STANDARD,
        )
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return value


class FilterPredicate(BaseModel):
    """One backend-neutral metadata predicate."""

    field: str
    operator: FilterOperator
    value: FilterValue

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("value", mode="before")
    @classmethod
    def validate_value_shape(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            if value.get("type") in {"objectId", "uuid", "date", "datetime"} and set(
                value
            ) == {
                "type",
                "value",
            }:
                return value
            raise ValueError("Filter values must be scalar values or scalar lists")
        if isinstance(
            value,
            str
            | int
            | float
            | bool
            | date
            | ObjectId
            | UUID
            | ObjectIdFilterValue
            | UUIDFilterValue
            | DateFilterValue
            | DatetimeFilterValue
            | list
            | tuple
            | set,
        ):
            return value
        raise ValueError("Filter values must be supported scalar values")

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        value = value.strip()
        if (
            not value
            or len(value) > 256
            or "$" in value
            or "\x00" in value
            or any(not segment for segment in value.split("."))
        ):
            raise ValueError("Filter field must be a valid dotted field path")
        return value

    @model_validator(mode="after")
    def validate_value(self) -> FilterPredicate:
        if self.operator in {"in", "nin"}:
            if not isinstance(self.value, list | tuple | set):
                raise ValueError(
                    f"Filter operator '{self.operator}' requires a list value"
                )
            if not 1 <= len(self.value) <= 100:
                raise ValueError(
                    "Filter membership lists must contain between 1 and 100 values"
                )
        elif isinstance(self.value, list | tuple | set):
            raise ValueError(
                f"Filter scalar operator '{self.operator}' requires a scalar value"
            )
        if self.operator == "exists" and type(self.value) is not bool:
            raise ValueError("Filter operator 'exists' requires a boolean value")
        if isinstance(self.value, list | tuple | set) and any(
            isinstance(item, Mapping | list | tuple | set) for item in self.value
        ):
            raise ValueError("Filter lists must contain scalar values")
        values = (
            self.value if isinstance(self.value, list | tuple | set) else [self.value]
        )
        for value in values:
            _validate_filter_scalar(value)
        if len({_bson_value_kind(value) for value in values}) > 1:
            raise ValueError("Filter membership values must use one BSON type")
        return self


class FilterConfig(BaseModel):
    """A conjunction or disjunction of public metadata predicates."""

    predicates: list[FilterPredicate]
    logic: FilterLogic = "and"
    negate: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("predicates")
    @classmethod
    def validate_predicates(cls, value: list[FilterPredicate]) -> list[FilterPredicate]:
        if not 1 <= len(value) <= 32:
            raise ValueError("FilterConfig requires between 1 and 32 predicates")
        return value


@dataclass(frozen=True)
class RetrievalSecurityContext:
    """Server-owned predicates that callers cannot weaken or replace."""

    mandatory_filter: FilterConfig
    additional_mandatory_filters: tuple[FilterConfig, ...] = ()

    @property
    def mandatory_filters(self) -> tuple[FilterConfig, ...]:
        """Return every server-owned expression in conjunction order."""
        return (self.mandatory_filter, *self.additional_mandatory_filters)


def _mql_predicate(predicate: FilterPredicate) -> dict[str, Any]:
    value = normalize_filter_value(predicate.value)
    return {predicate.field: {f"${predicate.operator}": value}}


def compile_filter_to_mql(config: FilterConfig) -> dict[str, Any]:
    """Compile a public expression to MongoDB Vector Search MQL syntax."""

    predicates = [_mql_predicate(predicate) for predicate in config.predicates]
    expression = (
        predicates[0] if len(predicates) == 1 else {f"${config.logic}": predicates}
    )
    return {"$nor": [expression]} if config.negate else expression


def _atlas_predicate(predicate: FilterPredicate) -> dict[str, Any]:
    value = normalize_filter_value(predicate.value)
    if predicate.operator == "exists":
        exists = {"exists": {"path": predicate.field}}
        return exists if value else {"compound": {"mustNot": [exists]}}
    if predicate.operator == "eq":
        return {"equals": {"path": predicate.field, "value": value}}
    if predicate.operator == "ne":
        return {
            "compound": {
                "mustNot": [{"equals": {"path": predicate.field, "value": value}}]
            }
        }
    if predicate.operator == "in":
        return {"in": {"path": predicate.field, "value": value}}
    if predicate.operator == "nin":
        return {
            "compound": {"mustNot": [{"in": {"path": predicate.field, "value": value}}]}
        }
    return {"range": {"path": predicate.field, predicate.operator: value}}


def compile_filter_to_atlas(config: FilterConfig) -> list[dict[str, Any]]:
    """Compile a public expression to Atlas Search compound-filter syntax."""

    predicates = [_atlas_predicate(predicate) for predicate in config.predicates]
    expression: dict[str, Any]
    if config.logic == "and":
        filters = predicates
        expression = (
            predicates[0]
            if len(predicates) == 1
            else {"compound": {"filter": predicates}}
        )
    else:
        expression = {"compound": {"should": predicates, "minimumShouldMatch": 1}}
        filters = [expression]
    return [{"compound": {"mustNot": [expression]}}] if config.negate else filters


def compile_retrieval_filter_to_mql(
    public_filter: FilterConfig | None,
    security_context: RetrievalSecurityContext | None,
) -> dict[str, Any] | None:
    """Compile the server and caller constraints as one mandatory expression."""
    expressions: list[dict[str, Any]] = []
    if security_context is not None:
        expressions.extend(
            compile_filter_to_mql(mandatory_filter)
            for mandatory_filter in security_context.mandatory_filters
        )
    if public_filter is not None:
        expressions.append(compile_filter_to_mql(public_filter))
    if not expressions:
        return None
    return expressions[0] if len(expressions) == 1 else {"$and": expressions}


def compile_retrieval_filter_to_atlas(
    public_filter: FilterConfig | None,
    security_context: RetrievalSecurityContext | None,
) -> list[dict[str, Any]]:
    """Compile server and caller constraints into Atlas Search filter clauses."""
    clauses: list[dict[str, Any]] = []
    if security_context is not None:
        for mandatory_filter in security_context.mandatory_filters:
            clauses.extend(compile_filter_to_atlas(mandatory_filter))
    if public_filter is not None:
        clauses.extend(compile_filter_to_atlas(public_filter))
    return clauses


@dataclass
class VectorSearchFilterConfig:
    """Configuration for vector search filters.

    All filters use STANDARD MongoDB operators:
    - Date ranges: $gte, $lte
    - Equality: $eq
    - In-list: $in
    - Comparison: $gt, $lt, $ne
    """

    # Date range filters
    start_date: datetime | None = None
    end_date: datetime | None = None
    timestamp_field: str = "timestamp"

    # Equality filters: {field_name: value}
    equality_filters: dict[str, Any] = field(default_factory=dict)

    # In-list filters: {field_name: [values]}
    in_filters: dict[str, list[Any]] = field(default_factory=dict)

    # Comparison filters: {field_name: {"$gt": value}} etc.
    comparison_filters: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Negation filters: {field_name: value} -> {field_name: {"$ne": value}}
    not_equal_filters: dict[str, Any] = field(default_factory=dict)


def build_vector_search_filters(config: VectorSearchFilterConfig) -> dict[str, Any]:
    """
    Build filter object for $vectorSearch using STANDARD MongoDB operators.

    IMPORTANT: This is for $vectorSearch prefiltering.
    Do NOT use Atlas Search operators (range, equals) here!

    Args:
        config: Filter configuration

    Returns:
        Filter dict using standard MongoDB operators

    Example output:
        {
            "timestamp": {"$gte": datetime(2024,1,1), "$lte": datetime(2024,12,31)},
            "senderName": {"$eq": "John"},
            "category": {"$in": ["tech", "science"]}
        }
    """
    filters: dict[str, Any] = {}

    # [L12] Validate field names to prevent operator injection
    all_field_names = (
        list(config.equality_filters.keys())
        + list(config.in_filters.keys())
        + list(config.comparison_filters.keys())
        + list(config.not_equal_filters.keys())
    )
    for field_name in all_field_names:
        if "$" in field_name or field_name.startswith("."):
            raise ValueError(
                f"Invalid filter field name: '{field_name}'. "
                "Field names must not contain '$' or start with '.'."
            )

    # Date range filters
    if config.start_date or config.end_date:
        date_filter: dict[str, datetime] = {}
        if config.start_date:
            date_filter["$gte"] = config.start_date
        if config.end_date:
            date_filter["$lte"] = config.end_date
        if date_filter:
            filters[config.timestamp_field] = date_filter

    # Equality filters using $eq
    for field_name, value in config.equality_filters.items():
        filters[field_name] = {"$eq": value}

    # In-list filters using $in
    for field_name, values in config.in_filters.items():
        if field_name in filters:
            filters[field_name]["$in"] = values
        else:
            filters[field_name] = {"$in": values}

    # Direct comparison filters (already in MongoDB format)
    for field_name, comparison in config.comparison_filters.items():
        if field_name in filters:
            # Merge with existing filter
            filters[field_name].update(comparison)
        else:
            filters[field_name] = comparison

    # Not-equal filters using $ne
    for field_name, value in config.not_equal_filters.items():
        if field_name in filters:
            filters[field_name]["$ne"] = value
        else:
            filters[field_name] = {"$ne": value}

    return filters


def build_vector_search_stage(
    index_name: str,
    query_vector: list[float],
    limit: int = 10,
    num_candidates: int = 100,
    path: str = "vector",
    filter_config: VectorSearchFilterConfig | None = None,
) -> dict[str, Any]:
    """
    Build complete $vectorSearch aggregation stage.

    Args:
        index_name: Name of the vector search index
        query_vector: Query embedding vector
        limit: Number of results to return
        num_candidates: Number of candidates to consider (should be >= limit * 10)
        path: Path to the vector field in documents
        filter_config: Optional filter configuration

    Returns:
        Complete $vectorSearch stage dict
    """
    stage: dict[str, Any] = {
        "$vectorSearch": {
            "index": index_name,
            "path": path,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }

    # Add filters if provided
    if filter_config:
        filters = build_vector_search_filters(filter_config)
        if filters:
            stage["$vectorSearch"]["filter"] = filters

    return stage

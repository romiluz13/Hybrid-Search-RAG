from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from typing import Any

from hybridrag.enhancements.filters import (
    FilterConfig,
    FilterPredicate,
    RetrievalSecurityContext,
)

_request_security_context: ContextVar[RetrievalSecurityContext | None] = ContextVar(
    "hybridrag_request_security_context",
    default=None,
)


def get_request_security_context() -> RetrievalSecurityContext | None:
    """Return the mandatory retrieval constraints for the current request."""
    return _request_security_context.get()


def set_request_security_context(
    context: RetrievalSecurityContext | None,
) -> Token[RetrievalSecurityContext | None]:
    """Set mandatory retrieval constraints for the current async request."""
    return _request_security_context.set(context)


def reset_request_security_context(
    token: Token[RetrievalSecurityContext | None],
) -> None:
    """Restore the request context after an HTTP request completes."""
    _request_security_context.reset(token)


def resolve_retrieval_security_context(
    *contexts: RetrievalSecurityContext | None,
) -> RetrievalSecurityContext | None:
    """Conjoin static, supplied, and request-scoped mandatory constraints."""
    filters: list[FilterConfig] = []
    for context in (*contexts, get_request_security_context()):
        if context is None:
            continue
        for mandatory_filter in context.mandatory_filters:
            if mandatory_filter not in filters:
                filters.append(mandatory_filter)
    if not filters:
        return None
    return RetrievalSecurityContext(
        mandatory_filter=filters[0],
        additional_mandatory_filters=tuple(filters[1:]),
    )


def _tenant_binding(
    context: RetrievalSecurityContext | None = None,
) -> tuple[str, str] | None:
    tenant_field = os.environ.get("HYBRIDRAG_TENANT_FIELD")
    if not tenant_field:
        return None
    if not tenant_field.startswith("metadata."):
        raise ValueError("HYBRIDRAG_TENANT_FIELD must use a metadata.<field> path")

    metadata_key = tenant_field.removeprefix("metadata.")
    if not metadata_key or "." in metadata_key:
        raise ValueError("tenant metadata must be a safe top-level metadata field")

    resolved_context = context or get_request_security_context()
    if resolved_context is None:
        raise ValueError("tenant-scoped operation requires an authenticated principal")

    for mandatory_filter in resolved_context.mandatory_filters:
        if mandatory_filter.negate or mandatory_filter.logic != "and":
            continue
        for predicate in mandatory_filter.predicates:
            if (
                predicate.field == tenant_field
                and predicate.operator == "eq"
                and isinstance(predicate.value, str)
                and predicate.value
            ):
                return metadata_key, predicate.value
    raise ValueError("authenticated principal has no mandatory tenant constraint")


def scope_document_metadata(
    metadata: Sequence[Mapping[str, Any]] | None,
    document_count: int,
    context: RetrievalSecurityContext | None = None,
) -> list[dict[str, Any]] | None:
    """Stamp trusted tenant ownership onto ingestion metadata.

    Args:
        metadata: Caller-provided metadata in document order.
        document_count: Number of documents being ingested.
        context: Authenticated server-owned constraints.

    Returns:
        Copied metadata with authoritative tenant ownership when configured.

    Raises:
        ValueError: If tenant configuration or metadata cardinality is invalid.
    """
    binding = _tenant_binding(context)
    if binding is None:
        return None if metadata is None else [dict(item) for item in metadata]
    if metadata is not None and len(metadata) != document_count:
        raise ValueError("Number of metadata objects must match number of documents")

    metadata_key, tenant_id = binding
    scoped = [dict(item) for item in metadata] if metadata is not None else []
    scoped.extend({} for _ in range(document_count - len(scoped)))
    for item in scoped:
        item[metadata_key] = tenant_id
    return scoped


def require_document_ownership(
    document: Mapping[str, Any] | object | None,
    context: RetrievalSecurityContext | None = None,
) -> None:
    """Fail closed when a tenant-scoped principal does not own a document.

    Args:
        document: Stored document status record.
        context: Authenticated server-owned constraints.

    Raises:
        PermissionError: If the document is absent or owned by another tenant.
        ValueError: If tenant configuration or authentication is invalid.
    """
    binding = _tenant_binding(context)
    if binding is None:
        return
    metadata_key, tenant_id = binding
    if document is None:
        raise PermissionError("document not found")
    if isinstance(document, Mapping):
        metadata = document.get("metadata")
    else:
        metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get(metadata_key) != tenant_id:
        raise PermissionError("document not found")


def require_unscoped_document_operation(
    context: RetrievalSecurityContext | None = None,
) -> None:
    """Reject collection-wide document mutations for scoped principals.

    Args:
        context: Authenticated server-owned constraints.

    Raises:
        PermissionError: If the request carries tenant or ACL constraints.
    """
    if context is not None or get_request_security_context() is not None:
        raise PermissionError("global document operation requires unscoped access")


def api_key_security_context(api_key: str | None) -> RetrievalSecurityContext | None:
    """Resolve an API key through the trusted server-side tenant mapping.

    Args:
        api_key: Authenticated API key supplied by the HTTP security layer.

    Returns:
        Mandatory tenant constraints, or ``None`` when tenant scoping is disabled.

    Raises:
        ValueError: If tenant scoping is enabled but the key mapping is invalid.
    """
    tenant_field = os.environ.get("HYBRIDRAG_TENANT_FIELD")
    if not tenant_field:
        return None
    if not api_key:
        raise ValueError("tenant-scoped retrieval requires an authenticated API key")

    raw_mapping = os.environ.get("HYBRIDRAG_API_KEY_TENANTS", "{}")
    try:
        mapping = json.loads(raw_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError("HYBRIDRAG_API_KEY_TENANTS must be valid JSON") from exc
    if not isinstance(mapping, dict):
        raise ValueError("HYBRIDRAG_API_KEY_TENANTS must be a JSON object")

    tenant_id = mapping.get(api_key)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("authenticated API key has no configured tenant mapping")

    return RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field=tenant_field,
                    operator="eq",
                    value=tenant_id,
                )
            ]
        )
    )


def principal_security_context(
    principal: dict[str, object],
) -> RetrievalSecurityContext | None:
    """Resolve a validated JWT principal into a mandatory tenant predicate.

    Args:
        principal: Identity produced by the validated JWT authentication layer.

    Returns:
        Mandatory tenant constraints, or ``None`` when tenant scoping is disabled.

    Raises:
        ValueError: If the configured tenant claim is absent or empty.
    """
    tenant_field = os.environ.get("HYBRIDRAG_TENANT_FIELD")
    if not tenant_field:
        return None

    tenant_claim = os.environ.get("HYBRIDRAG_TENANT_CLAIM", "username")
    if tenant_claim == "username":
        tenant_id = principal.get("username")
    else:
        metadata = principal.get("metadata")
        tenant_id = metadata.get(tenant_claim) if isinstance(metadata, dict) else None
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError(
            f"authenticated principal has no non-empty '{tenant_claim}' tenant claim"
        )

    return RetrievalSecurityContext(
        mandatory_filter=FilterConfig(
            predicates=[
                FilterPredicate(
                    field=tenant_field,
                    operator="eq",
                    value=tenant_id,
                )
            ]
        )
    )

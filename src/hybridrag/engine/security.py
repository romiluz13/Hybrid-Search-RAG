from __future__ import annotations

import json
import os
from contextvars import ContextVar, Token

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


def api_key_security_context(api_key: str | None) -> RetrievalSecurityContext | None:
    """Resolve an API key through the trusted server-side tenant mapping."""
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
    """Resolve a validated JWT principal into a mandatory tenant predicate."""
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

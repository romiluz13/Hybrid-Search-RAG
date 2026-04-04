"""Tests for deterministic MongoDB workspace resolution."""

from hybridrag.engine.kg.mongo_impl import resolve_workspace


def test_explicit_workspace_wins_over_environment(monkeypatch):
    """Explicit runtime workspace must override ambient shell state."""
    monkeypatch.setenv("MONGODB_WORKSPACE", "ambient-workspace")

    assert resolve_workspace("publish-gate") == "publish-gate"
    assert resolve_workspace("") == ""


def test_environment_workspace_is_used_only_as_fallback(monkeypatch):
    """Env workspace remains available when no explicit value is provided."""
    monkeypatch.setenv("MONGODB_WORKSPACE", "ambient-workspace")

    assert resolve_workspace(None) == "ambient-workspace"

"""Tests for Phase 1 core source fixes (C1 verify, C2 env leak, M1 async, M2 TTL)."""

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock

import pytest


class TestC1WontfixVerify:
    """C1: _create_llm_func returns a callable stub when enable_llm=False."""

    def test_llm_disabled_stub_is_callable(self):
        """When enable_llm=False, _create_llm_func must return a callable (not None)."""
        from hybridrag.core.rag import _create_llm_func

        settings = MagicMock()
        settings.enable_llm = False

        result = _create_llm_func(settings)
        assert callable(result), (
            "C1: _create_llm_func must return a callable stub when enable_llm=False "
            "because the engine calls partial(self.llm_model_func) at init time"
        )

    @pytest.mark.asyncio
    async def test_llm_disabled_stub_raises_on_call(self):
        """The callable stub should raise RuntimeError when actually invoked."""
        from hybridrag.core.rag import _create_llm_func

        settings = MagicMock()
        settings.enable_llm = False

        stub = _create_llm_func(settings)
        with pytest.raises(RuntimeError, match="LLM generation is disabled"):
            await stub("test prompt")


class TestC2EnvLeak:
    """C2: MONGO_URI should only be set in os.environ if not already present."""

    def test_env_var_set_is_conditional(self):
        """Source code must guard os.environ['MONGO_URI'] with 'not in os.environ'."""
        import hybridrag.core.rag as rag_module

        source = inspect.getsource(rag_module)
        # The fix should contain a conditional check before setting MONGO_URI
        assert (
            '"MONGO_URI" not in os.environ' in source
            or "'MONGO_URI' not in os.environ" in source
        ), (
            "C2: os.environ['MONGO_URI'] must be guarded with 'not in os.environ' "
            "to avoid leaking secrets when the env var is already set"
        )

    def test_env_var_mongo_database_is_conditional(self):
        """Source code must guard os.environ['MONGO_DATABASE'] with 'not in os.environ'."""
        import hybridrag.core.rag as rag_module

        source = inspect.getsource(rag_module)
        assert (
            '"MONGO_DATABASE" not in os.environ' in source
            or "'MONGO_DATABASE' not in os.environ" in source
        ), "C2: os.environ['MONGO_DATABASE'] must be guarded with 'not in os.environ'"


class TestM1RemoveAsync:
    """M1: index_create should not wrap sync code in async/asyncio.run."""

    def test_index_create_body_has_no_async_wrapper(self):
        """The index_create command must not use async def _create() wrapper."""
        import hybridrag.cli.app as app_module

        source = inspect.getsource(app_module.index_create)
        # Parse the AST to check for async function defs inside index_create
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_create":
                pytest.fail(
                    "M1: index_create still contains 'async def _create()' -- "
                    "remove it since there are zero await calls"
                )

    def test_index_create_no_asyncio_run(self):
        """The index_create command must not call asyncio.run()."""
        import hybridrag.cli.app as app_module

        source = inspect.getsource(app_module.index_create)
        assert "asyncio.run" not in source, (
            "M1: index_create still calls asyncio.run() -- "
            "remove it since the inner function has zero await calls"
        )


class TestM2TTLCast:
    """M2: TTL comparison must use int() cast on expireAfterSeconds."""

    def test_ttl_comparison_uses_int_cast(self):
        """conversation.py must cast expireAfterSeconds to int before comparing."""
        import hybridrag.memory.conversation as conv_module

        source = inspect.getsource(conv_module)
        # Look for int(... .get("expireAfterSeconds" ...)) pattern
        assert 'int(updated_at_index.get("expireAfterSeconds"' in source, (
            "M2: conversation.py must use int() cast on expireAfterSeconds "
            "to handle MongoDB returning float values"
        )

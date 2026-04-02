"""
OpenAI integration for LLM generation and embeddings.

Supports:
- GPT-4o, GPT-4, GPT-3.5 for generation
- text-embedding-3-large, text-embedding-3-small for embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class OpenAILLM:
    """
    OpenAI LLM wrapper.

    Provides async generation for HybridRAG.
    Supports custom base_url and headers for Azure/gateway endpoints.
    """

    api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 4096
    base_url: str | None = None
    default_headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers,
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate response from OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=messages,
        )

        return response.choices[0].message.content or ""


@dataclass
class OpenAIEmbedder:
    """
    OpenAI embedding wrapper.

    Supports text-embedding-3-large (3072 dims) and text-embedding-3-small (1536 dims).
    """

    api_key: str
    model: str = "text-embedding-3-large"
    batch_size: int = 100

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(api_key=self.api_key)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension based on model."""
        dims = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(self.model, 3072)

    async def embed_async(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Embed texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([], dtype=np.float32)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = await self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)


def create_openai_llm_func(
    api_key: str,
    model: str = "gpt-4o",
    max_tokens: int = 4096,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> Callable[..., str]:
    """
    Create LLM function using OpenAI.

    Args:
        api_key: OpenAI API key
        model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
        max_tokens: Maximum tokens for generation
        base_url: Custom API endpoint (for Azure/gateway proxies)
        default_headers: Extra headers (e.g. {"api-key": "..."} for Azure)

    Returns:
        Async LLM function
    """
    llm = OpenAILLM(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        base_url=base_url,
        default_headers=default_headers,
    )

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs,
    ) -> str:
        return await llm.generate_async(prompt, system_prompt, **kwargs)

    return llm_func


def create_openai_embedding_func(
    api_key: str,
    model: str = "text-embedding-3-large",
    batch_size: int = 100,
) -> Callable[[list[str]], np.ndarray]:
    """
    Create embedding function using OpenAI.

    Args:
        api_key: OpenAI API key
        model: Embedding model name
        batch_size: Batch size for API calls

    Returns:
        Async embedding function
    """
    embedder = OpenAIEmbedder(api_key=api_key, model=model, batch_size=batch_size)

    async def embed_func(texts: list[str]) -> np.ndarray:
        return await embedder.embed_async(texts)

    return embed_func

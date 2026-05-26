"""Async Azure OpenAI wrapper with deterministic local fallback."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.v1.core.tools.ai_search.interfaces import AzureOpenAIConfig
from app.v1.core.config import get_settings
from app.v1.utils.retry import http_retry_async

settings = get_settings()
logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the optional package exists.
    from openai import AsyncAzureOpenAI
except Exception:  # pragma: no cover - import failure is the expected test path.
    AsyncAzureOpenAI = None  # type: ignore[assignment,misc]

try:  # pragma: no cover - optional cloud dependency.
    from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
except Exception:  # pragma: no cover
    DefaultAzureCredential = None  # type: ignore[assignment,misc]
    get_bearer_token_provider = None  # type: ignore[assignment]



class AzureOpenAIClient:
    """Small async facade around Azure OpenAI.

    The wrapper intentionally keeps import and configuration failures non-fatal so
    local tests can run without Azure packages or credentials. Embedding fallback
    is deterministic, which makes assertions stable without network access.
    """

    def __init__(self, config: AzureOpenAIConfig | None = None, client: Any | None = None) -> None:
        self.config = config or AzureOpenAIConfig()
        self._client = client
        self._owns_client = client is None
        self._credential: Any | None = None

    @property
    def supports_remote_embeddings(self) -> bool:
        return bool(
            self.config.endpoint
            and self.config.embedding_deployment
            and AsyncAzureOpenAI is not None
            and self._has_azure_credential()
        )

    @property
    def supports_remote_chat(self) -> bool:
        return bool(
            self.config.endpoint
            and self.config.chat_deployment
            and AsyncAzureOpenAI is not None
            and self._has_azure_credential()
        )

    async def create_embedding(self, text: str) -> list[float]:
        embeddings = await self.create_embeddings([text])
        return embeddings[0]

    async def create_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.supports_remote_embeddings:
            try:
                client = self._get_client()

                @http_retry_async()
                async def _do_create_embeddings() -> Any:
                    return await client.embeddings.create(
                        input=list(texts),
                        model=self.config.embedding_deployment,
                    )

                response = await _do_create_embeddings()
                return [list(item.embedding) for item in response.data]
            except Exception:
                if not self.config.fallback_enabled:
                    raise
                logger.warning(
                    "azure_openai.embeddings_failed: falling back to deterministic embedding",
                    exc_info=True,
                )

        if not self.config.fallback_enabled:
            raise RuntimeError("Azure OpenAI embeddings are not configured")

        return [
            _deterministic_embedding(text, dimensions=self.config.fallback_dimensions)
            for text in texts
        ]

    async def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        """Return a chat completion, or a compact extractive fallback."""

        content, _ = await self.complete_with_meta(messages, **kwargs)
        return content

    async def complete_with_meta(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> tuple[str, str | None]:
        """Return ``(content, finish_reason)`` for a chat completion.

        ``finish_reason`` is ``"length"`` when the model hit ``max_tokens`` — callers
        that care about truncation should surface this. For the fallback path
        (Azure not configured / call failed and fallback enabled) finish_reason is
        ``None`` since no model decided when to stop.
        """

        if self.supports_remote_chat:
            try:
                client = self._get_client()

                @http_retry_async()
                async def _do_chat_complete() -> Any:
                    return await client.chat.completions.create(
                        messages=list(messages),
                        model=self.config.chat_deployment,
                        **kwargs,
                    )

                response = await _do_chat_complete()
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                return content, finish_reason
            except Exception:
                if not self.config.fallback_enabled:
                    raise
                logger.warning(
                    "azure_openai.chat_failed: falling back to extractive answer",
                    exc_info=True,
                )

        if not self.config.fallback_enabled:
            raise RuntimeError("Azure OpenAI chat is not configured")

        user_messages = [
            message.get("content", "") for message in messages if message.get("role") == "user"
        ]
        return "\n".join(part for part in user_messages if part).strip(), None

    async def close(self) -> None:
        client = self._client
        if client is not None and self._owns_client:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        credential = self._credential
        close = getattr(credential, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def __aenter__(self) -> AzureOpenAIClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _get_client(self) -> Any:
        if self._client is None:
            if AsyncAzureOpenAI is None:
                raise RuntimeError("openai package is not installed")
            kwargs: dict[str, Any] = {
                "azure_endpoint": self.config.endpoint,
                "api_version": self.config.api_version,
            }
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            elif self.config.use_managed_identity:
                if DefaultAzureCredential is None or get_bearer_token_provider is None:
                    raise RuntimeError("Azure OpenAI Managed Identity requires azure-identity")
                self._credential = DefaultAzureCredential()
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    self._credential,
                    settings.azure_openai_scope,
                )
            else:
                raise RuntimeError("Azure OpenAI requires an API key or Managed Identity")
            self._client = AsyncAzureOpenAI(**kwargs)
        return self._client

    def _has_azure_credential(self) -> bool:
        if self.config.api_key:
            return True
        return bool(
            self.config.use_managed_identity
            and DefaultAzureCredential is not None
            and get_bearer_token_provider is not None
        )


_default_client_lock = threading.Lock()
_default_client: AzureOpenAIClient | None = None


def get_default_client() -> AzureOpenAIClient:
    """Return a process-wide shared AzureOpenAIClient.

    Reusing one client preserves the underlying ``httpx.AsyncClient``
    connection pool and the Azure managed-identity token cache across calls.
    Construction is thread-safe; the returned client is bound to whichever
    event loop first uses it, so callers must run on a long-lived loop
    (e.g. the FastAPI lifespan loop).
    """

    global _default_client
    cached = _default_client
    if cached is not None:
        return cached
    with _default_client_lock:
        if _default_client is None:
            _default_client = AzureOpenAIClient()
        return _default_client


async def close_default_client() -> None:
    """Close the process-wide AzureOpenAIClient if one was created."""

    global _default_client
    client = _default_client
    if client is None:
        return
    _default_client = None
    try:
        await client.close()
    except Exception:
        # Best-effort shutdown; do not surface during process teardown.
        logger.warning("azure_openai.close_failed", exc_info=True)


def reset_default_client_for_tests() -> None:
    """Drop the cached client without closing it (test-only helper)."""

    global _default_client
    _default_client = None


def _deterministic_embedding(text: str, *, dimensions: int) -> list[float]:
    dimensions = max(1, dimensions)
    vector = [0.0] * dimensions
    tokens = list(_tokens(text)) or [""]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _tokens(text: str) -> Iterable[str]:
    token: list[str] = []
    for character in text.lower():
        if character.isalnum():
            token.append(character)
        elif token:
            yield "".join(token)
            token = []
    if token:
        yield "".join(token)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _prod_like_env() -> bool:
    return os.getenv("APP_ENV", "local").strip().lower() in {"stage", "prod", "production"}


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

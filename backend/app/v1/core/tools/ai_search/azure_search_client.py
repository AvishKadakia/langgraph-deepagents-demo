"""Async Azure AI Search wrapper with static local fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


from app.v1.core.config import get_settings
from .interfaces import SearchResult, AzureSearchConfig
from app.v1.utils.retry import http_retry_async

settings = get_settings()
logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional cloud dependency.
    from azure.core.credentials import AzureKeyCredential
except Exception:  # pragma: no cover
    AzureKeyCredential = None  # type: ignore[assignment,misc]

try:  # pragma: no cover - optional cloud dependency.
    from azure.identity.aio import DefaultAzureCredential
except Exception:  # pragma: no cover
    DefaultAzureCredential = None  # type: ignore[assignment,misc]

try:  # pragma: no cover - optional cloud dependency.
    from azure.search.documents.aio import SearchClient
    from azure.search.documents.models import VectorizedQuery
except Exception:  # pragma: no cover
    SearchClient = None  # type: ignore[assignment,misc]
    VectorizedQuery = None  # type: ignore[assignment,misc]


_INDEX_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


_LOCAL_STATIC_DOCUMENTS: dict[str, tuple[dict[str, Any], ...]] = {
    "tenant-a-index": (
        {
            "id": "tenant-a-portal-access",
            "tenant_id": "tenant-a",
            "title": "Tenant A portal access guide",
            "content": (
                "Tenant A portal access depends on successful authentication and approved "
                "group membership."
            ),
        },
        {
            "id": "tenant-a-ticket-triage",
            "tenant_id": "tenant-a",
            "title": "Tenant A ticket triage guide",
            "content": (
                "Tenant A tickets should be reviewed before searching related knowledge articles."
            ),
        },
    ),
    "tenant-b-index": (
        {
            "id": "tenant-b-portal-access",
            "tenant_id": "tenant-b",
            "title": "Tenant B portal access guide",
            "content": (
                "Tenant B portal access requires tenant-specific entitlements before the portal "
                "shows protected content."
            ),
        },
        {
            "id": "tenant-b-escalation",
            "tenant_id": "tenant-b",
            "title": "Tenant B escalation path",
            "content": "Tenant B escalations route to the application support queue first.",
        },
    ),
    "tenant-c-index": (
        {
            "id": "tenant-c-portal-access",
            "tenant_id": "tenant-c",
            "title": "Tenant C portal access guide",
            "content": (
                "Tenant C access issues are resolved by checking identity claims and the "
                "tenant-specific access package."
            ),
        },
        {
            "id": "tenant-c-knowledge-review",
            "tenant_id": "tenant-c",
            "title": "Tenant C knowledge review",
            "content": "Tenant C knowledge articles are reviewed monthly by the service team.",
        },
    ),
    "nfcu-rag-index-1": (
        {
            "id": "nfcu-rag-1-access",
            "tenant_id": "nfcu-rag-1",
            "title": "NFCU RAG index 1 access guide",
            "content": (
                "Index 1 placeholder content for validating local in-process knowledge search."
            ),
        },
    ),
    "nfcu-rag-index-2": (
        {
            "id": "nfcu-rag-2-access",
            "tenant_id": "nfcu-rag-2",
            "title": "NFCU RAG index 2 access guide",
            "content": (
                "Index 2 placeholder content for validating local in-process knowledge search."
            ),
        },
    ),
    "nfcu-rag-index-3": (
        {
            "id": "nfcu-rag-3-access",
            "tenant_id": "nfcu-rag-3",
            "title": "NFCU RAG index 3 access guide",
            "content": (
                "Index 3 placeholder content for validating local in-process knowledge search."
            ),
        },
    ),
}


class SearchConfigurationError(RuntimeError):
    """Raised when a live Azure Search call is requested without configuration."""


class SearchExecutionError(RuntimeError):
    """Raised when Azure Search fails and fallback is disabled."""

class AzureSearchClient:
    """Async client for keyword, semantic, vector, and hybrid retrieval.

    The index is deliberately a method argument. Callers must pass an authorized
    index per request; this class never infers one from natural-language prompts.
    """

    def __init__(
        self,
        config: AzureSearchConfig | None = None,
        *,
        static_documents: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        clients: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or AzureSearchConfig()
        documents = static_documents
        if documents is None and not self.config.endpoint:
            documents = _LOCAL_STATIC_DOCUMENTS
        self.static_documents = {
            index_name: [dict(document) for document in documents]
            for index_name, documents in (documents or {}).items()
        }
        self._clients: dict[str, Any] = dict(clients or {})
        self._credential: Any | None = None

    @property
    def supports_remote_search(self) -> bool:
        return bool(
            self.config.endpoint
            and SearchClient is not None
            and (
                self.config.api_key
                or (self.config.use_managed_identity and DefaultAzureCredential is not None)
            )
        )

    async def search(
        self,
        query: str,
        *,
        index_name: str,
        embedding: Sequence[float] | None = None,
        top_k: int | None = None,
        filter_expression: str | None = None,
        use_semantic: bool = True,
        use_vector: bool = True,
        use_hybrid: bool = True,
        select_fields: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        validated_index_name = validate_index_name(index_name)
        limit = max(1, top_k or self.config.default_top_k)

        if self.supports_remote_search:
            try:
                return await self._remote_search(
                    query,
                    index_name=validated_index_name,
                    embedding=embedding,
                    top_k=limit,
                    filter_expression=filter_expression,
                    use_semantic=use_semantic,
                    use_vector=use_vector,
                    use_hybrid=use_hybrid,
                    select_fields=select_fields,
                )
            except Exception as exc:
                if not self.config.fallback_enabled:
                    raise SearchExecutionError(str(exc)) from exc

        if not self.config.fallback_enabled:
            raise SearchConfigurationError("Azure AI Search is not configured")

        return self._static_search(query, index_name=validated_index_name, top_k=limit)

    async def close(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        self._clients.clear()

        credential = self._credential
        close = getattr(credential, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def __aenter__(self) -> AzureSearchClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _remote_search(
        self,
        query: str,
        *,
        index_name: str,
        embedding: Sequence[float] | None,
        top_k: int,
        filter_expression: str | None,
        use_semantic: bool,
        use_vector: bool,
        use_hybrid: bool,
        select_fields: Sequence[str] | None,
    ) -> list[SearchResult]:
        client = self._get_client(index_name)
        fields = list(select_fields or self.config.select_fields)
        search_text = query if (use_hybrid or not embedding) else "*"

        kwargs: dict[str, Any] = {
            "search_text": search_text,
            "top": top_k,
            "select": fields,
            "include_total_count": False,
        }
        if filter_expression:
            kwargs["filter"] = filter_expression

        vector_query = self._build_vector_query(embedding, top_k=top_k) if use_vector else None
        if vector_query is not None:
            kwargs["vector_queries"] = [vector_query]

        if use_semantic and self.config.semantic_configuration_name:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = self.config.semantic_configuration_name

        @http_retry_async()
        async def _do_search(**call_kwargs: Any) -> Any:
            return await client.search(**call_kwargs)

        try:
            response = await _do_search(**kwargs)
        except TypeError:
            logger.warning(
                "azure_search.vector_queries_unsupported: retrying without vector_queries",
                exc_info=True,
            )
            kwargs.pop("vector_queries", None)
            response = await _do_search(**kwargs)
        except Exception:
            if "query_type" in kwargs:
                logger.warning(
                    "azure_search.semantic_search_failed: retrying without semantic configuration",
                    exc_info=True,
                )
                kwargs.pop("query_type", None)
                kwargs.pop("semantic_configuration_name", None)
                response = await _do_search(**kwargs)
            else:
                raise

        results: list[SearchResult] = []
        async for item in response:
            results.append(_document_to_result(dict(item), self.config))
        return results

    def _get_client(self, index_name: str) -> Any:
        if index_name in self._clients:
            return self._clients[index_name]
        if SearchClient is None:
            raise SearchConfigurationError("azure-search-documents is not installed")
        if not self.config.endpoint:
            raise SearchConfigurationError("AZURE_SEARCH_ENDPOINT is not configured")

        credential = self._get_credential()
        self._clients[index_name] = SearchClient(
            endpoint=self.config.endpoint,
            index_name=index_name,
            credential=credential,
        )
        return self._clients[index_name]

    def _get_credential(self) -> Any:
        if self._credential is not None:
            return self._credential
        if self.config.api_key:
            if AzureKeyCredential is None:
                raise SearchConfigurationError("azure-core is not installed")
            self._credential = AzureKeyCredential(self.config.api_key)
            return self._credential
        if not self.config.use_managed_identity:
            raise SearchConfigurationError("Azure Search requires an API key or Managed Identity")
        if DefaultAzureCredential is None:
            raise SearchConfigurationError("Azure Search requires an API key or azure-identity")
        self._credential = DefaultAzureCredential()
        return self._credential

    def _build_vector_query(self, embedding: Sequence[float] | None, *, top_k: int) -> Any | None:
        if embedding is None or VectorizedQuery is None:
            return None
        return VectorizedQuery(
            vector=list(embedding),
            k_nearest_neighbors=top_k,
            fields=self.config.vector_field_name,
        )

    def _static_search(self, query: str, *, index_name: str, top_k: int) -> list[SearchResult]:
        documents = self.static_documents.get(index_name, ())
        scored: list[tuple[float, Mapping[str, Any]]] = []
        query_terms = _terms(query)

        for document in documents:
            haystack = " ".join(
                str(document.get(field, ""))
                for field in (
                    self.config.content_field_name,
                    self.config.title_field_name,
                    "content",
                    "chunk",
                    "text",
                    "title",
                    "document_title",
                )
            )
            haystack_terms = _terms(haystack)
            overlap = len(query_terms & haystack_terms)
            score = float(overlap) / max(1, len(query_terms))
            if not query_terms or overlap > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _document_to_result(dict(document, **{"@search.score": score}), self.config)
            for score, document in scored[:top_k]
        ]


def validate_index_name(index_name: str) -> str:
    candidate = (index_name or "").strip()
    if not candidate or not _INDEX_NAME_PATTERN.match(candidate):
        raise ValueError("index_name must be a structured, non-empty search index identifier")
    return candidate


def _document_to_result(document: dict[str, Any], config: AzureSearchConfig) -> SearchResult:
    content = _first_present(document, config.content_field_name, "content", "chunk", "text") or ""
    title = _first_present(
        document,
        config.title_field_name,
        "title",
        "document_title",
        "file_name",
    )
    document_id = _first_present(document, config.id_field_name, "id", "chunk_id", "key")
    source = _first_present(document, "source_url", "url", "source", "file_name")
    metadata = {
        key: value
        for key, value in document.items()
        if key not in {config.vector_field_name, "content_vector", "vector"}
    }
    return SearchResult(
        content=str(content),
        score=_coerce_float(
            document.get("@search.reranker_score") or document.get("@search.score")
        ),
        title=str(title) if title is not None else None,
        document_id=str(document_id) if document_id is not None else None,
        source=str(source) if source is not None else None,
        metadata=metadata,
    )


def _first_present(document: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = document.get(key)
        if value not in (None, ""):
            return value
    return None


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _prod_like_env() -> bool:
    return os.getenv("APP_ENV", "local").strip().lower() in {"stage", "prod", "production"}

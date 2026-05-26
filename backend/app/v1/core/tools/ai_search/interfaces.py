from __future__ import annotations

from typing import TYPE_CHECKING, Any
import os
from dataclasses import dataclass, field, asdict
from app.v1.core.config import get_settings
if TYPE_CHECKING:
    from app.v1.core.tools.ai_search.azure_openai_client import AzureOpenAIClient
    from app.v1.core.tools.ai_search.azure_search_client import AzureSearchClient

import httpx
settings = get_settings()

@dataclass(frozen=True)
class AISearchRequest:
    query: str
    index_name: str
    authorized_index_names: frozenset[str] = settings.authorized_index_names
    top_k: int | None = settings.ai_search_default_top_k
    filter_expression: str | None = settings.filter_expression
    context: str | None = settings.context
    access_token: str | None = settings.access_token
    agent_base_url: str | None = None
    agent_timeout_seconds: float = 10.0
    use_semantic: bool = settings.use_semantic
    use_vector: bool = settings.use_vector
    use_hybrid: bool = settings.use_hybrid
    compose_chars_per_result: int = settings.compose_chars_per_result
    compose_total_char_budget: int = settings.compose_total_char_budget
    _remote_http_client: httpx.AsyncClient | None = None
    openai_client: "AzureOpenAIClient | None" = None
    search_client: "AzureSearchClient | None" = None

@dataclass(frozen=True)
class AISearchResponse:
    answer: str
    citations: list[dict[str, Any]]
    results: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class SearchResult:
    content: str
    score: float | None = None
    title: str | None = None
    document_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI embeddings and chat calls."""

    endpoint: str | None = settings.endpoint
    api_key: str | None = settings.api_key
    api_version: str = settings.api_version
    embedding_deployment: str | None = settings.embedding_deployment
    chat_deployment: str | None = settings.chat_deployment
    use_managed_identity: bool = settings.use_managed_identity
    fallback_enabled: bool = settings.fallback_enabled
    fallback_dimensions: int = settings.fallback_dimensions

class AzureSearchConfig:
    endpoint: str | None = settings.azure_search_endpoint
    api_key: str | None = settings.azure_search_api_key
    semantic_configuration_name: str | None = settings.azure_search_semantic_configuration
    vector_field_name: str = settings.azure_search_vector_field
    content_field_name: str = settings.azure_search_content_field
    title_field_name: str = settings.azure_search_title_field
    id_field_name: str = settings.azure_search_id_field
    default_top_k: int = settings.azure_search_default_top_k
    request_timeout_seconds: float = settings.azure_search_request_timeout_seconds
    use_managed_identity: bool = settings.azure_search_use_managed_identity
    fallback_enabled: bool = settings.azure_search_fallback_enabled
    select_fields: tuple[str, ...] = settings.azure_search_select_fields
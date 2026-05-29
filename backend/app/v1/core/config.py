
from __future__ import annotations
from functools import lru_cache
import json
import os
from typing import Annotated
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.v1.utils.helper import _split_csv
from pydantic import (
    AnyHttpUrl,
    BeforeValidator
)
StringList = Annotated[list[str], BeforeValidator(_split_csv)]

class Settings(BaseSettings):
    """Runtime configuration for the demo backend.

    CORS_ORIGINS is intentionally stored as a plain string because pydantic-settings
    expects complex env values like list[str] to be JSON. A comma-separated value is
    friendlier for Docker Compose, so we parse it ourselves through cors_origin_list.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_bearer_token: str = Field(
        default="dev-token-change-me",
        alias="API_BEARER_TOKEN",
        description="A token used to authenticate API requests. In production, use a secure, randomly generated token and keep it secret.",
    )
    app_name: str = "DeepAgent CopilotKit AG-UI Demo"
    agent_name: str = Field(default="deepagent-demo", alias="AGENT_NAME")
    agent_description: str = Field(
        default="A LangGraph DeepAgent demo exposed through CopilotKit AG-UI.",
        alias="AGENT_DESCRIPTION",
        description="A brief description of the agent's purpose and capabilities.",
    )
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/deepagent?sslmode=disable",
        alias="DATABASE_URL",
    )
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )
    agent_max_steps: int = Field(default=15, alias="AGENT_MAX_STEPS")
    #Entra auth config
    entra_tenant_id: str | None = Field(default=None, alias="ENTRA_TENANT_ID")
    entra_client_id: str | None = Field(default=None, alias="ENTRA_CLIENT_ID")
    entra_audience: str | None = Field(default=None, alias="ENTRA_AUDIENCE")
    entra_issuer: AnyHttpUrl | None = Field(default=None, alias="ENTRA_ISSUER")
    entra_jwks_url: AnyHttpUrl | None = Field(default=None, alias="ENTRA_JWKS_URL")
    entra_required_scopes: StringList = Field(default_factory=list, alias="ENTRA_REQUIRED_SCOPES")
    entra_group_claim: str = Field(default="groups", alias="ENTRA_GROUP_CLAIM")

    tenant_group_index_mapping: dict[str, str] = Field(
        default_factory=dict,
        alias="TENANT_GROUP_INDEX_MAPPING",
        description=(
            "JSON object mapping Entra group IDs or names to Azure AI Search index names, "
            'for example: {"group-id": "search-index"}'
        ),
    )
    #Search Configs
    ai_search_default_top_k: int = 7
    #Azure Search Config
    azure_search_endpoint: str | None = Field(
        default=None,
        alias="AZURE_SEARCH_ENDPOINT",
        agent_description="The endpoint URL for the Azure Search service, e.g., https://my-search.search.windows.net",
    )
    azure_search_api_key: str | None = Field(
        default=None,
        alias="AZURE_SEARCH_API_KEY",
        agent_description="The API key for the Azure Search service."
    )
    azure_ai_search_default_index: str = Field(
        default="documents",
        alias="AZURE_AI_SEARCH_DEFAULT_INDEX",
        agent_description=(
            "The default Azure Search index name to query if no index is specified. "
            "This should match the name of the index you created and populated with your documents. "
            "You can override this on a per-query basis if you have multiple indexes."
        )
    )
    azure_search_select_fields: tuple[str, ...] = Field(
        default=(
        "id",
        "chunk_id",
        "title",
        "document_title",
        "source",
        "source_url",
        "url",
        "file_name",
        "content",
        "chunk",
        "text",
        "x_trace_id",
        ),
        alias="AZURE_SEARCH_SELECT_FIELDS",
        agent_description=(
            "A comma-separated list of fields to select in Azure Search queries. "
            "Defaults to 'id,title,content' which should be sufficient for most use cases. "
            "You can add additional fields if your index contains useful metadata you want included in search results."
        ),
    )

    #Azure Openai Config
    endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
        "AZURE_OPENAI_ENDPOINT",
        "API_ENDPOINT",
        ),
        agent_description="The base URL for the Azure OpenAI resource, e.g., https://my-resource.openai.azure.com/",
    )
    api_key: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_API_KEY",
        agent_description="The API key for authenticating with Azure OpenAI. Required if API_ENDPOINT is set.",
    )
    api_version: str = Field(
        default="2026-05-05",
        alias="AZURE_OPENAI_API_VERSION",
        agent_description="The API version to use for Azure OpenAI requests.",
    )
    embedding_deployment: str | None = Field(
        default="text-embedding-3-large",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
        ),
        agent_description="The deployment name for the embedding model.",
    )
    chat_deployment: str | None = Field(
        default="gpt-chat-latest",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
        ),
        agent_description="The deployment name for the chat model.",
    )
    
    use_managed_identity: bool = Field(
        default=True,
        alias="AZURE_OPENAI_USE_MANAGED_IDENTITY",
        agent_description="Whether to use managed identity for Azure OpenAI authentication.",
    )
    fallback_enabled: bool = Field(
        default=True,
        alias="AZURE_OPENAI_FALLBACK_ENABLED",
        agent_description="Whether to enable fallback for Azure OpenAI requests.",
    )
    fallback_dimensions: int = Field(
        default=1536,
        alias="AZURE_OPENAI_FALLBACK_DIMENSIONS",
        agent_description="The dimensions for fallback embeddings.",
    )
    azure_openai_scope: str = Field(
        default="https://cognitiveservices.azure.com/.default",
        alias="AZURE_OPENAI_SCOPE",
        agent_description="The scope to use for Azure OpenAI authentication. Typically, this should not need to be changed unless you have a custom Azure setup.",
    )
    azure_openai_embedding_version: str = Field(
        default="1",
        alias="AZURE_OPENAI_EMBEDDING_API_VERSION",
         agent_description="The API version to use for Azure OpenAI embedding requests.",
    )

    
    


    @property
    def cors_origin_list(self) -> list[str]:
        """Return CORS origins from either JSON-list or comma-separated env syntax."""
        value = self.cors_origins.strip()
        if not value:
            return []

        # Also support JSON arrays for users who prefer Pydantic's native style:
        # CORS_ORIGINS='["http://localhost:5173", "http://127.0.0.1:5173"]'
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError("CORS_ORIGINS JSON value must be a list of strings")
            return [origin.strip() for origin in parsed if origin.strip()]

        return [origin.strip() for origin in value.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

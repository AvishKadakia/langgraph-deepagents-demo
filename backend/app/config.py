from functools import lru_cache
import json

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the demo backend.

    CORS_ORIGINS is intentionally stored as a plain string because pydantic-settings
    expects complex env values like list[str] to be JSON. A comma-separated value is
    friendlier for Docker Compose, so we parse it ourselves through cors_origin_list.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_bearer_token: str = Field(
        default="Bearer token for securing API routes.",
        alias="API_BEARER_TOKEN",
    )
    app_name: str = "DeepAgent CopilotKit AG-UI Demo"
    agent_name: str = Field(default="deepagent-demo", alias="AGENT_NAME")
    agent_description: str = Field(
        default="A LangGraph DeepAgent demo exposed through CopilotKit AG-UI.",
        alias="AGENT_DESCRIPTION",
    )
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/deepagent?sslmode=disable",
        alias="DATABASE_URL",
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="openai:gpt-4o-mini", alias="OPENAI_MODEL")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
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

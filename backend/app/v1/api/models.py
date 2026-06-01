"""Pydantic contracts for the Parent Agent HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class ChatRequest(BaseModel):
    """Public chat request accepted by the Parent Agent.

    Mirrors the body sent by the Agent web UI (``src/providers/Stream.tsx``):
    ``message``/``session_id``/``metadata`` plus the per-turn model controls
    (``model``/``top_k``/``temperature``) and the optional ``system_prompt``
    surfaced only when advanced settings are enabled. ``extra="forbid"`` is
    retained so unexpected top-level fields (e.g. a caller-injected
    ``owner_id``) are still rejected; declaring the UI's fields here is what
    keeps the contract strict *and* unblocked.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=16000,
        description="User chat/query text.",
    )
    session_id: str | None = Field(default=None, max_length=256)
    tenant_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model: str | None = Field(default=None, max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=100)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    system_prompt: str | None = Field(default=None, max_length=16000)


class ChatResponse(BaseModel):
    """Public response returned by the Parent Agent."""

    request_id: str
    session_id: str | None = None
    answer: str
    errors: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class ConversationThreadHistoryResponse(BaseModel):
    """Owner-scoped run history for one resumable conversation thread."""

    thread_id: str
    session_id: str
    internal_thread_id: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    runs: list[dict[str, Any]] = Field(default_factory=list)

class ConversationThreadResponse(BaseModel):
    """Owner-scoped conversation thread returned to the Agent web UI."""

    thread_id: str
    session_id: str
    internal_thread_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    status: str = "idle"
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, Any] | None = None
    run_count: int = 0
    latest_request_id: str | None = None
    title: str | None = None

class PostgresRuntime:
    """
    Holds *open* Postgres-backed resources for the lifetime of the process.
    We keep the async context managers alive so connections/pools are reused.
    """
    store: AsyncPostgresStore
    checkpointer: AsyncPostgresSaver
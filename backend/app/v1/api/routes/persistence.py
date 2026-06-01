"""State, debug, thread-history, and export routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response

from agent_orchestration.api.models import (
    ConversationThreadHistoryResponse,
    ConversationThreadResponse,
    DebugRunResponse,
    StateHealthResponse,
    StateRunResponse,
    ThreadSearchRequest,
)
from agent_orchestration.api.security import BearerCredentials
from agent_orchestration.exports import AUDIT_COLUMNS, CHECKPOINT_COLUMNS
from agent_orchestration.security import sanitize_for_logging


@dataclass(frozen=True)
class PersistenceRouterDeps:
    """Runtime hooks supplied by ``api.py`` to keep route wiring testable."""

    authenticate_parent_request: Callable[[str | None, Request], tuple[Any, str]]
    authorization_header: Callable[[Any], str | None]
    state_store_status: Callable[[], dict[str, Any]]
    load_owner_run_or_404: Callable[[str, Any], dict[str, Any]]
    debug_run_projection: Callable[[Mapping[str, Any]], dict[str, Any]]
    fetch_conversation_threads: Callable[..., list[dict[str, Any]]]
    fetch_conversation_thread_runs: Callable[..., list[dict[str, Any]]]
    conversation_thread_from_latest_run: Callable[..., dict[str, Any]]
    conversation_thread_from_runs: Callable[..., dict[str, Any]]
    conversation_run_projection: Callable[[Mapping[str, Any]], dict[str, Any]]
    fetch_audit_events: Callable[..., list[dict[str, Any]]]
    fetch_checkpoints: Callable[..., list[dict[str, Any]]]
    format_export_response: Callable[..., Response]
    as_utc: Callable[[datetime | None], datetime | None]
    normalize_session_id: Callable[[str | None], str]
    custom_persistence_enabled: Callable[[], bool]


_DISABLED_DETAIL = "Custom persistence is disabled for this deployment."


def create_persistence_router(deps: PersistenceRouterDeps) -> APIRouter:
    """Build routes that read persisted state without owning DB details."""

    router = APIRouter()

    @router.get("/state/health", response_model=StateHealthResponse)
    def state_health(
        request: Request,
        credentials: BearerCredentials,
    ) -> StateHealthResponse:
        deps.authenticate_parent_request(deps.authorization_header(credentials), request)
        if not deps.custom_persistence_enabled():
            return StateHealthResponse(
                enabled=False,
                status="disabled",
                table="parent_agent_runs",
                detail=_DISABLED_DETAIL,
            )
        try:
            status_payload = deps.state_store_status()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Postgres state store is not healthy: {exc}",
            ) from exc
        return StateHealthResponse.model_validate(status_payload)

    @router.get("/state/runs/{request_id}", response_model=StateRunResponse)
    def state_run(
        request_id: str,
        request: Request,
        credentials: BearerCredentials,
    ) -> StateRunResponse:
        authorization = deps.authorization_header(credentials)
        principal, _ = deps.authenticate_parent_request(authorization, request)
        if not deps.custom_persistence_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)
        return StateRunResponse.model_validate(deps.load_owner_run_or_404(request_id, principal))

    @router.get("/debug/runs/{request_id}", response_model=DebugRunResponse)
    def debug_run(
        request_id: str,
        request: Request,
        credentials: BearerCredentials,
    ) -> DebugRunResponse:
        authorization = deps.authorization_header(credentials)
        principal, _ = deps.authenticate_parent_request(authorization, request)
        if not deps.custom_persistence_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)
        run = deps.load_owner_run_or_404(request_id, principal)
        return DebugRunResponse.model_validate(deps.debug_run_projection(run))

    @router.post("/threads/search", response_model=list[ConversationThreadResponse])
    def search_threads(
        payload: ThreadSearchRequest,
        request: Request,
        credentials: BearerCredentials,
    ) -> list[ConversationThreadResponse]:
        """List conversation threads for the authenticated user."""

        authorization = deps.authorization_header(credentials)
        principal, _ = deps.authenticate_parent_request(authorization, request)
        if not deps.custom_persistence_enabled():
            return []
        try:
            rows = deps.fetch_conversation_threads(
                owner_id=principal.identity,
                limit=payload.limit,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Postgres conversation history is not healthy: {exc}",
            ) from exc
        return [
            ConversationThreadResponse.model_validate(
                deps.conversation_thread_from_latest_run(row, include_messages=False)
            )
            for row in rows
        ]

    @router.get("/threads/{session_id}", response_model=ConversationThreadResponse)
    def get_thread(
        session_id: str,
        request: Request,
        credentials: BearerCredentials,
        limit: int = Query(200, ge=1, le=1000),
    ) -> ConversationThreadResponse:
        """Return one resumable conversation thread for the authenticated user."""

        authorization = deps.authorization_header(credentials)
        principal, _ = deps.authenticate_parent_request(authorization, request)
        normalized_session_id = deps.normalize_session_id(session_id)
        rows = _fetch_thread_runs_or_http_error(deps, principal, normalized_session_id, limit)
        return ConversationThreadResponse.model_validate(
            deps.conversation_thread_from_runs(rows, session_id=normalized_session_id)
        )

    @router.get(
        "/threads/{session_id}/history",
        response_model=ConversationThreadHistoryResponse,
    )
    def get_thread_history(
        session_id: str,
        request: Request,
        credentials: BearerCredentials,
        limit: int = Query(200, ge=1, le=1000),
    ) -> ConversationThreadHistoryResponse:
        """Return persisted run history for one authenticated user's conversation."""

        authorization = deps.authorization_header(credentials)
        principal, _ = deps.authenticate_parent_request(authorization, request)
        normalized_session_id = deps.normalize_session_id(session_id)
        rows = _fetch_thread_runs_or_http_error(deps, principal, normalized_session_id, limit)
        thread = deps.conversation_thread_from_runs(rows, session_id=normalized_session_id)
        return ConversationThreadHistoryResponse.model_validate(
            {
                "thread_id": thread["thread_id"],
                "session_id": thread["session_id"],
                "internal_thread_id": thread.get("internal_thread_id"),
                "values": thread.get("values") or {"messages": []},
                "runs": [deps.conversation_run_projection(row) for row in rows],
            }
        )


def _fetch_thread_runs_or_http_error(
    deps: PersistenceRouterDeps,
    principal: Any,
    session_id: str,
    limit: int,
) -> Sequence[Mapping[str, Any]]:
    if not deps.custom_persistence_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)
    try:
        rows = deps.fetch_conversation_thread_runs(
            owner_id=principal.identity,
            session_id=session_id,
            limit=limit,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Postgres conversation history is not healthy: {exc}",
        ) from exc
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No conversation thread found for that session_id.",
        )
    return rows


__all__ = ["PersistenceRouterDeps", "create_persistence_router"]

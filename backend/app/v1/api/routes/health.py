"""Operational health, readiness, and startup routes."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.v1.api.security import BearerCredentials
from backend.app.v1.utils.langsmith import langsmith_status

logger = logging.getLogger(__name__)

AuthFn = Callable[[str | None, Request], tuple[Any, str]]
AuthorizationHeaderFn = Callable[[Any], str | None]
CheckpointerStatusFn = Callable[[], dict[str, Any]]
RequireCheckpointerFn = Callable[[], bool]

_DEFAULT_STARTER_PROMPTS: list[dict[str, str]] = [
    {
        "label": "End-to-End Data Lineage",
        "message": "Show me the end-to-end data lineage for ACCT_ID from Landing"
        " through RAW, INT, CUR to ASL in the Core Banking STTM",
    },
    {
        "label": "Field Mapping Details",
        "message": "What are the field mappings and transformation logic from RAW"
        " to INT for payment transactions?",
    },
    {
        "label": "Search Knowledge Base",
        "message": "What documentation do we have about the loan origination process?",
    },
    {
        "label": "Browse Available Documents",
        "message": "What STTM workbooks and documentation are available in the"
        " knowledge base?",
    },
]


def _starter_prompts() -> list[dict[str, str]]:
    """Resolve starter prompts from ``AGENT_STARTER_PROMPTS`` JSON or defaults.

    The web UI's landing screen fetches ``GET /starter-prompts`` (unauthenticated)
    and expects ``{"prompts": [{"label", "message"}, ...]}``. Operators can
    override the defaults with a JSON array in ``AGENT_STARTER_PROMPTS``; any
    malformed or empty value falls back to the built-ins so the endpoint never
    fails the UI.
    """

    raw = os.getenv("AGENT_STARTER_PROMPTS", "").strip()
    if not raw:
        return _DEFAULT_STARTER_PROMPTS
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("AGENT_STARTER_PROMPTS is not valid JSON; using defaults.")
        return _DEFAULT_STARTER_PROMPTS
    prompts: list[dict[str, str]] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        message = item.get("message")
        if isinstance(label, str) and isinstance(message, str) and label and message:
            prompts.append({"label": label, "message": message})
    return prompts or _DEFAULT_STARTER_PROMPTS


def create_health_router(
    *,
    app_state: Any,
    authenticate_parent_request: AuthFn,
    authorization_header: AuthorizationHeaderFn,
    public_checkpointer_status: CheckpointerStatusFn,
    require_checkpointer: RequireCheckpointerFn,
) -> APIRouter:
    """Build probe routes around the FastAPI app state."""

    router = APIRouter()

    @router.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "parent-agent",
            "status": "ok",
            "docs": "/docs",
            "health": "/healthz",
            "liveness": "/livez",
            "readiness": "/readyz",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "state_health": "/state/health",
            "state_run": "/state/runs/{request_id}",
            "debug_run": "/debug/runs/{request_id}",
            "threads_search": "/threads/search",
            "thread": "/threads/{session_id}",
            "thread_history": "/threads/{session_id}/history",
            "export_audit": "/export/audit",
            "export_checkpoints": "/export/checkpoints",
            "starter_prompts": "/starter-prompts",
            "message": (
                "Use POST /chat or POST /chat/stream for parent-agent orchestration requests."
            ),
        }

    @router.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @router.get("/starter-prompts")
    def starter_prompts() -> dict[str, Any]:
        """Public landing-screen starter prompts for the web UI."""

        return {"prompts": _starter_prompts()}

    @router.get("/livez")
    def livez() -> dict[str, Any]:
        return {"status": "ok"}

    @router.get("/startupz")
    def startupz(
        request: Request,
        credentials: BearerCredentials,
    ) -> dict[str, Any]:
        """Authenticated startup-detail probe."""

        authenticate_parent_request(authorization_header(credentials), request)
        return {
            "status": "ok",
            "graph": "compiled" if getattr(app_state, "graph", None) is not None else "missing",
            "checkpointer": public_checkpointer_status(),
            "observability": {"langsmith": langsmith_status()},
        }

    @router.get("/readyz")
    def readyz() -> dict[str, Any]:
        readiness: dict[str, Any] = {
            "status": "ok",
            "graph": "compiled" if getattr(app_state, "graph", None) is not None else "missing",
            "checkpointer": public_checkpointer_status(),
            "observability": {"langsmith": langsmith_status()},
        }
        if readiness["graph"] != "compiled":
            readiness["status"] = "error"
        checkpointer = readiness["checkpointer"]
        if require_checkpointer() and checkpointer.get("status") != "ok":
            readiness["status"] = "error"
        if readiness["observability"]["langsmith"].get("status") == "error":
            readiness["status"] = "error"
        if readiness["status"] != "ok":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness,
            )
        return readiness

    return router


__all__ = ["create_health_router"]

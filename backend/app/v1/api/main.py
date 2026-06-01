"""FastAPI application assembly and chat graph execution for the Parent Agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, AsyncExitStack

from datetime import UTC, datetime
from typing import Any, TypedDict, cast
from uuid import uuid4
from backend.app.v1.api.models import PostgresRuntime
from psycopg.rows import dict_row

import anyio.to_thread
import structlog.contextvars as _structlog_ctxv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.v1.api.models import (
    ChatRequest,
    ChatResponse
)

from backend.app.v1.api.routes.health import create_health_router
from backend.app.v1.api.routes.persistence import (
    PersistenceRouterDeps,
    create_persistence_router,
)
from backend.app.v1.api.security import BearerCredentials

from backend.app.v1.api.security import (
    AuthConfigurationError,
    AuthValidationError,
    extract_bearer_token,
)
from backend.app.v1.utils.auth import (
    authenticator as _default_authenticator,
)

from backend.app.v1.utils.langsmith import (
    log_extra,
    public_auth_metadata,
    safe_langsmith_metadata,
)

from backend.app.v1.utils.helper import (
    redact_text_uncapped,
    sanitize_for_logging,
    sanitize_for_streaming,
    hash_identifier,
)
from typing import Optional
from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from backend.app.v1.core.config import get_settings
from backend.app.v1.core.agent import build_agent, close_agent_resources
settings = get_settings()

global_runtime: Optional[PostgresRuntime] = None
_runtime_lock = asyncio.Lock()
_global_stack = AsyncExitStack()
logger = logging.getLogger(__name__)

# This module owns FastAPI app construction, lifespan, request logging, and
# chat graph execution. Route modules own stable non-chat endpoint declarations.


def _default_parent_graph() -> Any:
    """Lazily resolve the checkpointer-less parent graph (single lazy compile)."""

    return build_agent()

async def get_postgres_runtime(progress_url) -> PostgresRuntime:
    """Initializes the shared connection pool, store, checkpointer, MCPs, and plugins."""
    global global_runtime
    
    async with _runtime_lock:
        if global_runtime is None:
            logger.debug("--- Initializing Shared Global Database Pool & MCPs ---")
            
            pool_instance = AsyncConnectionPool(
                conninfo=settings.postgress_url, 
                max_size=10, 
                kwargs={"autocommit": True,"row_factory": dict_row}
            )
            
            # 1. Create a single, shared AsyncConnectionPool bound to the exit stack
            shared_pool = await _global_stack.enter_async_context(
                pool_instance
            )
            await shared_pool.open()

            # =========================================================
            # 2. Set up LangGraph Checkpointer, Store, and Jade Plugins
            # =========================================================
            store = AsyncPostgresStore(shared_pool)
            checkpointer = AsyncPostgresSaver(shared_pool)

            await asyncio.gather(
                store.setup(),
                checkpointer.setup(),
            )

            global_runtime = PostgresRuntime(store=store, checkpointer=checkpointer)
            
    return global_runtime

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if settings.enable_postgres_checkpointer:
        await get_postgres_runtime()


app = FastAPI(
    title="Parent Agent API",
    version="0.1.0",
    description="Minimal chat API that validates caller tokens and invokes the parent LangGraph.",
    lifespan=_lifespan,
)
# app.state.graph is populated by the lifespan via _wire_postgres_checkpointer
# (success path) or via the parent_graph fallback (failure path). Avoid
# eagerly building the graph here so we don't compile twice on every cold
# start when Postgres checkpointing succeeds.
app.state.graph = None
app.state.checkpointer = None
app.state.checkpointer_status = {"enabled": False, "status": "disabled"}
_cors_origins = [origin.strip() for origin in os.getenv("APP_CORS_ORIGINS", "").split(",")]
_cors_origins = [origin for origin in _cors_origins if origin]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["authorization", "content-type", "x-request-id"],
        expose_headers=["x-request-id"],
        max_age=600,
    )
def _authorization_header(credentials: HTTPAuthorizationCredentials | None) -> str | None:
    if credentials is None:
        return None
    return f"{credentials.scheme} {credentials.credentials}"

def _authenticate_parent_request(
    authorization: str | None,
    request: Request,
) -> tuple[Any, str]:
    try:
        access_token = extract_bearer_token(authorization)
        principal = _default_authenticator.authenticate_authorization(
            authorization,
            request.headers,  # type: ignore[arg-type]
        )
    except AuthConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except AuthValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        ) from exc
    return principal, access_token


def _safe_user_message(message: str) -> str:
    safe_message = sanitize_for_logging(message)
    return safe_message if isinstance(safe_message, str) else str(safe_message)


def _safe_response_text(value: Any) -> str:
    # Credential-scrub, but do NOT apply the log/state length cap — chat
    # answers and error messages should be returned to the caller in full.
    return redact_text_uncapped(str(value or ""))

def _state_safe_auth_context(principal: Any) -> dict[str, Any]:
    context = dict(public_auth_metadata(principal))
    context.update(
        {
            "identity": principal.identity,
            "tenant_id": principal.tenant,
            "groups": list(principal.groups),
            "roles": list(principal.permissions),
            "scopes": list(principal.scopes),
            "authenticated": principal.authenticated,
            "metadata": public_auth_metadata(principal),
        }
    )
    return cast(dict[str, Any], sanitize_for_logging(context))

def _normalize_session_id(value: str | None) -> str:
    """Return a stable, bounded session id suitable for thread_id construction."""

    raw = (value or "default").strip() or "default"
    if len(raw) <= 128 and re.fullmatch(r"[A-Za-z0-9._:-]+", raw):
        return raw

    return f"session:{hash_identifier(raw)}"

def _new_turn_resets() -> dict[str, Any]:
    """Clear per-turn graph fields that must not leak across checkpoints."""

    return {
        "input_safe": None,
        "safety_reasons": [],
        "intent": None,
        "route": None,
        "route_label": None,
        "authorization_denied": None,
        "original_route": None,
        "search_query": None,
        "resolution_query": None,
        "servicenow_result": None,
        "ai_search_result": None,
        "answer": None,
        "response": None,
        "final_answer": None,
        "errors": [],
    }
@app.middleware("http")
async def _structured_request_logging(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log request start/end (no headers, bodies, or secrets) and bind
    request_id/method/path into structlog contextvars for correlation."""

    request_id = request.headers.get("x-request-id") or str(uuid4())
    bound_token = _structlog_ctxv.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    started = time.perf_counter()
    try:
        logger.info(
            "http.request.start",
            extra={
                "event": "http.request.start",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.exception(
                "http.request.error",
                extra={
                    "event": "http.request.error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "error": _safe_response_text(exc),
                },
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http.request.end",
            extra={
                "event": "http.request.end",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
    finally:
        _structlog_ctxv.reset_contextvars(**bound_token)

def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    credentials: BearerCredentials,
) -> ChatResponse:
    pass

@app.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    credentials: BearerCredentials,
) -> StreamingResponse:
    """Server-sent events variant of /chat that streams LangGraph node updates."""

    authorization = _authorization_header(credentials)
    principal, access_token = await anyio.to_thread.run_sync(
        _authenticate_parent_request, authorization, request
    )
    request_id = str(uuid4())
    auth_context = _state_safe_auth_context(principal)
    session_id = _normalize_session_id(payload.session_id)
    thread_id = f"{principal.identity}:{session_id}"
    user_message = _safe_user_message(payload.message)
    state = {
        "user_input": user_message,
        "session_id": session_id,
        "user_id": principal.identity,
        "tenant_id": principal.tenant or "default",
        "tenant_context": dict(payload.tenant_context),
        "auth_context": auth_context,
        "metadata": sanitize_for_logging(payload.metadata),
        **_new_turn_resets(),
    }
    config = {
        "metadata": safe_langsmith_metadata(
            principal=principal,
            request_metadata=payload.metadata,
            extra={"request_id": request_id},
        ),
        "configurable": {
            "thread_id": thread_id,
            "auth_context": auth_context,
            "langgraph_auth_user": principal.to_langgraph_user(),
            
        },
        "payload": payload
    }
    

    graph = build_agent(checkpointer=app.state.checkpointer,config=config) if app.state.graph is None else app.state.graph
    # async def _generator() -> AsyncIterator[str]:
    async for chunk, metadata in graph.astream(
        payload, 
        config=config, 
        stream_mode="messages"
    ):
        # 1. Handle Text Content
        if hasattr(chunk, "content") and chunk.content:
            if isinstance(chunk.content, str):
                logger.output(chunk.content)
            elif isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict) and "text" in block:
                        logger.output(block["text"])
                        
        # 2. Handle Tool Calls
        elif hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            for tc_chunk in chunk.tool_call_chunks:
                if tc_chunk.get("name"):
                    logger.output(f"\n🔧 Starting Tool Call: {tc_chunk['name']}...")
                
                if tc_chunk.get("args"):
                    args_text = tc_chunk["args"]
                    # 🛑 HIDE MASSIVE BASE64 ARGUMENTS
                    if "data:image" in args_text and len(args_text) > 200:
                        args_text = '{"image_url": "[... BASE64 IMAGE TRUNCATED FOR LOGS ...]"}'
                    
                    logger.output(args_text)
            
        # 3. Handle Tool Responses
        elif hasattr(chunk, "name") and chunk.type == "tool":
            display_content = chunk.content
            
            # 🛑 HIDE MASSIVE BASE64 RESULTS (Like your Subagent Report)
            if isinstance(display_content, list):
                display_content = [
                    {"type": "image_url", "image_url": "[📸 IMAGE HELD IN MEMORY - TRUNCATED FOR LOGS]"} 
                    if isinstance(item, dict) and item.get("type") == "image_url" 
                    else item
                    for item in display_content
                ]
            elif isinstance(display_content, str) and "data:image" in display_content:
                display_content = "[📸 IMAGE HELD IN MEMORY - TRUNCATED FOR LOGS]"

            logger.output(f"\n✅ Tool {chunk.name} returned: {display_content}\n")
        yield _sse_event("update", {"chunk": sanitize_for_streaming(chunk), "metadata": sanitize_for_streaming(metadata)})
    logger.output("\n\n✅ Execution Complete. Can extract trajectory now.")
    return {'status': "completed"}

    # return StreamingResponse(
    #     _generator(),
    #     media_type="text/event-stream",
    #     headers={
    #         "Cache-Control": "no-cache",
    #         "X-Accel-Buffering": "no",
    #         "Connection": "keep-alive",
    #     },
    # )

